# 03 · RadixAttention：用一棵树自动复用前缀 KV

> 对应进阶 [03 Radix tree 内部实现](../advanced/03_radix_cache_internals.md)。
> 源码：`python/sglang/srt/mem_cache/radix_cache.py`（913 行）
> 先读目录契约：`python/sglang/srt/mem_cache/README.md`（84 行，必读）

---

## 3.1 为什么是 "Radix" 而不是 "Prefix" 缓存

朴素的 prefix cache 用哈希表存 `前缀字符串 → KV`，问题很明显：

- 只能匹配"完整前缀"，`[A B C]` 和 `[A B C D]` 要存两份
- 前缀长度不固定，表会爆炸

**Radix tree（基数树 / 压缩前缀树）** 把 token 序列组织成一棵树，每条边存一段
token 序列而不是单个 token：

```
                    root
                     │ key=[S1 S2 S3 S4]
                     ▼
                 [S1 S2 S3 S4]              ← 节点：一段 token + 对应的 KV 索引
                ╱            ╲
    key=[A1 A2]              key=[B1 B2]
        ▼                        ▼
    [A1 A2]                  [B1 B2]
        │
    key=[A3]
        ▼
      [A3]
```

**关键性质**：从 root 到任意节点的路径，就是一段可以被复用的前缀。
所以"最长前缀匹配"就是"从 root 往下尽量走"。复杂度 O(路径长度)，与表大小无关。

而且**天然的自动共享**：`[S1..S4, A1 A2]` 和 `[S1..S4, B1 B2]` 共享父节点，
`S1..S4` 的 KV 只存一份。

---

## 3.2 节点里存什么

```python
# python/sglang/srt/mem_cache/radix_cache.py:256
class TreeNode:
    counter = 0

    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None                    # 这一段 token 序列
        self.value: Optional[torch.Tensor] = None    # 对应的 KV 池槽位索引
        self.lock_ref = 0                            # >0 表示正在被请求使用，不可驱逐
        self.last_access_time = time.monotonic()
        self.hit_count = 0
        self.host_value: Optional[torch.Tensor] = None   # HiCache: host 侧副本（第 06 章）
        self.priority = priority
```

三件最重要的事：

1. **`key`** —— 这段路径上的 token ids。注意相邻节点间没有冗余：如果两个节点
   可以合并，就会合并（见 3.4 的插入逻辑）。
2. **`value`** —— 不是 KV 数据本身，而是 **KV 池里的槽位索引**（int64 tensor）。
   真正的 KV 放在 `memory_pool.py` 的大 tensor 里，节点只是"目录"。这是关键设计：
   KV 数据只有一份，树只是索引。
3. **`lock_ref`** —— 引用计数。正在被 running 请求使用的节点不能驱逐。

`host_value` 是 HiCache 用的（KV 的 CPU 内存副本），第 06 章再讲。

---

## 3.3 一次请求的完整生命周期

这是理解 RadixAttention 的主线，四个 API：

### ① 请求进来：`match_prefix`

```python
# radix_cache.py:396
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    key = params.key.page_aligned(self.page_size)   # 对齐到 page（默认 1，即不切页）
    value, last_node = self._match_prefix_helper(self.root_node, key)
    value = torch.cat(value) if value else ...
    return MatchResult(device_indices=value, last_device_node=last_node, ...)
```

返回的 `device_indices` 就是**已经算好的前缀 KV 在池子里的槽位**。

调度器拿到之后：

```python
# 伪代码，见 scheduler.py handle_generate_request
prefix_indices, last_node = tree_cache.match_prefix(key=req.origin_input_ids)
req.prefix_indices = prefix_indices      # 复用这些槽位
req.extend_input_len = len(input_ids) - len(prefix_indices)   # 只需算剩下的
```

**`extend_input_len` 就是实际要算的 token 数**。命中 1000 tokens 前缀，
prefill 的计算量直接减少 1000 token 的活。

> 命中率高的时候，TTFT 能降一个数量级。这是 SGLang 最出名的优化。

`_match_prefix_helper` 还会顺手做一件重要的事：**在匹配终止处分裂节点**。
如果匹配停在某个节点的中间（`key=[A1 A2 A3]` 但只命中到 `A1`），
就把这个节点砍成 `[A1]` 和 `[A2 A3]` 两部分。这样下次匹配就能精确停在边界上。

### ② 请求在跑：`inc_lock_ref` / `dec_lock_ref`

```python
# radix_cache.py:709
def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)     # 从"可驱逐"移走
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent
    return IncLockRefResult(delta=delta)
```

从匹配到的节点**一路向上到 root** 全部 `+1`。因为这条路径上的每一段 KV
都在被这个请求使用，都不能被驱逐。

`delta` 是"可驱逐容量减少量"，调度器用它来判断还剩多少余量。

### ③ 请求结束：`cache_finished_req` / `cache_unfinished_req`

请求结束（或 chunked prefill 的一个 chunk 做完）时，把新算出来的 KV **插入树中**：

```python
# radix_cache.py:456 insert / 478 cache_finished_req
def insert(self, params: InsertParams) -> InsertResult:
    key = params.key.page_aligned(self.page_size)
    value = params.value
    prefix_len, last_node = self._insert_helper(self.root_node, key, value, priority, chunked)
    return InsertResult(prefix_len=prefix_len, last_device_node=last_node)
```

`_insert_helper` 的逻辑（radix_cache.py:788）是经典的：

```
从 root 开始，逐段比较：
  1. 如果 token 完全匹配某个已有子节点 → 往下走，value 已存在（复用，不重复存）
  2. 如果部分匹配 → 分裂该子节点，创建中间节点
  3. 如果没有匹配 → 创建新节点，挂上 value
```

**注意一个细节**：如果路径上已有节点，插入时**不会重复存 KV**——因为已有的
槽位就是同一个前缀的 KV，直接复用。

### ④ 显存不够：`evict`

```python
# radix_cache.py:643
def evict(self, params: EvictParams) -> EvictResult:
    leaves = list(self.evictable_leaves)
    eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        # 父节点变成叶子且没被引用 → 也可以被驱逐了，推进堆里
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (…, x.parent))
        self.kv_events.record_remove(x)
    return EvictResult(num_tokens_evicted=num_evicted)
```

三个要点：

1. **只驱逐叶子**。因为父节点的 KV 是子节点共享的，父节点被驱逐会破坏子节点。
2. **最小堆按优先级**。默认 `lru`（`last_access_time` 最早的先走），
   还有 `lfu` / `slru` / `priority` / `tlru`（`--radix-eviction-policy`）。
3. **级联**：叶子删掉后，如果父节点变成新叶子且 `lock_ref == 0`，就把它也推进堆。

---

## 3.4 缓存的三种状态与 `lock_ref`

理解 `lock_ref` 是理解 SGLang 显存管理的钥匙：

```
KV 池的槽位分三类：

  ① 正在被 running 请求使用          → lock_ref > 0，不可驱逐
  ② 在 radix tree 里但没人用          → lock_ref == 0，可驱逐（evictable）
  ③ 完全空闲                          → 未分配（available）

    available_size + evictable_size + protected_size ≈ max_total_num_tokens
```

调度器每次决定要不要收新请求时，看的就是这几个数：

```python
# scheduler/schedule_policy.py
def rem_total_tokens(self):
    return self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size()
```

**能收新请求的余量 = 空闲槽位 + 可驱逐槽位。**

> 这就是为什么 `cache_hit_rate` 高的时候吞吐也高：命中 → 不用新分配 KV →
> 腾出余量给更多并发请求。

---

## 3.5 谁在维护这棵树（"自动"是什么意思）

**没有任何用户 API 需要调用它。** 全流程在 `Scheduler` 里自动发生：

| 时机 | 调用 | 位置 |
|---|---|---|
| 请求进入 `handle_generate_request` | `match_prefix` | `scheduler.py:2750` |
| 请求加入调度队列 | `inc_lock_ref` | `schedule_policy.py:979 _req_inc_lock_ref` |
| chunked prefill 每个 chunk 结束 | `cache_unfinished_req` | `scheduler.py:3516 stash_chunked_request` |
| 请求结束（`process_batch_result`） | `cache_finished_req` | scheduler 结果处理 |
| 显存不足 | `evict` | `schedule_policy.py` 的预算检查 |

所以你只是启动了服务，就自动获得了前缀复用。这是 SGLang 与 vLLM 早期版本
最大的设计差异之一（vLLM 需要显式 `--enable-prefix-caching` 且 API 上更保守）。

---

## 3.6 什么时候该关掉它

`--disable-radix-cache`。什么时候关：

| 场景 | 原因 |
|---|---|
| 每个请求前缀都完全不同 | 命中率 ~0，只剩开销 |
| 延迟极度敏感、请求极短 | radix 查找 + 锁 + 分裂有 CPU 开销，短请求下占比高 |
| 需要严格确定性 | 缓存命中会让同样的输入走不同的计算路径，浮点结果可能微异（见 `--enable-deterministic-inference`） |

**测法**：同一个 workload 跑两次，对比 `cache_hit_rate` 和 TTFT P99。

---

## 3.7 一个常见的误解：命中缓存 ≠ 结果完全一样

命中前缀 KV 后，被跳过的那些 token **不会重新做 prefill**，但 attention 的
其余部分（后续 token 对前缀的注意力）仍然要算。

所以：
- **省掉的是**：这些 token 的 QKV 投影 + 它们的注意力计算
- **没省掉的是**：后续 token 对它们的 attention（还是要读它们的 KV）

数学上结果一样（KV 是确定的），但**数值上可能有极微小差异**，因为
"先算好再读"和"一次算完"的累加顺序不同。这就是 `--enable-deterministic-inference`
存在的原因。

---

## 3.8 本章自检

1. 为什么 radix tree 只驱逐叶子节点？
2. `lock_ref` 从匹配节点到 root 全部 +1，为什么？
3. `value` 存的是 KV 数据还是槽位索引？为什么这样设计？
4. "能收新请求的余量" 由哪三项组成？
5. 你的 workload 命中率 5%，你应该做什么？

---

下一页 → [04 调度器](04_scheduler.md)

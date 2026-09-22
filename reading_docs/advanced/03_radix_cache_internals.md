# 进阶 03 · Radix Cache 内部实现

> 对应入门 [03 RadixAttention](../beginner/03_radix_attention.md)。
> 源码：`python/sglang/srt/mem_cache/radix_cache.py`（913 行）、
> `mem_cache/evict_policy.py`、`mem_cache/base_prefix_cache.py`

---

## 1. `RadixKey`：token 序列的封装

在 `TreeNode` 之前，先看它存的东西：

```python
# radix_cache.py:59
class RadixKey:
    def __init__(self, token_ids, extra_key=None, is_bigram=False, ...):
        ...
    def _raw_len(self) -> int: ...
    def raw_token_ids(self) -> array: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> RadixKey: ...
    def page_aligned(self, page_size: int) -> RadixKey: ...
    def maybe_to_bigram_view(self, is_eagle: bool) -> tuple: ...
    def match(self, other, page_size: int = 1) -> int: ...
    def match_at(self, other, offset: int, page_size: int = 1) -> int: ...
    def child_key(self, page_size: int = 1): ...
    def hash_page(self, start, end, prior_hash=None) -> str: ...
```

四个设计点：

### ① `extra_key`：命名空间隔离

从 `match_prefix` 的 docstring 抄：

> Entries that share identical leading token ids but have *different* `extra_key`
> values are intentionally kept disjoint and never share prefix nodes. This is useful to:
> - Isolate KV cache lines for different LoRA / adapter IDs.
> - Separate requests that intentionally should not share state (e.g., different
>   sampling salt, cache version, or retrieval augmentation context).

**同一个 token 序列，不同的 `extra_key` 不会共享 KV。** 这是 LoRA 多适配器场景
正确性的基础——base model 相同但 adapter 不同，KV 不能混用。

### ② `maybe_to_bigram_view`：EAGLE 投机解码的特殊处理

EAGLE 的 draft 模型以 bigram（相邻 token 对）为输入。这个方法把 token 序列
转成 bigram 视图，避免在投机路径上做额外的拷贝。

### ③ `page_aligned`：页对齐

```python
def page_aligned(self, page_size: int) -> RadixKey:
    # 截断到 page_size 的整数倍
```

`--page-size > 1` 时，匹配和插入都以页为单位。**对齐是必须的**——
否则无法用 `free_segment` 的 `free_index[::page_size]` 技巧（见第 08 章）。

### ④ `hash_page`：给 KV 事件和存储用的页哈希

注意 `prior_hash` 参数——**链式哈希**。每页的哈希依赖前一页，这样：

- 相同前缀 → 相同哈希（可跨实例去重）
- 哈希值本身就能证明前缀相同

`hash_value` 是**懒计算**的（`insert` 时不算，只在发事件时算）：

```python
# _insert_helper 里的注释
# Hash will be computed lazily during event emission
self.kv_events.record_store(new_node)
```

---

## 2. `TreeNode`：一张完整的字段表

```python
# radix_cache.py:256
class TreeNode:
    counter = 0

    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)      # 子节点，key 是 child_key
        self.parent: TreeNode = None
        self.key: RadixKey = None                   # 这一段 token
        self.value: Optional[torch.Tensor] = None   # KV 槽位索引
        self.lock_ref = 0                           # 引用计数，>0 不可驱逐
        self.last_access_time = time.monotonic()    # LRU 用
        self.creation_time = time.monotonic()       # FIFO 用
        self.hit_count = 0                          # LFU / SLRU 用
        self.host_ref_counter = 0                   # HiCache: host 侧引用
        self.host_value: Optional[torch.Tensor] = None   # host 槽位索引
        self.write_through_pending_id: Optional[int] = None
        self.hash_value: Optional[List[str]] = None
        self.event_hash_value: Optional[List[str]] = None
        self.priority = priority
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
```

### 三个 property

```python
@property
def evicted(self):
    return self.value is None

@property
def backuped(self):
    return self.host_value is not None
```

### `__lt__`：给 heapq 用的

```python
def __lt__(self, other):
    return self.last_access_time < other.last_access_time
```

**注意**：`evict()` 里显式构造 `(priority, node)` 元组来避开这个默认比较
（因为策略可能不只依赖时间），`__lt__` 只是给别处兜底。

---

## 3. `_match_prefix_helper`：LRU 触碰与节点分裂

```python
# radix_cache.py:729
def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    access_time = time.monotonic()
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time          # ← LRU 触碰，零额外遍历
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            # 部分匹配 → 分裂
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
    return value, node
```

**四个要点**：

1. **`child.last_access_time = access_time` 在匹配路径上顺手做了 LRU 更新**。
   不需要额外的遍历——这是"把元数据更新融进主路径"的典型手法。

2. **`access_time` 在循环外取一次**，不是每次循环重新 `time.monotonic()`。
   同一个请求匹配到的所有节点共享同一个时间戳，这样它们的相对顺序稳定。

3. **`prefix_len < len(child.key)` 触发分裂**。存储是 `[a,b,c,d]`，只匹配到 `[a,b]`，
   就砍成 `[a,b]` + 子节点 `[c,d]`。**匹配边界变成真实节点**，
   下次匹配就能精确停在这里。

4. **返回 `value` 列表**（每个节点的 `value` 切片），最终 `torch.cat` 成一个 tensor。

### `_split_node`：两个 `.clone()` 是必须的

```python
# radix_cache.py:753
def _split_node(self, key, child, split_len):
    new_node = TreeNode()
    new_node.parent = child.parent
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()      # ← clone
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()         # ← clone
    ...
    # 同时分裂 hash_value
    split_node_hash_value(child, new_node, split_len, ...)
```

**为什么必须 clone？** 两个节点必须各自拥有自己的槽位索引数组切片。
不 clone 的话是同一块内存的视图，后续对其中一个做 in-place 修改会污染另一个。

`hash_value` 也要分裂——`split_node_hash_value` 把页哈希列表按 `split_len` 切开。

---

## 4. `_insert_helper`：什么时候不重复存

```python
# radix_cache.py:788
def _insert_helper(self, node, key, value, priority, chunked):
    while len(key) > 0:
        child_key = key.child_key(self.page_size)
        if child_key in node.children:
            child = node.children[child_key]
            prefix_len = child.key.match(key, ...)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                node = new_node
                break
            else:
                node = child
                key = key[prefix_len:]
                value = value[prefix_len:]
        else:
            break

    if len(key):
        new_node = TreeNode(priority=priority)
        new_node.parent = node
        new_node.key = key
        new_node.value = value.clone()
        self._inc_hit_count(new_node, chunked)
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
        self._update_leaf_status(node)
        self._update_leaf_status(new_node)
        # Hash will be computed lazily during event emission
        self.kv_events.record_store(new_node)
```

### 关键性质：路径上已有的节点不重复存

如果走过了已有节点，说明那段 KV **已经在池子里**（这就是它们的 `value`），
不需要重新分配、重新写入。**插入只创建"新的尾巴"**。

### `_inc_hit_count` 对 chunked 请求跳过

```python
# :783-787
def _inc_hit_count(self, node, chunked):
    if chunked:
        return          # chunked prefill 会给自己刚创建的节点刷 hit_count
    ...
```

**原因**：chunked prefill 每处理一块就把这块插进树，下一块匹配时又会命中自己
刚插入的节点。如果不跳过，`hit_count` 会被自己刷爆，污染 LFU / SLRU 的统计。

---

## 5. `cache_finished_req` / `cache_unfinished_req`：两条接入路径

```python
def cache_finished_req(self, req: Req, is_insert: bool = True, *, owned_kv_len: int):
    """请求结束时把新算的 KV 挂进树。"""

def cache_unfinished_req(self, req: Req, chunked=False):
    """chunked prefill 的每个 chunk 结束时挂一次。"""
```

区别：

| | `cache_finished_req` | `cache_unfinished_req` |
|---|---|---|
| 时机 | 请求完全结束 | 每个 chunk 结束 |
| `chunked` 标志 | `False` | `True` |
| `hit_count` | 正常递增 | 跳过（见上） |
| 节点状态 | 可以立即被驱逐 | 请求还在跑，需要 `lock_ref` |

调度器的调用点：

- 结束：`process_batch_result_*` 里
- chunk：`scheduler.py:3516 stash_chunked_request` → `cache_unfinished_req`

---

## 6. `evict`：堆 + 级联

```python
# radix_cache.py:643
def evict(self, params: EvictParams) -> EvictResult:
    if self.disable:
        return EvictResult()
    start_time = time.perf_counter()
    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        # Tree values are page-aligned copies of a kv row: page-exact segment.
        self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
        self.kv_events.record_remove(x)
    self.update_eviction_metrics(num_evicted, start_time)
    return EvictResult(num_tokens_evicted=num_evicted)
```

### 四个设计点

**① `free_segment(x.value, start_pos=0)`**：
注释说 "Tree values are page-aligned copies of a kv row: page-exact segment."
因为节点的 `value` 已经是页对齐的连续段，可以直接走 `free_segment` 的
免 `torch.unique` 快路径（第 08 章）。

**② 父节点惰性入堆**：

```python
if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
    heapq.heappush(eviction_heap, (...))
```

**父节点只在它的最后一个子节点消失后才入堆。** 所以堆里永远只有真正的叶子。
比"一开始把所有节点都放进堆再过滤"高效得多。

**③ `lock_ref == 0` 的检查**：被引用（正在跑的请求用着）的父节点不能驱逐。

**④ `evictable_leaves` 是维护好的集合**，不是每次重新遍历树：

```python
def _update_leaf_status(self, node):
    # node 是叶子-eligible 当且仅当：未驱逐 且 lock_ref == 0 且没有活着的子节点
```

---

## 7. `inc_lock_ref` / `dec_lock_ref`：为什么是整条路径

```python
# radix_cache.py:709
def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent
    return IncLockRefResult(delta=delta)
```

**从匹配节点一路 `+1` 到 root**。为什么？

因为**驱逐是级联的**（`evict` 里父节点会在最后一个子节点消失后入堆）。
如果只锁住叶子节点，它的祖先仍可能被驱逐——那会破坏整条路径的 KV。

所以锁的是**整条前缀路径**。

`delta` 是"可驱逐容量减少量"，调度器用它判断还有多少余量：

```python
def dec_lock_ref(self, node, params=None):
    while node != self.root_node:
        if node.lock_ref == 1:
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
            delta += len(node.key)
        node.lock_ref -= 1
        ...
```

`dec_lock_ref` 里还有一个一致性断言：

```python
if node.parent is None:
    assert node is self.root_node, "This request holds the node from another tree"
```

**防止跨树的节点引用**——多 tree 场景（如 `--enable-session-radix-cache`）下的防御。

---

## 8. 驱逐策略：5 种 + 2 种高级

`mem_cache/evict_policy.py`：

| 类 | `get_priority` 返回 | 含义 |
|---|---|---|
| `LRUStrategy` | `node.last_access_time` | 默认 |
| `LFUStrategy` | `(hit_count, last_access_time)` | 频率优先，同频看新旧 |
| `FIFOStrategy` | `creation_time` | |
| `MRUStrategy` | `-last_access_time` | 最近用过的先走 |
| `FILOStrategy` | `-creation_time` | |
| `PriorityStrategy` | `(node.priority, last_access_time)` | 低优先级先驱逐 |
| `TLRUStrategy` | `(-1 if tel_safe else 0, last_access_time)` | Tail-Optimized LRU |
| `SLRUStrategy` | `(is_protected, last_access_time)` | 两段式 |

`--radix-eviction-policy` 的 choices：`["lru", "lfu", "slru", "priority", "tlru"]`。

### TLRU：有意思的一个

Tail-Optimized LRU（arXiv:2510.15152）。核心洞察：

> 一段对话只需要 `L + Q_hat - threshold` 个 token 常驻，就能让下一次 prefill
> 落在 TTFT 预算内。超出的部分是 "TEL-safe"，可以标记为"无限旧"优先排空。

实现上用 `node._tlru_history_len`（一个**只增不减**的高水位标记）而不是当前常驻量。

### SLRU：两段式

按 `hit_count >= protected_threshold`（默认 2）分段 —— 段 0（probation）永远
在段 1（protected）之前被排空。防止"偶然命中一次"的节点长期占据空间。

---

## 9. 与调度器的接口

Radix cache 暴露给调度器的核心查询：

```python
# radix_cache.py
def evictable_size(self):
    return self.evictable_size_

def protected_size(self):
    return self.protected_size_
```

调度器侧的用法（`schedule_policy.py`）：

```python
def rem_total_tokens(self):
    return (self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size())
```

**能接新请求的余量 = 空闲槽位 + 可驱逐槽位。**

### 跨 tree 的操作

`--enable-session-radix-cache` 引入会话级引用：

```python
# arg_groups/fields/memory.py
enable_session_radix_cache: A[bool,
    "Track per-session references on UnifiedRadixCache KV: eviction consumes "
    "unreferenced entries before referenced ones, and closing a session only "
    "dereferences its KV."]
```

会话持有的 KV 在驱逐时排最后。

---

## 10. 调优视角

| 现象 | 参数 | 说明 |
|---|---|---|
| 命中率低、开销却存在 | `--disable-radix-cache` | 无共享前缀的负载下纯成本 |
| 长对话被反复驱逐 | `--radix-eviction-policy tlru` | 保护 chat 尾部 |
| 少量热点节点长期占据 | `--radix-eviction-policy slru` | 两段式，冷启动快 |
| 命中率还行但延迟抖动 | `--schedule-policy lpm` | 让命中率变成实际收益（需要调度器配合） |
| 需要跨实例共享 KV | `--hicache-storage-backend` | 见第 09 章 |

### 一个反直觉的观察

**`cache_hit_rate` 高 ≠ 吞吐高。** 命中率只衡量"有多少 token 不需要重算"，
但真正决定吞吐的是**可驱逐容量**（`evictable_size`）。

极端情况：一个巨型 system prompt 被 100% 命中，但它占满了 KV 池，
导致并发只有 5。此时命中率 99%，吞吐却很低。

**正确的看板**：`cache_hit_rate` + `kv_evictable_tokens` + `token_usage` 三个一起看。

---

## 11. 自检

1. `extra_key` 解决什么问题？不用它会出什么错？
2. `_split_node` 里的两个 `.clone()` 去掉会怎样？
3. `_match_prefix_helper` 里的 LRU 触碰为什么可以"免费"？
4. 为什么 `inc_lock_ref` 要从匹配节点一路锁到 root？
5. `evict` 里父节点为什么要惰性入堆？
6. chunked prefill 为什么跳过 `_inc_hit_count`？
7. 命中率高但吞吐低，可能是什么原因？

---

下一页 → [04 调度器内部实现](04_scheduler_internals.md)

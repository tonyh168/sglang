# 进阶 09 · HiCache 与多级存储

> 对应入门 [06 显存与内存池](../beginner/06_memory_and_kv_pool.md) 的 HiCache 一节。
> 源码：`mem_cache/hiradix_cache.py`、`mem_cache/memory_pool_host.py`、
> `mem_cache/pool_host/`、`mem_cache/storage/`、`mem_cache/hicache_storage.py`

---

## 1. 三层存储的定位

```
L1  device pool   （GPU HBM）      带宽 ~3 TB/s    容量 ~几十 GB    每 GB ~$20
L2  pool_host     （CPU DRAM）     带宽 ~50 GB/s   容量 ~几百 GB    每 GB ~$3
L3  storage       （SSD / 远端）   带宽 ~GB/s      容量 ~TB         每 GB ~$0.1
```

**复用半径 vs 传输开销**是这个设计的全部矛盾：

| 层 | 能复用的时间尺度 | 一次命中的代价 |
|---|---|---|
| L1 | 秒级（并发请求之间） | 0 |
| L2 | 分钟级（连续会话） | H2D 传输 |
| L3 | 小时/天级（跨实例、跨重启） | 网络/磁盘 + H2D |

**L2 的收益来自"对话历史 + system prompt"这类跨请求复用的内容。**

---

## 2. `HiRadixCache` 的构造

```python
# hiradix_cache.py:83
class HiRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        ...
```

### host pool 的类选择

```python
# 按 device pool 的类型选 host pool
if isinstance(device_pool, MHATokenToKVPool):
    host_pool_cls = get_mha_host_pool_cls(...)
elif isinstance(device_pool, MLATokenToKVPool):
    host_pool_cls = MLATokenToKVPoolHost(..., dcp_size=..., dcp_rank=...)
# DSA / MSA 由 attach helper 后续填充
```

**限制**（:119）：

```python
raise ValueError("HiRadixCache only supports MHA, MLA, DSA, and MSA models")
```

### 写策略决定驱逐路径

```python
# hiradix_cache.py:1206
def evict(self, params: EvictParams) -> EvictResult:
    if self.cache_controller.write_policy == "write_back":
        num_evicted = self._evict_write_back(num_tokens)
    else:
        num_evicted = self._evict_write_through(num_tokens)
```

---

## 3. 两种写策略的对比

### `write_through`（默认）

```python
# _evict_write_through，:1230
# "Nothing is staged to host during eviction, so this is a plain on-the-fly pass."
```

**备份是"插入时"就 eager 做的**，驱逐时什么都不用做。

**优点**：驱逐路径极快（驱逐发生在 decode 的热路径上）。
**缺点**：每次插入都有 H2D 传输开销。

### `write_back`

```python
# _evict_write_back，:1248
# backup 发生在驱逐时：self.write_backup(x, write_back=True)
# 暂存到 staged 列表，之后批量 flush
```

源码注释：

> this path will be deprecated in the future.

**写策略的工程结论**：`write_through`（默认）是推荐路径。
`write_back` 虽然平时零开销，但把成本堆到了最不该卡的驱逐路径上。

---

## 4. `write_through` 的连续性不变式

这是 HiCache 里最重要的一个约束：

```python
# hiradix_cache.py:857 write_backup
# Backup invariant (for write-through mode): backed-up nodes must form a
# contiguous prefix from root — no gaps.  Skip if parent isn't backed up yet;
if not write_back and (node.parent != self.root_node and not node.parent.backuped):
    return 0
host_indices = self.cache_controller.write(device_indices=node.value, node_id=node.id, ...)
if host_indices is None:
    self.evict_host(len(node.value))
    host_indices = self.cache_controller.write(...)   # 重试一次
```

### 这个不变式换来了什么

**host 侧的 `match_prefix` 退化成一次简单的向下遍历，不需要处理碎片。**

如果允许空洞，host 上就要维护一棵有自己的空洞的树，或者要能跳过空洞继续匹配——
复杂度和错误概率都大幅上升。

### 代价

**host 上的内容总是 device 内容的"前缀子集"。** 如果一个 node 的父节点还没备份，
它就备份不了，只能等父节点先被写。

### `write_through_threshold`

```python
# :207-209
write_through_threshold = 1 if write_through else 2
```

`write_back` 阈值是 2——因为 write_back 的备份链更长。

### 分裂时的 ack 重定向

`_track_write_through_node` / `_finish_write_through_ack`（:889-932）维护节点上的
`write_through_pending_id`。

**问题**：一个正在 DMA 到 host 的 node，如果中途被 `_split_node` 分裂了怎么办？

```python
# _replace_pending_write_through_node，:893-920
# 把 pending ack 重新指向正确的节点
```

**所以 in-flight 的 DMA 完成后，credit 能落到正确的节点上。**
这是一个容易漏掉的并发场景。

### L2 → L3 的链式备份

```python
# write_backup_storage，:933
# 串起 L2 → L3
# 用 _concat_split_chain（:960+）重建分裂前的内容
```

---

## 5. Host 池的内存预算

### `HostKVCache`（`pool_host/base.py:122`）

```python
HICACHE_HOST_MEMORY_RESERVE_BYTES = 10 GiB     # :30
```

### 三步分配

```python
# :77-95 host_memory_budget_bytes
# 1. 从 cgroup 限制（memory.max / memory.high，或 v1 limits）出发
# 2. 扣掉 10 GiB 预留
# 3. 按同机 co-located 的 rank 数均分
```

`--hicache-host-memory-fraction`（默认 0.8）只在既没设 `--hicache-ratio`
也没设 `--hicache-size` 时生效。

**为什么要扣 10 GiB？** 因为 host 上还有页面缓存、Python 解释器、NCCL buffer、
以及各种临时张量。不预留的话会把机器打爆。

### PP 组内取 MIN

```python
# :98-119 sync_fixed_hicache_size
# all-reduce MIN across the PP group
```

**每个 PP stage 必须暴露相同的 token 容量**——否则各 stage 的 KV 视图不一致，
前缀匹配会出错。

---

## 6. 主机池的布局（`--hicache-mem-layout`）

| 值 | 说明 |
|---|---|
| `layer_first` | 按层排布 |
| `page_first`（默认） | 按页排布 |
| `page_first_direct` | 按页 + 直接 IO |
| `page_first_kv_split` | K/V 分开 |
| `page_head` | 页头布局 |

**`page_first` 是默认**，因为它让 H2D 传输能按页粒度做（一次传输一整页，
不需要跨层 gather）。

### IO 后端（`--hicache-io-backend`）

| 值 | 说明 |
|---|---|
| `direct` | 直接 `cudaMemcpy` |
| `kernel`（默认） | 用自定义 kernel 做传输，可以利用 SM 做并行的分片拷贝 |
| `kernel_ascend` | NPU 版本 |

**`kernel` 更快的原因**：`cudaMemcpy` 是单流的，而自定义 kernel 可以用多个
SM 并行搬运，还能和计算重叠。

---

## 7. L3 存储后端

`mem_cache/storage/` 下的目录：

```
aibrix_kvcache, eic, file, flexkv, hf3fs, lmcache, mmap,
mooncake_store, nixl, npu_memcache, shm, sim_storage, simm,
tensorcast_store, umbp
```

`--hicache-storage-backend` 的 choices 包括：
`file`、`sim`、`mooncake`、`npu_memcache`、`hf3fs`、`nixl`、`aibrix`、
`dynamic`、`eic`、`simm`、`mori`、`shm`、`tensorcast`。

### 常见的三个

| 后端 | 场景 |
|---|---|
| `file` | 本地 SSD，单机场景，最简单 |
| `nixl` | NVIDIA 的传输库，跨节点 |
| `mooncake` | 国内常用的 KV store，也是 PD 分离的默认后端 |
| `hf3fs` | 高性能分布式文件系统 |

### 接口

```python
# hicache_storage.py
class HiCacheStorage(abc.ABC):
    def get(self, key, target_location=None): ...
    def set(self, key, value, target_location=None): ...
    def exists(self, key): ...
    ...
```

**key 用 `RadixKey.hash_page` 的链式哈希**（第 03 章）——
这样不同实例算出同样的 key，天然支持跨实例去重。

---

## 8. 预取机制

这是 L3 能用起来的关键。

### 流程

```python
# hiradix_cache.py
def query_storage_hit_length(self, ...): ...     # :1486 查 L3 命中长度
def init_load_back(self, ...): ...               # :1464 初始化回载
def load_back(self, ...): ...                    # :1391 执行回载
def check_prefetch_progress(self, ...): ...      # :1638 检查进度
def _handle_prefetch_result(self, ...): ...      # :1673 处理结果
def _revoke_pending_prefetch(self, ...): ...     # :1624 撤销
```

### 预取策略（`--hicache-storage-prefetch-policy`）

| 值 | 含义 |
|---|---|
| `best_effort`（?） | 拿到的部分就直接用 |
| `wait_complete` | 等全部传完再开始 |
| `timeout`（默认） | 超时就用手上有的 |

### 重试

```python
# _retry_storage_prefetch / _prefetch_after_device_hit_loss
```

**为什么需要 `_prefetch_after_device_hit_loss`？**

场景：请求在 device 上命中了，但请求处理期间那些 node 被驱逐了。
此时要重新发起预取。

**这是一个真实的竞态**——预取是异步的，而驱逐发生在热路径上。

### 调度器侧的接入

```python
# scheduler.py:3609
def _process_hicache_events(self):
    ...
```

在 `get_next_batch_to_run` 开头调用（:3625）。

```python
# :3114
def _prefetch_kvcache(self, req, storage_hit_end=None): ...
# :3165
def _process_storage_prefetch_retries(self): ...
```

**请求进 waiting_queue 时就开始预取**，而不是等被调度时——
这样 H2D 传输能与前面的请求重叠。

---

## 9. 完整配置示例

### 场景 A：多轮对话，长 system prompt

```bash
--enable-hierarchical-cache \
--hicache-ratio 2.0 \
--hicache-write-policy write_through \
--hicache-mem-layout page_first \
--hicache-io-backend kernel
```

**预期**：host 池是 device 池的 2 倍，跨请求复用 system prompt 和历史。

### 场景 B：RAG，同一份文档被问多个问题

```bash
--enable-hierarchical-cache \
--hicache-storage-backend file \
--hicache-storage-prefetch-policy timeout \
--hicache-storage-prefetch-retry-poll-interval 8
```

**预期**：文档的 KV 从 SSD 预取，跨进程重启也能命中。

### 场景 C：PD 分离的 decode 节点

```bash
--disaggregation-mode decode \
--hicache-host-memory-mode buffer_only \
--hicache-storage-backend nixl
```

`buffer_only` 模式下 host 只是 GPU 和 storage 之间的**中转缓冲**，
不是持久的缓存层。**要求必须有 storage backend。**

`hicache_hook.py`（:217-236）的校验：`buffer_only` **不能和 `write_back` 一起用**，
且在 decode 节点上有额外限制。

---

## 10. 参数速查

| Flag | 默认 | 说明 |
|---|---|---|
| `--enable-hierarchical-cache` | False | 总开关 |
| `--hicache-host-memory-mode` | `cache` | `cache` / `buffer_only` |
| `--hicache-ratio` | 2.0（cache）/ 1.2（buffer_only） | host:device 比例 |
| `--hicache-size` | 0 | 直接指定 GB，覆盖 ratio |
| `--hicache-host-memory-fraction` | 0.8 | 只在 ratio 和 size 都没设时生效 |
| `--hicache-write-policy` | `write_through` | `write_back` / `write_through` / `write_through_selective` |
| `--hicache-io-backend` | `kernel` | `direct` / `kernel` / `kernel_ascend` |
| `--hicache-mem-layout` | `page_first` | 见上表 |
| `--hicache-storage-backend` | None | 见上表 |
| `--hicache-storage-prefetch-policy` | `timeout` | `best_effort` / `wait_complete` / `timeout` |
| `--hicache-storage-prefetch-retry-poll-interval` | 8 | 重试前等几个调度轮 |

---

## 11. 什么时候 HiCache 是负收益

| 场景 | 原因 |
|---|---|
| 请求前缀完全不共享 | 命中率 ~0，只剩 H2D 传输和 host 内存占用 |
| device 池本来就够 | host 层的管理开销（radix tree 第二份、引用计数、ack）纯成本 |
| 对 TPOT 极度敏感 | `write_through` 的插入期 H2D 会占用 PCIe 带宽 |
| 单请求超长、无跨请求复用 | 没有复用半径 |

**测法**：同一个 workload 跑 two-run 对比，看 `hicache_host_used_tokens`
和 TTFT P99 的变化。

> **注意**：HiCache 的收益主要体现在 **TTFT**（因为省掉了重新 prefill），
> 对 **TPOT 没有直接收益**，反而可能因为 PCIe 竞争略微劣化。

---

## 12. 自检

1. `write_through` 的"连续前缀"不变式换来了什么？代价是什么？
2. 为什么 `write_back` 路径标注了"will be deprecated"？
3. host 内存为什么要扣 10 GiB？
4. PP 组内为什么要对 host 容量取 MIN？
5. `_prefetch_after_device_hit_loss` 处理的是什么竞态？
6. 预取为什么在请求进 waiting_queue 时就开始，而不是等调度？
7. HiCache 主要优化 TTFT 还是 TPOT？为什么？

---

下一页 → [10 并行策略内部实现](10_parallelism_internals.md)

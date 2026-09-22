# 06 · 显存怎么分：内存池、分配器与 HiCache

> 对应进阶 [08 内存池与分配器](../advanced/08_kv_pool_and_allocators.md)、
> [09 HiCache 多级存储](../advanced/09_hicache_and_multitier.md)。
> **先读 `python/sglang/srt/mem_cache/README.md`（84 行）**，它把目录分层讲得很清楚。

---

## 6.1 先看这张图

`mem_cache/README.md` 里的分层图，值得背下来：

```
        scheduler / model_runner / attention backend
                          │
                          ▼
        allocation.py                 每批的分配策略
                          │
                          ▼
        hybrid_cache/                 多池路由（layer_id → pool）
                          │
                          ▼
        allocator/                    "给我 N 个槽位"  (need_size → indices)
                          │ 持有引用
                          ▼
        pool/ (device, L1)  --hicache-->  pool_host/ (host, L2)  -->  storage/ (L3)
        (layer_id, indices)                device_indices <-> host_indices
             <-> tensor

        radix cache          独立一轴：决定"保留什么、驱逐什么"
```

**记住三个概念的分工**：

| 层 | 关心什么 | 输入 → 输出 |
|---|---|---|
| `allocator/` | 哪些槽位是空闲的 | `need_size` → `indices` |
| `pool/` | KV 的物理布局 | `(layer_id, indices)` ↔ tensor |
| radix cache | 保留/驱逐策略 | token 前缀 → node |

---

## 6.2 显存被谁占了

SGLang 启动日志里最关键的一行：

```
max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384,
max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

显存的四个消费者：

```
总显存 = 模型权重 + KV 池 + CUDA Graph 缓冲区 + 激活（activation）
         └──────┬──────┘   └────────┬────────┘   └────────┬────────┘
                │                    │                     │
        mem_fraction_static 决定前两项之和             预留 5–8 GB
```

`--mem-fraction-static` 的定义（官方文档原话）：

```
mem_fraction_static = (模型权重 + KV 池) / GPU 显存容量
```

**调参方法**（来自 `docs/docs/advanced_features/hyperparameter_tuning.mdx`）：

- 看 `available_gpu_mem`：**5–8 GB 是健康值**
- 太高（10–20 GB）→ 提高 `--mem-fraction-static`，把显存还给 KV 池
- 太低 → 有 OOM 风险，降低它
- 简单粗暴法：以 0.01 为步长往上加，直到 OOM

### 容量是怎么算出来的

源码在 `mem_cache/kv_cache_configurator.py` 和 `pool_configurator.py`。模型很简单：

```python
# pool_configurator.py 的文档字符串
available_bytes = max_tokens * coeff + bias
max_tokens = (available_bytes - bias) / coeff
```

两步：

**① 算可用字节**（`kv_cache_configurator.py:2214`）：

```python
available_gpu_memory = get_available_gpu_memory(...)
slack_gb = pre_model_load_memory * (1 - get_schedule().mem_fraction_static)
rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb
return int(rest_memory * (1 << 30))
```

> 注意：测量前会先 `gc.collect()`。因为权重加载期间的临时张量如果还被引用，
> 会把"可用显存"低估好几个数量级。

**② 除以每 token 字节数**（`pool_configurator.py:551`）：

```python
max_total_num_tokens = available_bytes // self._cell_size
max_total_num_tokens = max_total_num_tokens // page_size * page_size   # 对齐到 page
```

`_cell_size` 就是"一个 token 在所有层上的 KV 字节数"。这就是第 01 章那个
`144 KB/token` 的来源。

---

## 6.3 `req_to_token`：整个系统的核心间接表

这是 SGLang（和 vLLM）最巧妙的设计之一。**KV 数据本身是连续的，但每个请求的
KV 在物理上是不连续的。**

```python
# memory_pool.py:293
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len), dtype=torch.int32, device=device
)
self.free_slots = list(range(1, self._alloc_size))
```

一张 `[max_running_requests + 1, max_context_len]` 的表：

```
req_to_token[请求行号][token 位置] = KV 池里的物理槽位号

            位置0  位置1  位置2  位置3  位置4 ...
请求 3   →  [ 517,  518,  1024, 1025,  42,  ... ]
请求 7   →  [ 900,  901,  902,  519,  520,  ... ]
请求 0   →  [  0,    0,     0,    0,    0,  ... ]   ← 填充行，垃圾数据落这里
```

**为什么第 0 行是填充行？** 注释说得很直白：

```python
# +1 padding row at index 0: cuda-graph padded batches default
# req_pool_indices to 0, so dummy reads/writes land here harmlessly.
```

CUDA Graph 需要固定的 batch shape，所以 batch 不足时会**填充到 captured 的 bucket 大小**。
填充出来的假请求全部指向第 0 行，它们的读写就落在垃圾区，不影响真实数据。

**attention kernel 怎么用它**：

```
kernel 拿到：
  req_pool_indices  → 该请求在第几行
  seq_lens          → 序列多长
  然后读 req_to_token[row, 0:seq_len] 得到所有 KV 的物理地址
```

一次间接查表，就把"逻辑上连续的 KV 序列"映射到"物理上散落的页"。

---

## 6.4 KV 池的物理布局

### MHA 模型

```python
# memory_pool.py:2285 _kv_buffer_shapes
rows = self.size + self.page_size
return (
    (rows, self.head_num, self.head_dim),
    (rows, self.head_num, self.v_head_dim),
)
```

于是 `k_buffer: list[Tensor[rows, H, D]]`，**每层一个 tensor**（不是一个大 tensor）。
`rows = size + page_size`，多出来的一页吸收填充 token 的越界写入。

### MLA 模型（DeepSeek 系）

```python
# memory_pool.py:4446
self.kv_buffer = [
    torch.zeros((self.size + self.page_size, 1, self.kv_cache_dim), ...)
    for _ in range(self.layer_num)
]
```

MLA 只存**一个压缩后的 latent 向量**，不分开存 K 和 V。
`kv_cache_dim = kv_lora_rank + qk_rope_head_dim ≈ 512 + 64 = 576`。

对比一下每 token 字节数：

```
MHA 8 KV heads, head_dim 128, bf16:  2 × 8 × 128 × 2 = 4096 Byte
MLA (576 dim), bf16:                 1 × 576 × 2     = 1152 Byte
```

**MLA 是 MHA 的 28%**。这就是为什么 DeepSeek 这类模型能塞下超长上下文。

### `set_kv_buffer`：写入路径

```python
# memory_pool.py:148
row_bytes = row_dim * store_dtype.itemsize
if (_is_cuda or _is_hip) and can_use_store_cache(row_bytes, v_row_bytes):
    return store_cache(k.view(-1, row_dim), v.view(-1, v_row_dim), ...)
```

行字节数合适时走融合的 `store_cache` kernel；否则退回 `k_cache[indices] = k` 的
scatter 写法。CUDA Graph 捕获期间，K 和 V 的 scatter 会分流到两条 stream 上重叠执行。

---

## 6.5 分配器：为什么分配能做得这么快

`mem_cache/allocator/` 下有 `base.py` / `paged.py` / `token.py` / `page_interleave.py`。

### 核心思路：空闲链表放在 GPU 上

```python
class BaseTokenToKVPoolAllocator:
    self.free_pages = None         # 一个 int64 device tensor
    self.release_pages = None
    # None: 立刻释放；list: 攒着，等 free_group_end() 一起释放
    self.free_group: list[torch.Tensor] | None = None
```

`free_pages` **是一个 GPU tensor**，不是一个 Python list。

### 分配从头部切片，不循环

```python
# paged.py:160
def alloc(self, need_size: int):
    num_pages = need_size // self.page_size
    if num_pages > len(self.free_pages):
        self.merge_and_sort_free()
    if num_pages > len(self.free_pages):
        return None                                # 真的没内存了
    out_pages = self.free_pages[:num_pages]
    self.free_pages = self.free_pages[num_pages:]
    out_indices = (out_pages[:, None] * self.page_size
                   + torch.arange(self.page_size, device=self.device)).reshape(-1)
```

**没有 Python 循环**。切片 + 广播，全是 tensor 操作。

而 `alloc_extend` / `alloc_decode` 更是直接 **launch Triton kernel**：

```python
# paged.py:208
alloc_extend_kernel[(bs,)](
    prefix_lens, seq_lens, last_loc, self.free_pages,
    out_indices, next_power_of_2(bs), self.page_size,
)
self.free_pages = self.free_pages[num_new_pages:]
```

> **这是 SGLang 分配性能的关键**：分配是设备端 kernel，不是主机端循环。
> 主机只需要读 `len(free_pages)`（tensor 元数据，**不触发同步**）来判断能不能接新请求。

### 释放也做了优化

`free_segment`（paged.py:282）避免了一次 `torch.unique`：

```python
ps = self.page_size
assert start_pos % ps == 0, f"segment start {start_pos} is not page-aligned"
reps = free_index[::ps]        # 因为段是页对齐且连续的，每隔 page_size 取一个就是每页一次
self.free_page_ids(reps // ps)
```

如果段是页对齐且连续的，`free_index[::page_size]` 每个页恰好命中一次，
所以**不需要 `torch.unique`**（那会强制一次设备端排序）。

### `free_group`：批量释放

```python
def free_group_begin(self):  self.free_group = []
def free_group_end(self):    self._release_page_ids(*self.free_group); self.free_group = None
```

把 N 次小释放攒成一次 `torch.cat`。少了 N-1 次分配。

---

## 6.6 HiCache：KV 放不下怎么办

### 三层存储

```
L1  device pool   （GPU HBM）      最快、最小、最贵
L2  pool_host     （CPU DRAM）     慢 10×、大 10×、便宜
L3  storage       （SSD / 远端）   慢 100×、几乎无限
```

对应文件：`mem_cache/pool_host/` 和 `mem_cache/storage/`。

### 写策略（`--hicache-write-policy`）

| 策略 | 行为 | 代价 |
|---|---|---|
| `write_through`（默认） | **插入时就立刻备份到 host** | 每次插入都有传输开销，但驱逐时零开销 |
| `write_back` | **驱逐时才备份** | 平时零开销，驱逐时可能卡住 |
| `write_through_selective` | 只备份访问次数够多的 | 折中 |

源码里，写策略直接决定驱逐路径走哪个分支（`hiradix_cache.py:1206`）：

```python
def evict(self, params: EvictParams) -> EvictResult:
    if self.cache_controller.write_policy == "write_back":
        num_evicted = self._evict_write_back(num_tokens)
    else:
        num_evicted = self._evict_write_through(num_tokens)
```

### write_through 的一个重要不变式

```python
# hiradix_cache.py:857 write_backup
# Backup invariant (for write-through mode): backed-up nodes must form a
# contiguous prefix from root — no gaps. Skip if parent isn't backed up yet;
if not write_back and (node.parent != self.root_node and not node.parent.backuped):
    return 0
```

**host 上的备份必须是"从 root 开始的连续前缀"，不能有空洞。** 这看起来是个限制，
但它让 host 侧的 `match_prefix` 退化成一次简单的向下遍历，不需要处理碎片。

### 主机内存预算

```python
# pool_host/base.py:30
HICACHE_HOST_MEMORY_RESERVE_BYTES = 10 GiB
```

SGLang 会：
1. 从 cgroup 限制（或系统内存）里扣掉 10 GiB 预留
2. 把剩下的按同机 co-located 的 rank 数均分（`host_memory_budget_bytes`）
3. PP 组内 **all-reduce MIN**（`sync_fixed_hicache_size`），保证每个 stage 暴露
   相同的 token 容量

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--enable-hierarchical-cache` | `False` | 总开关 |
| `--hicache-ratio` | `2.0`（cache 模式） | host 池相对 device 池的倍数 |
| `--hicache-size` | `0` | 直接指定 GB，覆盖 ratio |
| `--hicache-write-policy` | `write_through` | 见上表 |
| `--hicache-storage-backend` | `None` | `file`/`nixl`/`mooncake`/`hf3fs`/`mori`/`shm`… |
| `--hicache-storage-prefetch-policy` | `timeout` | `best_effort`/`wait_complete`/`timeout` |
| `--hicache-host-memory-mode` | `cache` | `cache` 或 `buffer_only` |

> ⚠️ **注意 `write_back` 路径的源码注释**：
> "this path will be deprecated in the future." —— 新部署优先用 `write_through`。

---

## 6.7 什么时候该动这些参数

| 现象 | 动作 |
|---|---|
| `token usage` 长期 > 0.95 且频繁 retract | 提高 `--schedule-conservativeness`（1.3） |
| 并发上不去、显存有余量 | 提高 `--mem-fraction-static`（0.01 一档，直到 OOM 边缘） |
| 并发上不去、显存已满 | 考虑 KV 量化（`--kv-cache-dtype fp8_e4m3`，KV 字节直接减半） |
| 多请求共享长前缀但前缀太长 | 开 HiCache（`--enable-hierarchical-cache`） |
| 长 system prompt 反复出现 | HiCache + `--hicache-storage-backend` |
| prefill OOM | 降 `--chunked-prefill-size` 到 4096 / 2048 |
| decode OOM | 降 `--max-running-requests` |

**KV 量化是最便宜的并发提升手段**：fp8 让每 token 字节数减半 → `max_total_num_tokens`
直接翻倍 → 并发放大一倍，代价只是少量精度损失（`layers/quantization/kv_cache.py`）。

---

## 6.8 本章自检

1. `req_to_token` 这张表解决什么问题？为什么要多一行 padding row？
2. 为什么分配器的 `free_pages` 要放在 GPU 上？
3. `write_through` 的"连续前缀"不变式换来了什么？
4. `--mem-fraction-static` 里包含权重吗？
5. 你的 decode 并发只有 30，显存 80GB 用了 40GB，你该动哪个参数？

---

下一页 → [07 并行策略](07_parallelism.md)

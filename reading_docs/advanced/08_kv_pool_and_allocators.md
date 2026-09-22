# 进阶 08 · KV 池与分配器

> 对应入门 [06 显存与内存池](../beginner/06_memory_and_kv_pool.md)。
> **先读 `python/sglang/srt/mem_cache/README.md`（84 行）**。
> 源码：`mem_cache/memory_pool.py`（5904 行）、`mem_cache/allocator/`、
> `mem_cache/kv_cache_configurator.py`、`mem_cache/pool_configurator.py`

---

## 1. `ReqToTokenPool`：整个系统的间接表

```python
# memory_pool.py:272
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device, ...):
        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._alloc_size))
```

形状 `[max_running_requests + 1, max_context_len]`，**int32**。

### 两套分配系统并存

| | 存什么 | 数据结构 | 性能特性 |
|---|---|---|---|
| **row** | 请求的槽位（`req_to_token` 的行） | **Python list** `free_slots` | 数量级几千，Python list 足够 |
| **KV slot** | KV 的物理位置 | **device tensor** `free_pages` | 数量级百万，必须设备端 |

`alloc_rows`（:333）的一个细节：

```python
# :339
# Pop from the tail: O(need_size), unlike a prefix pop which is O(len(free_slots)).
select_index = self.free_slots[-need_size:]
del self.free_slots[-need_size:]
self.req_generation[select_index] += 1
```

**从尾部弹**：`del list[-n:]` 是 O(n)，而 `del list[:n]` 是 O(len(list))。
一个请求几千行、每步都要分配时，这个差别会累积。

### `req_generation`：ABA 防护

每次分配 bump 一次。用于检测"某个 row 在图捕获和 replay 之间被回收又复用"。

**为什么这是真问题？** CUDA Graph 捕获时把 `req_pool_indices` 写死在图里。
如果 replay 时那个 row 已经被别的请求复用，图就会读到错误的数据。
generation 让这种复用可被检测。

---

## 2. `MHATokenToKVPool`：物理布局

```python
# memory_pool.py:2285 _kv_buffer_shapes
def _kv_buffer_shapes(self):
    rows = self.size + self.page_size
    return (
        (rows, self.head_num, self.head_dim),
        (rows, self.head_num, self.v_head_dim),
    )
```

于是：

```python
self.k_buffer: list[torch.Tensor]   # 每层一个 [rows, H, D]
self.v_buffer: list[torch.Tensor]   # 每层一个 [rows, H, D_v]
```

**每层一个 tensor，不是一个大 tensor。** 三个理由：

1. 每层的 KV 独立索引（`out_cache_loc` 是层无关的槽位号，但每层的数据分开）
2. 便于按层做量化/布局变体
3. 避免单 tensor 超过 2^31 元素的索引限制（长上下文 + 大 batch 时是真问题）

`rows = size + page_size`：**多出来的一页吸收填充 token 的越界写入**
（和 `ReqToTokenPool` 的 padding row 是同一个思路）。

### 三种可选布局

```python
kv_cache_layout == "vectorized_5d"    # K: (num_blocks, H, D_k//x, page, x)
                                      # V: (num_blocks, H, page//x, D_v, x)
use_hnd  →  (num_pages, H, page, D)
（默认）  →  (rows, H, D)
```

`vectorized_5d` 是一种 swizzle，为了更好的访存合并（coalescing）。

### `MLATokenToKVPool`：一个 latent 向量

```python
# memory_pool.py:4446 _create_buffers
self.kv_buffer = [
    torch.zeros(
        (self.size + self.page_size, 1, self.kv_cache_dim),
        dtype=self.store_dtype, device=self.device,
    )
    for _ in range(self.layer_num)
]
```

`kv_cache_dim = kv_lora_rank + qk_rope_head_dim`（约 512 + 64 = 576）。

**关键**：`get_value_buffer` 返回的是**前 `kv_lora_rank` 列的视图**（:4500 附近）——
没有独立的 V 存储。

### 字节数对比

```
MHA 8 KV heads, head_dim 128, bf16:  2 × 8 × 128 × 2 = 4096 Byte/token/layer
MLA (576 dim), bf16:                 1 × 576 × 2     = 1152 Byte/token/layer
```

**MLA 是 MHA 的 28%。** 这是 DeepSeek 类模型能撑超长上下文的根本原因。

### `set_kv_buffer`：写入路径的融合

```python
# memory_pool.py:148
row_bytes = row_dim * store_dtype.itemsize
v_row_bytes = ...
if (_is_cuda or _is_hip) and can_use_store_cache(row_bytes, v_row_bytes):
    return store_cache(k.view(-1, row_dim), v.view(-1, v_row_dim), ...)
```

**行字节数"友好"时走融合的 `store_cache` kernel**，否则退回
`k_cache[indices] = k` 的 scatter。

CUDA Graph 捕获期间有个额外优化（:198-207）：

```python
# 把 K 和 V 的 scatter 分到两条 stream 上重叠执行
```

---

## 3. 分配器：`BaseTokenToKVPoolAllocator`

```python
# allocator/base.py:41
class BaseTokenToKVPoolAllocator(abc.ABC):
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        self.free_pages = None
        self.release_pages = None
        # None: free right away. A list: hold frees until free_group_end().
        self.free_group: list[torch.Tensor] | None = None
```

### `free_group`：批量释放

```python
def free_group_begin(self):
    self.free_group = []

def free_group_end(self):
    self._release_page_ids(*self.free_group)
    self.free_group = None
```

把 N 次小释放攒成**一次 `torch.cat`**。少了 N−1 次分配。

### `PagedTokenToKVPoolAllocator.alloc`

```python
# allocator/paged.py:160
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

**空闲页表在 GPU 上**，而且是**排序的**（从小到大），所以切片出来的页是连续的。
`out_indices` 就是"页号 × page_size + 页内偏移"的展开。

> `len(self.free_pages)` 是 tensor 元数据读取，**不触发设备同步**。
> 这是"主机侧能立刻判断能不能接新请求"的关键。

### `alloc_extend` / `alloc_decode`：Triton kernel

```python
# allocator/paged.py:183
def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                 last_loc, extend_num_tokens, ...):
    ...
    out_indices = torch.empty((bs, self.page_size), dtype=torch.int64, device=self.device)
    self.alloc_extend_kernel[(bs,)](
        prefix_lens, seq_lens, last_loc, self.free_pages,
        out_indices, next_power_of_2(bs), self.page_size,
    )
    ...
    self.free_pages = self.free_pages[num_new_pages:]
```

**这是关键的性能设计**：extend（prefill）的分配是**设备端 kernel**，
不是主机端循环。

三个输入值得注意：

- `last_loc`：每个请求当前最后一个槽位。**让 kernel 能复用请求已有页的剩余空间**，
  而不是浪费掉（如果最后一个页没写满，继续往里写）。
- `prefix_lens` / `seq_lens`：设备端张量版本，避免主机同步。
- `next_power_of_2(bs)`：block 维度用小幂次的 2，避免超大批次下的 illegal launch。

### `free_segment`：避开 `torch.unique`

```python
# allocator/paged.py:282
def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
    ps = self.page_size
    assert start_pos % ps == 0, f"segment start {start_pos} is not page-aligned"
    reps = free_index[::ps]
    ...
    self.free_page_ids(reps // ps)
```

**为什么可以免 `torch.unique`？**

因为 radix tree 节点的 `value` 是**页对齐且连续**的段
（`evict` 里的注释："Tree values are page-aligned copies of a kv row:
page-exact segment"）。

所以 `free_index[::page_size]` **每个页恰好命中一次**——
不需要 `torch.unique`（那会强制一次设备端排序，在百万级页表上很贵）。

**这是 radix tree 的 `page_aligned` 设计在分配器侧的回报。**

### `merge_and_sort_free`：只在慢路径

```python
# allocator/paged.py:153
def merge_and_sort_free(self):
    if self.num_staged_pages > 0:
        self.free_pages = torch.cat((self.free_pages, *self.staged_pages))
        ...
        self.free_pages, _ = torch.sort(self.free_pages)
        self.staged_pages = []
        self.num_staged_pages = 0
```

```python
# :326 _release_page_ids
if self.need_sort:
    self.staged_pages.extend(page_ids)
    self.num_staged_pages += sum(ids.numel() for ids in page_ids)
else:
    self.free_pages = torch.cat((*page_ids, self.free_pages))
```

**释放的页先暂存在 `staged_pages`（无序），只有 alloc 不够用时才排序。**
把排序成本从每次释放推迟到"真的需要"时。

### ROCm 的预热

```python
# allocator/paged.py:131-142
# 预 JIT rocPRIM 的 sort/unique，否则首次调用要 ~200 ms
```

**这是"第一次请求特别慢"的一个常见原因。**

### `TokenToKVPoolAllocator`：page_size = 1 的变体

```python
self.free_pages = torch.arange(1, size + 1)
```

`alloc` 从头部切片，`free` cat 到头部。用于 `page_size == 1`。

---

## 4. 容量计算：从 `--mem-fraction-static` 到 `max_total_num_tokens`

### 模型（`pool_configurator.py` 的 docstring）

```
available_bytes = max_tokens * coeff + bias
max_tokens = (available_bytes - bias) / coeff
```

### 第一步：可用字节（`kv_cache_configurator.py:2214`）

```python
def _profile_available_bytes(self):
    available_gpu_memory = get_available_gpu_memory(...)
    slack_gb = pre_model_load_memory * (1 - get_schedule().mem_fraction_static)
    ...
    rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb
    ...
    return int(rest_memory * (1 << 30))       # 返回到字节
```

**`--mem-fraction-static` 的准确含义**：模型加载**前**的可用显存中，
有多大比例是"静态分配"。

```
KV 预算 = 当前可用显存 − 非静态 slack − 多模态预留
         ↑ 权重已经常驻，所以这里已经是"扣掉权重之后"的数
```

### `gc.collect()` 的必要性

```python
# :2221-2226
# 测量前先 gc.collect()
```

注释解释了：**权重加载期间的临时张量如果还被引用，
会把"可用显存"低估好几个数量级**。这是"KV 池比预期小很多"的经典原因。

### 第二步：token 数（`pool_configurator.py:551`）

```python
def calculate_pool_sizes(self, available_bytes):
    available_bytes = max(available_bytes, 0)
    max_total_num_tokens = (
        available_bytes // self._cell_size if self._cell_size else self._zero_kv_max_tokens
    )
    max_total_num_tokens = max_total_num_tokens // page_size * page_size
    return max_total_num_tokens
```

`_cell_size` = **一个 token 在所有层上的 KV 字节数**。

### `_cell_size` 的构成

标准 MHA/MLA 按架构推导。**投机解码会放大它**：

```python
# :594-599
# EAGLE / STANDALONE:
_cell_size *= (1 + draft_num_layers / num_layers)
```

因为 draft 模型也需要自己的 KV 池。DFLASH 有自己的几何（`_dflash_draft_cell_size`，:125）。

混合 SWA 模型走 `HybridSWAPoolConfigurator`（:566+），
按 `--swa-full-tokens-ratio` 在 full 池和 SWA 池之间分配预算。

### 失败时的报错

```python
# MemoryPoolConfig.__post_init__（:85-91）
raise ValueError("Not enough memory. Please try to increase --mem-fraction-static.")
```

### `allocation_sizing.py`：每步的预留（不是池大小）

```python
def get_alloc_len_per_decode(...) -> int: ...

def get_alloc_reserve_per_decode(...) -> int:
    # The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap mode
    return 2 * get_alloc_len_per_decode()
```

**"2x 是双缓冲"**——overlap 模式下 `kv_committed_len` 有一轮的滞后，
所以要预留两份。

---

## 5. SWA / 混合池 / 统一池

### `SWAKVPool`（`swa_memory_pool.py:24`）

> Hybrid full/SWA cache composed from independently configurable pools.

注意力层被分成 `swa_attention_layer_ids` / `full_attention_layer_ids`。

**SWA（sliding window attention）层只需要 `sliding_window` 个 token 的历史**，
所以它的池子小得多。

```python
# arg_groups/fields/schedule.py:151
swa_full_tokens_ratio: A[Optional[float],
    "The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the
     number of swa:full layers. It should be between 0 and 1.
     E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens."]
```

默认 0.8。

**准入下限**（`pool_configurator.py:173 validate_swa_pool_size`）：
池子必须至少能装一个请求，即 `sliding_window_size + page_size`。

### `hybrid_cache/`

- `hybrid_pool_assembler.py`：`layer_id -> pool` 的路由
- `hybrid_cache_controller.py`：`HybridCacheController`（:169+），多池的 HiCache 控制器

`transfer_layer_id_max` 跨越所有 component pool
"including holes for uncached layers that the anchor pool alone cannot describe"。

### `UnifiedKVPool`（`unified_memory_pool.py:18`）

> one physical `uint8` byte buffer shared by N sub-pools. Two END
> `MultiEndedAllocator`s grow inward from opposite ends; optional 'float'
> MIDDLE pools live between their frontiers.

**核心思想**：
- 一个物理 uint8 buffer，N 个子池共享
- 两个"末端"分配器从**相反方向**向内生长（类似双向栈）
- 布局是 envelope-major，所以"一个槽位释放出来的区域正好是邻居能生长的区域"
- **上层存储的是虚拟槽位 ID，压缩只改虚拟↔物理映射表，不需要改写任何引用**

这是 KV 内存管理的收敛目标（issue #20415）。

### `mem_cache/layout/page_major.py`

`build_mha_views` / `build_mla_views`——让共享 buffer 能按 family 寻址的物理布局。

配套 flag：`--enable-page-major-kv-layout`（`fields/memory.py`）：

> lay out the Mamba state and full/SWA KV caches in a page-granularity envelope
> (page is the outermost axis, layer-major within a page) instead of the
> default per-layer (layer-major) layout. Requires the Triton attention /
> linear-attn / Mamba backends.

---

## 6. KV cache 量化

```bash
--kv-cache-dtype fp8_e4m3
```

choices 在 `mem_cache/kv_cache_dtype.py`：`auto`、`fp8_e5m2`、`fp8_e4m3`、`mxfp8`。

### `BaseKVCacheMethod` 不计算

```python
# layers/quantization/kv_cache.py:20
class BaseKVCacheMethod(QuantizeMethodBase):
    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer):
        # 建 k_scale / v_scale，初值是 -1.0（无效哨兵）
        # 两个 scale 都加载了就用它们的；都 ≤ 0 就用 1.0；只找到一个就复制
        ...
```

**它只是 scale 载体**，真正的量化/反量化发生在 attention 层的 `forward` 里。

**只支持 per-tensor scaling**（:73-74 会 `raise ValueError`）。

fp8-fnuz 会**把 scale 数量翻倍**（:50-52）。

### 家族变体

`fp4_kv_cache_quant_method.py`、`nvfp4_kv_cache.py`、`mxfp4_tensor.py`、`kvfp4_tensor.py`。

### 收益计算

```
bf16 → fp8：每 token 字节减半 → max_total_num_tokens 翻倍 → 并发放大一倍
bf16 → fp4：再减半
```

**这是最便宜的并发提升手段**，代价只是少量精度损失。

---

## 7. 调优决策树

```
KV 池不够
├── 显存还有余量？
│   └── 提高 --mem-fraction-static（0.01 一档，看 available_gpu_mem 到 5–8 GB）
├── 显存已满？
│   ├── 能接受精度损失
│   │   └── --kv-cache-dtype fp8_e4m3         ← 最便宜，KV 减半
│   ├── 有 CPU 内存
│   │   └── --enable-hierarchical-cache        ← 见第 09 章
│   ├── 模型支持 MLA？
│   │   └── 已经是 MLA 就没什么可压的了
│   └── 混合 SWA 模型？
│       └── 调 --swa-full-tokens-ratio
└── 都不行
    └── 加卡 / 降 --max-running-requests
```

### 一个诊断顺序

```bash
# 1. 看启动日志的 max_total_num_tokens 和 available_gpu_mem
# 2. 跑 benchmark，看 token_usage 和 kv_evictable_tokens
# 3. 如果 kv_evictable_tokens 很大但 token_usage 也高
#    → radix tree 占了很多，考虑调驱逐策略（第 03 章）
# 4. 如果 available 很小、evictable 也很小
#    → 真不够，走上面的决策树
```

---

## 8. 自检

1. `ReqToTokenPool` 为什么用 Python list 存 row，而 KV slot 用 device tensor？
2. `_kv_buffer_shapes` 里的 `+ page_size` 是干什么的？
3. `free_segment` 怎么避开 `torch.unique` 的？依赖什么前提？
4. `merge_and_sort_free` 为什么只在慢路径调用？
5. MLA 的 `get_value_buffer` 为什么能返回视图？
6. `gc.collect()` 在容量测量前为什么必要？
7. 投机解码为什么要放大 `_cell_size`？
8. `get_alloc_reserve_per_decode` 里的 2x 是为什么？

---

下一页 → [09 HiCache 与多级存储](09_hicache_and_multitier.md)

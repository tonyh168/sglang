# 进阶 06 · ModelRunner 与 CUDA Graph

> 对应入门 [05 ModelRunner](../beginner/05_model_runner.md)。
> 源码：`model_executor/model_runner.py`（2314 行，**frozen core file**）、
> `model_executor/runner/`、`model_executor/runner_backend/`、`model_executor/cuda_graph_config.py`

---

## 1. `ModelRunner.__init__`：顺序即契约

> ⚠️ `model_runner.py` 是 **frozen core file**。修改前必须先读
> `.claude/skills/large-class-style/SKILL.md`（见
> `.claude/rules/modify-component-must-read.md`）。

`__init__`（:315）只做编排，每步一个命名方法。关键顺序（带注释的理由）：

```python
# 1. 参数解析
self.mem_fraction_static, self.gpu_id, self.model_config, self.server_args = ...
self.is_draft_worker = ...
assert_published(server_args, role="scheduler")     # 运行上下文契约

# 2. 启动可观测性
self.init_startup_observability()                   # weight_load_time / graph_memory_usage
self.init_remote_instance_weight_transporter()
self.init_msprobe()
self.init_spec_aux_hidden_state()

# 3. 一些全局开关
misc_utils.maybe_disable_chunked_prefix_cache(...)
if self.enable_tf32_matmul:
    torch.set_float32_matmul_precision("high")

# 4. 设备设置 —— 注释：必须在 TransferEngine 初始化之前
torch.get_device_module(device).set_device(gpu_id)

# 5. 分布式初始化 —— 注释：Get available memory before model loading
self.init_torch_distributed()

# 6. 创建 forward stream —— overlap 调度用的那条
self.forward_stream = <device>.Stream()
self.shared_read_done_event = None                  # WAR barrier 读的邮箱

# 7. 杂项
create_offloader(...)
self.WeightChecker = ...
self.init_mindspore_runner()
deep_gemm_wrapper.update_deep_gemm_config(gpu_id)

# 8. 主力初始化
self.initialize()
self.check_quantized_moe_compatibility()
self._initialize_elastic_ep_joiner()
# 检查 model.forward 的签名里有没有 pp_proxy_tensors 来判断是否支持 PP
self.support_pp = ...
self.init_weight_updater()
self.init_weight_exporter()
```

### `initialize()`（:643）：顺序是 load-bearing 的

```python
def initialize(self):
    self.init_memory_saver_adapter()
    self.maybe_init_remote_instance_transfer_engine()   # 必须在权重加载之前
    self.maybe_init_expert_location_metadata()
    self.maybe_init_lplb_solvers()
    self.maybe_init_eplb_manager()
    self.maybe_init_elastic_ep()
    self.init_token_oracle()
    self.sampler = create_sampler()
    self.load_model()                                    # ← 加载权重
    prepare_moe_topk(model=..., moe_ep_size=..., moe_ep_rank=...)
    self.maybe_init_dwdp()
    # 注释：routed-experts capture must be disabled *before* backend/graph init
    #      so no draft graph records a capture-write kernel
    disable_routed_experts_capture_for_draft(self.model)
    self.maybe_init_expert_backup_client()
    self.layer_info = resolve_layer_indices(...)          # ModelLayerInfo: PP 的层范围
    adjust_hybrid_swa_layer_ids(...)
    self.maybe_apply_post_load_model_transforms()
    self.maybe_init_lora_manager()
    self.maybe_enable_batch_invariant_mode()
    configure_kv_cache_dtype()
```

**三个位置的依赖关系**：

1. `maybe_init_remote_instance_transfer_engine()` 在 `load_model()` 之前——
   因为权重加载可能要走 transfer engine。
2. `disable_routed_experts_capture_for_draft()` 在 backend/graph 初始化之前——
   否则捕获出来的 draft graph 里会录进一个写 capture buffer 的 kernel，之后 replay 会污染。
3. `configure_kv_cache_dtype()` 在最后——因为要等权重加载完才知道模型实际的 dtype。

### `load_model()`（:1178）的两个性能细节

```python
torch.set_num_threads(1)      # "reduce thread conflicts and speed up weight loading"
set_cuda_arch()
...
self.loader, self.model, self.startup_weight_load = load_model_with_memory_saver(...)
...
reserve_rope_cache_for_long_sequences(...)   # 见下
```

**`reserve_rope_cache_for_long_sequences`**：

> Pre-expand RoPE cache before CUDA Graph capture. Otherwise a long-context first
> request would allocate inside a graph.

**如果 RoPE 的 cos/sin 表是懒扩展的，第一个长上下文请求会在 CUDA Graph 的
捕获区里做内存分配**——CUDA Graph 不允许捕获期间分配显存，直接崩。
所以必须在捕获前把所有可能用到的长度都预留好。

这是一个典型的"性能优化引入的新约束"。

### `init_attention_backends()`（:1021）：顺序也有讲究

```python
configure_aux_hidden_state_capture(...)     # 注释：must precede CUDA graph capture
resolve_attention_backend_strs()            # → prefill_attention_backend_str
                                            #   / decode_attention_backend_str
build_attention_backends()                  # → self.attn_backend / self.decode_attn_backend
                                            #   / self.decode_attn_backend_group
```

---

## 2. CUDA Graph 的架构

### 两层分工

```
BaseCudaGraphRunner                 ← 编排：bucket 选择、静态 buffer 填充、
（runner/base_cuda_graph_runner.py）    attention metadata 初始化、replay 分派、输出切片
        │
        └── self.backend: BaseCudaGraphBackend    ← 机制：捕获/回放
```

### 四种 backend

| backend | 文件 | 做什么 |
|---|---|---|
| `full`（默认） | `runner_backend/full_cuda_graph_backend.py:77` | 整个 forward 捕获成**一个** `torch.cuda.CUDAGraph` per shape |
| `breakable` | `breakable_cuda_graph_backend.py:59` | 图可打断，某些 op 在 replay 中走 eager（`--debug-cuda-graph` 用） |
| `tc_piecewise` | `tc_piecewise_cuda_graph_backend.py:89` | torch.compile 驱动：在 attention 处 FX 切分，每段内部捕子图 |
| `disabled` | — | eager |

`BaseCudaGraphRunner` 的 docstring（:107）列出了四个 hook：

```python
def capture_prepare(self, size, *args, **kwargs) -> Any: ...
def capture(self) -> None: ...
def capture_one_shape(self, size, *args, **kwargs) -> Any: ...
@staticmethod
def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int: ...
```

**`tc_piecewise` 的特殊性**：torch.compile 自己管理 per-shape 的缓存，
所以这个 backend **没有 `_graphs` 表**，只有一个 `_compiled_fn` 被所有 shape 复用。

---

## 3. bucket 构造：`get_batch_sizes_to_capture`

```python
# runner/base_cuda_graph_runner.py:67
def get_batch_sizes_to_capture(model_runner, captured_req_width):
    capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size
    mul_base = get_cuda_graph_batch_size_alignment()
    alignment_width = captured_req_width
    if get_exec().overlap.enable_two_batch_overlap:
        alignment_width = 1
    # pad `num_max_requests` to avoid being filtered out
    num_max_requests = get_cuda_graph_max_batch_size(num_max_requests)
    if max(capture_bs) > num_max_requests:
        capture_bs += [num_max_requests]
    # Model input token count = bs * alignment_width; must be a multiple of attn_tp_size.
    capture_bs = [bs for bs in capture_bs if bs * alignment_width % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= get_exec().graph.torch_compile_max_bs]
        if get_flags().capture.enable_torch_compile else []
    )
    return capture_bs, compile_bs
```

### 三个过滤条件及理由

**① `bs * alignment_width % mul_base == 0`**

源码注释：

> Model input token count = bs * alignment_width; must be a multiple of attn_tp_size.

模型输入张量有 `bs × tokens_per_req` 行，attention-TP 切分要求这个数能被整除。
**不满足的 bucket 捕获出来的图永远 replay 不了**——必须提前过滤，
否则白捕获（浪费显存和时间）。

**② `bs <= num_max_requests`**

超过 `req_to_token_pool.size` 的 bucket 不可能被用到。

**③ `num_max_requests = get_cuda_graph_max_batch_size(num_max_requests)`**

先对 `num_max_requests` 做 padding，**避免它自己被条件 ① 过滤掉**——
最大 bucket 是最重要的一个。

### `alignment` 的组成（`utils/common.py:4053`）

```python
def get_cuda_graph_batch_size_alignment():
    alignment = 1
    if get_exec().overlap.enable_two_batch_overlap:
        alignment *= 2                                   # TBO 要拆成两个子批
    if require_gathered_buffer():
        alignment *= get_parallel().attn_tp_size         # DP attention 的 gather buffer
    if alignment % get_parallel().attn_cp_size != 0:
        alignment *= get_parallel().attn_cp_size         # CP padding
    return alignment
```

**这条链条说明了为什么改一个并行参数会连锁影响 CUDA Graph 的 bucket**。

---

## 4. padding：`_pad_to_bucket`

```python
# runner/base_cuda_graph_runner.py:157
@staticmethod
def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
    assert raw_size <= buckets[-1], (
        f"size {raw_size} exceeds max captured bucket {buckets[-1]}; "
        f"can_run_graph should have rejected this batch")
    index = bisect.bisect_left(buckets, raw_size)
    return buckets[index]
```

raw batch = 37，buckets = `[..., 32, 40, 48, ...]` → replay 40 的图，填充 3 行。

### 填充的安全性：`req_to_token` 的第 0 行

```python
# memory_pool.py:293
# +1 padding row at index 0: cuda-graph padded batches default
# req_pool_indices to 0, so dummy reads/writes land here harmlessly.
self._alloc_size = size + 1
self.req_to_token = torch.zeros((self._alloc_size, max_context_len),
                                dtype=torch.int32, device=device)
self.free_slots = list(range(1, self._alloc_size))
```

**填充行的 `req_pool_indices` 全默认成 0**，所以读写落在第 0 行这块垃圾区。
这就是 `size + 1` 存在的唯一理由。

### `req_generation`：ABA 防护

```python
# memory_pool.py:302
self.req_generation[select_index] += 1
```

`alloc_rows`（:333）里每分配一行就 bump 一次 generation。
**目的是检测"某个 row 在图捕获和 replay 之间被回收又复用"**。

```python
# :339
# Pop from the tail: O(need_size), unlike a prefix pop which is O(len(free_slots)).
select_index = self.free_slots[-need_size:]
del self.free_slots[-need_size:]
```

注意 `free_slots` 是**普通 Python list**（不是 device tensor）——
row 的数量级是几千，Python list 足够快，而且不需要设备同步。

---

## 5. `can_run_graph`：什么时候不能走图

`decode_cuda_graph_runner.py:646`。拒绝条件：

| 条件 | 原因 |
|---|---|
| `raw_size > max_bs` | 超出最大捕获尺寸 |
| `replace_embeds is not None` | 动态的 per-request embedding，形状不固定 |
| per-request 宽度 != `captured_req_width` | 投机解码的 batch 形状不匹配 |
| encoder-decoder 批里有 `encoder_len == 0` | 形状不合法 |
| TBO 不支持的批 | — |
| ngram 批的 `batch_size * captured_req_width != input_ids.numel()` | 形状不匹配 |

**`can_run_graph` 是 `_pad_to_bucket` 那个 assert 的前置保证**——
先拒绝，再断言。

---

## 6. attention metadata 的三段式拆分

这是 CUDA Graph 给 attention 后端带来的最主要接口变化。

```python
# layers/attention/base_attn_backend.py:36
def init_forward_metadata(self, fb):
    """eager 入口，默认 = _out_graph + _in_graph"""
    self.init_forward_metadata_out_graph(fb)
    self.init_forward_metadata_in_graph(fb)

def init_forward_metadata_out_graph(self, fb, in_capture=False):
    """在 `with graph.capture():` 之外跑
    主机侧 op / 动态形状 / 不可录制的逻辑"""

def init_forward_metadata_in_graph(self, fb):
    """在捕获区之内跑的、可录制的静态形状 GPU op
    会被 graph.replay() 自动重放"""
```

**旧的 `init_forward_metadata_capture_cuda_graph` / `..._replay_cuda_graph`
已经从 ABC 里移除了。**

**这个拆分是纯 CUDA Graph 驱动的**：

```
out_graph 里的逻辑：每次 replay 前都要重新跑（主机侧算 indptr、算 page table 等）
in_graph 里的逻辑：只需在捕获时录一次，replay 时自动执行
```

如果全都放在 out_graph，每次 replay 都要做一遍主机侧工作，overlap 就打折了。
如果全都放在 in_graph，那些不能录制的 op（主机侧同步、动态分配）会让捕获失败。

---

## 7. 相关 CLI 速查

| Flag | 默认 | 说明 |
|---|---|---|
| `--cuda-graph-backend-decode` | `full` | `full` / `breakable` / `tc_piecewise` / `disabled` |
| `--cuda-graph-backend-prefill` | 见 `cuda_graph_config.py` | 同上 |
| `--cuda-graph-max-bs-decode` | 自动 | decode 最大捕获 batch |
| `--cuda-graph-max-bs-prefill` | 自动 | prefill 最大捕获 batch |
| `--cuda-graph-max-seq-len-prefill` | — | 超过则走 eager prefill |
| `--cuda-graph-bs-decode` / `-prefill` | — | 显式指定 bucket 列表 |
| `--cuda-graph-prefill-max-context`（别名 `--context-bucket`） | — | prefill 图按 context 分桶 |
| `--cuda-graph-tc-compiler` | `eager` | `eager` / `inductor` |
| `--disable-prefill-cuda-graph` | False | |
| `--disable-decode-cuda-graph` | False | |
| `--disable-cuda-graph-padding` | False | 关掉填充（要求精确匹配） |
| `--enable-profile-cuda-graph` | False | |
| `--enable-cudagraph-gc` | **False** | 注释：GC 在捕获期间被冻结以加速 |
| `--debug-cuda-graph` | False | 用 breakable backend 调试 |

### prefill CUDA Graph

`cuda_graph_config.py` 里的注释：

> full for prefill captures one whole-forward graph per num_tokens bucket with a
> fixed request-slot count; replay pads num_tokens up to the nearest captured
> bucket. **Opt-in: the padding waste is the operator's call.**

**prefill 的图 bucketing 维度和 decode 不同**：decode 按 batch size 分桶
（因为每个请求固定 1 个 token），prefill 按 **num_tokens** 分桶。

理由：prefill 的 batch 形状是 `[总 token 数, hidden]`，总 token 数才是决定
kernel 形状的量。而且 prefill 的 batch size 变化范围极大（1 到几千），
按 batch size 分桶会浪费很多。

**代价是 padding 浪费**：一个 8000 token 的 prefill 落到 8192 的 bucket，
多算 192 个 token 的活。**所以这是 opt-in 的**——"padding waste is the
operator's call"。

---

## 8. 一个具体的数字算例

**模型**：60 层，hidden=7168，TP=8。

**每层的 kernel 数**（粗算）：

```
input_layernorm (fused_add_rmsnorm)     1
qkv_proj (FP8 GEMM)                     1
attention (flashinfer decode)           1
o_proj (FP8 GEMM)                       1
post_attention_layernorm                1
gate_up_proj                            1
topk routing                            1
fused_moe (或 dispatch + gemm + combine)  1–3
down_proj                               1
TP all-reduce (×2，可能融合)             1–2
────────────────────────────────────────────
≈ 10–13 个 kernel/层
```

```
60 层 × 11 ≈ 660 个 kernel
```

**eager 模式**：

```
CPU 侧：660 × 3 μs ≈ 2.0 ms 纯启动开销
GPU 侧：可能只有 1–3 ms 实际计算（batch 小时）
→ CPU 占了 40–70% 的墙钟时间
```

**CUDA Graph 模式**：

```
CPU 侧：1 次 cudaGraphLaunch ≈ 10 μs
GPU 侧：不变
→ 启动开销从 2ms 降到 0.01ms
```

**这就是 CUDA Graph 对 decode 是"必需品"而不是"优化项"的原因**——
不做它，TPOT 会被 CPU 锁死在 2ms + GPU 时间。

---

## 9. 自检

1. 为什么 `reserve_rope_cache_for_long_sequences` 必须在图捕获之前？
2. `disable_routed_experts_capture_for_draft` 为什么必须在 backend/graph 初始化之前？
3. bucket 的整除过滤条件去掉会怎样？
4. `req_to_token` 为什么需要 `size + 1` 行？
5. `init_forward_metadata` 拆成三段分别解决什么？
6. prefill 的 CUDA Graph 为什么按 num_tokens 而不是 batch size 分桶？
7. `--enable-cudagraph-gc` 默认 False 是为什么？

---

下一页 → [07 Attention 后端与 Kernel](07_attention_backends_and_kernels.md)

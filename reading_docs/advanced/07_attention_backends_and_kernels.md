# 进阶 07 · Attention 后端与性能关键 Kernel

> 对应入门 [05 ModelRunner](../beginner/05_model_runner.md) 的后端与 kernel 部分。
> 源码：`layers/attention/`（40+ 文件）、`layers/sampler.py`、`layers/layernorm.py`、
> `layers/activation.py`、`layers/linear.py`、`layers/radix_attention.py`

---

## 1. Attention 后端注册表

```python
# layers/attention/attention_registry.py
ATTENTION_BACKENDS = {}

def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn
    return decorator
```

每个条目是一个工厂 `f(runner) -> AttentionBackend`。

### 选择逻辑

`model_runner_components/attention_backend_setup.py`：

```python
resolve_attention_backend_strs(...)    # :178 从配置读 (prefill, decode) 两个名字
_build_resolved_backend(...)           # :200 构建
```

当 `decode != prefill` 时组合成 `HybridAttnBackend`。源码注释解释了一个坑：

> Compose the two full-attention backends first, then apply model-level wrappers
> once. Wrapping each child independently duplicates the linear/sparse side
> backend for hybrid models (for example, two GDN dispatchers for Qwen3.5 ...),
> duplicating initialization and associated state while only one side backend
> can be active in a forward pass.

**先组合，再套 wrapper**——否则混合模型（attention + linear attention）
会初始化两份线性注意力后端。

### 完整 choices 列表

```python
# arg_groups/choices.py:74
ATTENTION_BACKEND_CHOICES = [
    # Common
    "triton", "torch_native", "flex_attention",
    "dsa", "nsa",        # nsa 是 dsa 的废弃别名
    "qsa", "dsv4",       # compressed 是 dsv4 的别名
    # NVIDIA specific
    "fa3", "fa4", "flashinfer", "flashmla", "trtllm_mla", "cutedsl_mla",
    "tokenspeed_mla", "trtllm_mha", "hpc_ops",
    "minicpm_flashattn", "minicpm_flashinfer",
    # AMD specific
    "aiter", "wave",
    # Other platforms
    "intel_amx", "ascend", "intel_xpu",
]
```

另外两组独立列表：

```python
DRAFT_ATTENTION_BACKEND_CHOICES = [
    "flashinfer", "fa3", "fa4", "triton", "ascend", "trtllm_mha", "intel_xpu",
]
DETERMINISTIC_ATTENTION_BACKEND_CHOICES = [
    "ascend", "fa3", "fa4", "flashinfer", "intel_xpu", "triton",
]
```

**`DETERMINISTIC_ATTENTION_BACKEND_CHOICES` 是 `--enable-deterministic-inference`
的白名单**——只有这些后端能保证确定性。

### 相关 flag

| Flag | 说明 |
|---|---|
| `--attention-backend` | 统一指定 |
| `--prefill-attention-backend` | 单独指定 prefill（**优先级更高**） |
| `--decode-attention-backend` | 单独指定 decode |

三个都带 `resolvable=True`，即支持运行上下文覆盖。

---

## 2. 后端逐个看

### `flashinfer` —— NVIDIA 默认（MHA）

- prefill: `BatchPrefillWithPagedKVCacheWrapper`（`forward_extend`，:1307）
- decode: `BatchDecodeWithPagedKVCacheWrapper`（:1468）
- 每个 layer group 一个 wrapper（`_get_wrapper_idx(layer)`）

**关键优化：wrapper 是 `plan()` 一次、多次使用的。**

```python
# 每批 plan 一次，所有层复用
wrapper.plan(...)
# 每层只是调用 kernel，不再重复 planning
```

planning 的成本被摊到几十层上。

还有 `prefill_uses_dequant_workspace` / `decode_uses_dequant_workspace`——
FP4/NVFP4 的 KV 要先 dequant 到 workspace 供 chunk-prefill / cache-reuse 使用。

MLA 模型会自动切到 `FlashInferMLAAttnBackend`。

### `fa3` / `fa4` —— prefill 主力

```python
# flashattention_backend.py
# Guard: (major == 8 and not runner.use_mla_backend) or major == 9
```

即 **SM80（仅 MHA）或 SM90**。`fa4` 是同一个类，`fa_impl_ver=4`。

- prefill: `flash_attn_varlen_func`
- decode: `flash_attn_with_kvcache`
- **预计算 FA3 的 scheduler metadata**（:101），避免每层都做 `prepare_varlen_num_blocks`

一个值得注意的分支（:368）：

> skip KV cache write and use flash_attn_varlen_func with raw K/V instead of
> flash_attn_with_kvcache, **bypassing paged KV cache entirely**

这是 sliding-window / prefix 复用的路径——可以直接用明文 K/V，绕过分页。

### `triton` —— 跨平台 fallback

```python
from sglang.kernels.ops.* import (
    extend_attention_fwd, extend_attention_fwd_unified,
    decode_attention_fwd, dense_prefill_attention_fwd,
)
```

这些 kernel 被 `torch.compiler.disable(...)` 包着（:178-192），
防止 torch.compile 尝试 trace 它们。

还有 `get_num_kv_splits_triton` 做 split-KV decode 的归约。

**`--attention-backend triton` 是所有厂商都能跑的路径**，也是硬件适配的起点。

### `torch_native` —— 正确性基线

纯 PyTorch，用 `torch.nn.functional.scaled_dot_product_attention`。
**明确是 fallback，不是性能路径。**

### `flashmla` —— MLA decode

```python
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
PAGE_SIZE = 64
```

MLA 专用（`PAGE_SIZE` 硬编码为 64）。它继承 `FlashInferMLAAttnBackend` 来复用 prefill。

### `trtllm_mla` —— TRT-LLM GenMLA

包含 MLA 的 data-prep / workspace / prefill 管线，`cutedsl_mla_backend.py` 复用这些：

> Subclasses TRTLLMMLABackend (backend="cute-dsl") to reuse its MLA data
> preparation, workspace, and prefill plumbing.

**一个重要的限制**（来自注册表的错误信息）：

> `trtllm_mla` cannot serve decode context parallelism with spec decoding — it
> does not forward cyclic DCP metadata and returns no rank-local LSE for the
> cross-rank merge.

**不支持 DCP + 投机解码的组合**，且是启动时报错。

### `aiter` —— ROCm/AMD

AITER kernel 全家桶，"end to end attention solution with aiter kernels"。
AMD 上还有 `wave`。

---

## 3. `RadixAttention`：薄封装的全部职责

```python
# layers/radix_attention.py:98
class RadixAttention(nn.Module):
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, layer_id,
                 logit_cap=0.0, tp_q_head_num=None, tp_k_head_num=None, ...):
        # 只有几何信息，没有任何 KV 存储
        self.tp_q_head_num = tp_q_head_num
        self.tp_k_head_num = tp_k_head_num
        self.head_dim = head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size
        self.use_irope = use_irope
        self.k_scale = None            # KV 量化的 scale（由 kv_cache method 填）
        self.v_scale = None
```

`forward`（:157）做三件事：

### ① 规范化 shape

```python
k = k.reshape(-1, self.tp_k_head_num, self.qk_head_dim)   # 或 v_head_dim（MLA absorbed）
v = v.reshape(-1, self.tp_k_head_num, self.v_head_dim)
```

`k_rope` 在 kwargs 里时说明是 MLA 的 absorbed 形式，用 `v_head_dim`。

### ② tc-piecewise 路径（:176）

```python
if forward_mode.is_extend() and tc_piecewise_forward_context is not None \
        and not _force_eager_attn:
    output = q.new_empty(...)                    # 静态形状输出
    unified_attention_with_output(query, key, value, output, ...)
```

`_unified_attention_with_output_impl`（:331）做的事：

```python
query = query[:real_query_num_tokens]           # 收窄到真实 token 数
# 临时改写 forward_batch 的 out_cache_loc / positions 为收窄后的视图
forward_batch._attn_output = output[:real_query_num_tokens]
get_attn_backend().forward(...)
```

**然后对 idle 的 DP rank 清零输出，对 PCG replay 的填充尾部清零**
（`_zero_padded_pcg_tail`）：

> NaN/Inf cannot reach residual / MoE routing / allreduce

**这是关键**：填充出来的假 token 如果算出 NaN，会通过残差连接和 MoE 路由
污染真实数据（MoE 的 routing 会 all-reduce softmax 权重）。

### ③ 普通路径（:290）

```python
return get_attn_backend().forward(q, k, v, self, forward_batch, save_kv_cache, **kwargs)
```

**整个 wrapper 的职责就是**：规范化 shape、给后端一个可录制的 op 边界、
隔离填充/idle 的清零。**它不做任何数学。**

---

## 4. Sampler：融合采样

```python
# layers/sampler.py:108
class Sampler(nn.Module):
    def forward(self, logits_output, sampling_info):
        logits = logits_output.next_token_logits
        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                original_logprobs = logprobs = F.log_softmax(logits, dim=-1)
        else:
            simple_sampling_case = (not sampling_info.need_top_p_sampling
                                    and not sampling_info.need_top_k_sampling
                                    and not sampling_info.need_min_p_sampling)
            ...
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)   # In-place op to save memory
            probs = logits
            batch_next_token_ids, sampling_mask_capture = self._sample_from_probs(
                probs, sampling_info, positions, simple_sampling_case)
```

### `_sample_from_probs`（:317）

```python
if simple_sampling_case:
    return sampling_from_probs_torch(probs, sampling_seed, positions)

# 按 sampling_backend 分派
if sampling_backend == "flashinfer":
    return top_k_top_p_sampling_from_probs(probs.contiguous(), top_ks, top_ps,
                                           filter_apply_order="joint")
elif sampling_backend == "pytorch":
    return top_k_top_p_min_p_sampling_from_probs_torch(...)
```

### 为什么必须融合

batch 有 B 个序列，每个都要：
1. 在 V=128k 的词表上做 top-k
2. top-p（排序/扫描）
3. 多项式采样

用 torch 原语：

```
top_k:  1 次 kernel + 1 个 [B, V] 中间张量
top_p:  1 次 kernel（含 sort）+ 1 个 [B, V] 中间张量
采样:    1 次 kernel
≈ 5–8 次 kernel 启动 + 若干次 [B, 128000] 的 HBM 往返
```

每次 `[B=64, 128000] × 4 Byte = 32 MB`，来回几次就是几百 MB 的 HBM 流量。

**融合 kernel `top_k_top_p_sampling_from_probs` 一趟搞定**——
logits 已经在寄存器/shared memory 里，不需要 materialize 中间结果。

### 一个必须保持的不变量

源码在捕获点（`sampling_mask_capture`）的注释：

> the fused joint sampler and these separate renormalization primitives must
> share cutoff, tie, and joint-support semantics so captured positive support
> exactly describes the sampler's action space.

**融合采样器和分开的 renormalize 原语必须在截断点、平局处理、联合支撑集上
语义一致**——否则"捕获到的正支撑集"不能正确描述采样器的动作空间。

这是 CUDA Graph + 采样结合时的隐蔽陷阱。

### 全 greedy 的捷径

`SamplingBatchInfo`（`sampling/sampling_batch_info.py:50`）在**每批构造时**
计算一次：

```python
self.is_all_greedy = ...
self.need_top_p_sampling = ...
self.need_top_k_sampling = ...
self.need_min_p_sampling = ...
```

`merge_batch` 时用 `&=` / `|=` 合并（:494-498）。

**如果整个 batch 的请求都是 greedy，采样器走 `argmax` 分支，
跳过 softmax 和所有过滤 kernel。**

### 被刻意跳过的 TP 同步

```python
def _sync_token_ids_across_tp(self, batch_next_token_ids, sampling_info):
    if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
        # For performance reasons, SGLang does not sync the final token IDs
        # across TP ranks by default. This saves one all-reduce, but the
        # correctness ... depends on the determinism of the last all-reduce,
        # the last lm_head matmul, and all sampling kernels.
        torch.distributed.all_reduce(batch_next_token_ids, op=dist.ReduceOp.MIN,
                                     group=self.tp_sync_group)
```

**默认不同步**，省一次 all-reduce。代价是要求那几步 kernel 是确定性的。
grammar 场景必须同步（FSM 状态要一致）。

---

## 5. RMSNorm + 残差：`fused_add_rmsnorm`

```python
# layers/layernorm.py:439 class RMSNorm(BaseFusedOp)
else:
    # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.
    # but right now we can only have hidden_states+(residual+post_residual_addition).
    # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
    # we probably need another parameter to fused_add_rmsnorm
    if post_residual_addition is not None:
        residual = residual + post_residual_addition
    fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
    result = x, residual
```

### 朴素实现 vs 融合

```
朴素：
  h = x + residual          读 x, 读 residual, 写 h        (1 kernel)
  var = h.pow(2).mean(-1)   读 h（2 次：pow 和 mean）       (2 kernel)
  h = h * rsqrt(var+eps)    读 h, 写 h                     (1 kernel)
  out = h * weight          读 h, 读 w, 写 out             (1 kernel)
  ─────────────────────────────────────────────────────
  5 个 kernel，[T, H] 张量读了 ~6 遍

融合 fused_add_rmsnorm：
  读 x, 读 residual, 读 w, 写 x, 写 residual
  1 个 kernel，每个张量读一遍写一遍
```

**60 层 × 2 个 norm × 8192 token × 7168 hidden × 2 Byte**：
朴素实现每次前向要多搬 **~20 GB** 的 HBM 流量。

### 原地修改的语义

`fused_add_rmsnorm(x, residual, ...)` **原地修改 `x` 和 `residual`**，
返回 `(x, residual)`。所以下一层复用同一个 buffer——**零额外分配**。

这也是为什么 `fp32_residual` 是个 flag：residual 累加器的精度要显式控制。

### 三个相关变体

```python
forward_with_per_tensor_quant_fusion(...)      # 把下游的 FP8 per-tensor 量化融进 norm
_flashinfer_fused_add_rmsnorm_quant(...)
flashinfer_allreduce_residual_rmsnorm(...)     # all-reduce + residual add + RMSNorm 一个 kernel
```

最后这个见第 10 章。

---

## 6. `silu_and_mul`：SwiGLU 的融合

```python
# layers/activation.py:130 class SiluAndMul
def forward_cuda(self, x):
    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,),
                      dtype=x.dtype, device=x.device)
    silu_and_mul(x, out)
    return out
```

```
朴素：x 宽度 2N（gate ‖ up）
  silu(x[..., :N])         1 kernel
  torch.chunk / split      0 kernel（视图）但 mul 需要 materialize
  silu_result * x[..., N:] 1 kernel
  ─────────────────────
  2–3 个 kernel + 一个 2N 宽的中间张量

融合 silu_and_mul：
  读一次 2N 宽的 x，算 silu(gate) * up，写 N
  1 kernel，无中间张量
```

同样有 `gelu_and_mul` / `gelu_tanh_and_mul` / `SituAndMul`。

**都是 `BaseFusedOp` 子类**，这样 tc-piecewise backend 能在编译/不编译模式间切换
（`tc_piecewise_cuda_graph_backend._toggle_fused_ops`）。

---

## 7. `linear.py`：量化如何接入而不污染模型代码

```python
# layers/linear.py:145
class LinearBase(torch.nn.Module):
    def __init__(self, input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix):
        self.quant_method = quant_config.get_quant_method(self, prefix)
        ...

    def forward(self, x, bias=None):
        return self.quant_method.apply(self, x, bias)
```

**`forward` 只有一行：委托给 `quant_method`。**

四个子类：`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` /
`MergedColumnParallelLinear`——都是 TP 切分的变体。

`quant_config` 可以是 `UnquantizedLinearMethod`，所以**无量化路径也是同一个接口**。

> ⚠️ 没有 `--linear-attn-backend` 这个 flag。相关的选择器是
> `--quantization`、`--dtype`、`--kv-cache-dtype`，以及 GEMM runner
> （`--fp8-gemm-backend` / `--fp4-gemm-backend`）。

---

## 8. 性能分析矩阵

| kernel | 融合前 | 融合后 | 省掉的 HBM 流量（60 层，8192 token，H=7168，bf16） |
|---|---|---|---|
| `fused_add_rmsnorm` | 5 kernel | 1 kernel | ~20 GB/forward |
| `silu_and_mul` | 3 kernel | 1 kernel | ~4 GB/forward |
| `top_k_top_p_sampling` | 5–8 kernel | 1 kernel | 每步 ~200 MB |
| `allreduce + rmsnorm` | 2 kernel + 1 次往返 | 1 kernel | ~7 GB/forward（TP=8） |
| `store_cache` | scatter | 融合 | 每步 KV 写入 |

> 这些都是量级估算，实际取决于 hidden size、token 数、TP size。
> **关键是理解每一项省掉的是"一次完整的 HBM 往返"，而不是"几个 kernel 启动"。**

---

## 9. 自检

1. FlashInfer 的 wrapper 为什么要 `plan()` 一次让所有层复用？
2. `_zero_padded_pcg_tail` 防的是什么？不清零会出什么错？
3. 为什么采样必须融合成一个 kernel？
4. `is_all_greedy` 的快捷路径省掉了哪些 kernel？
5. 为什么默认不做 TP 间的 token id 同步？
6. `fused_add_rmsnorm` 为什么能原地修改？这带来什么好处？
7. `LinearBase.forward` 只有一行，好处是什么？

---

下一页 → [08 KV 池与分配器](08_kv_pool_and_allocators.md)

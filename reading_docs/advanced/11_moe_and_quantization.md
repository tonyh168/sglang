# 进阶 11 · MoE 与量化

> 对应入门 [08 投机解码与高级特性](../beginner/08_speculative_and_advanced.md) 的
> MoE / 量化部分。
> 源码：`layers/moe/`、`layers/quantization/`、`compilation/`

---

## 1. MoE 的分组 GEMM：两种范式

一个 MoE 层把每个 token 路由到 top-k 个专家（DeepSeek-V3 是 256 选 8）。
**token→专家是数据相关的**，所以 GEMM 的形状是运行时才知道的。

### 范式 (a)：按专家排序 + 每专家一个 GEMM

```
1. moe_align_block_size 算出每个专家的 token 数
2. 按专家 id 排序 token
3. 对每个专家跑一次 GEMM（或用一个 grouped GEMM kernel 循环）

优点：每个 GEMM 内部规整，算术强度好
缺点：小 batch 时每专家的 token 数很少且参差不齐 → 大量小 GEMM
```

### 范式 (b)：块稀疏 GEMM

```
1. 把所有 token 聚成一个连续 buffer
2. 每个 block 带一个 expert id
3. 一个 kernel 迭代所有 block，逐块查专家

优点：小 batch 时更优（一个 kernel 覆盖所有 token）
缺点：大 batch 时不如规整 grouped GEMM
```

### SGLang 的具体实现

Triton 的 `fused_moe` kernel 走范式 (b)：

```python
moe_align_block_size(topk_ids, block_size, num_experts,
                     sorted_token_ids, expert_ids, num_tokens_post_padded)
# 然后一个 kernel 按 block 迭代，逐块查专家
```

DeepGEMM / CUTLASS / TRT-LLM 的变体走范式 (a) 或混合。

**`--moe-runner-backend` 的 choices**（`arg_groups/choices.py:139`）：

```python
["auto", "deep_gemm", "triton", "triton_kernel", "flashinfer_trtllm",
 "experimental_sgl_trtllm", "flashinfer_trtllm_routed", "flashinfer_cutlass",
 "flashinfer_mxfp4", "flashinfer_cutedsl", "flashinfer_megamoe", "cutlass",
 "aiter", "marlin", "humming", "experimental_sgl_marlin", "hpc_ops",
 "megamoe", "intel_xpu"]
```

---

## 2. `MoeRunner`：core 与 fused 两条路

```python
# moe/moe_runner/runner.py:53
class MoeRunner:
    def __init__(self, config: MoeRunnerConfig, ...):
        ...
```

`MoeRunnerConfig` 在 `moe_runner/base.py:41`。

### 两种执行形状

**`runner_core`（非融合路径）**：

```python
TritonRunnerCore        # moe_runner/triton.py
DeepGemmRunnerCore      # moe_runner/deep_gemm.py
TritonKernelsRunnerCore
AiterRunnerCore
HummingRunnerCore
AscendRunnerCore
```

```python
def run(self, runner_input, quant_info):
    ...
    return self.runner_core.run(runner_input, quant_info)
```

**`fused_func`（融合路径）**：从全局 `FusedOpPool` 按
`(a2a_backend, runner_backend)` 查出。

**Marlin / FlashInfer-TRTLLM / CuteDSL / CUTLASS / HPC-Ops / MegaMOE 只有 fused 路径**
（`fused_func` 是 `None` 时表示它们不支持 non-fused）。

```python
# run() 优先用 fused_func，除非开了 LoRA（LoRA 需要 non-fused 路径）
```

### 启动时的密集校验

```python
# 值得引用为设计意图
# deepep_v2 必须配 deep_gemm runner：
#   "deepep_v2 dispatches into the deep_gemm grouped-GEMM layout
#    (FP8 activations plus scales, or BF16 activations for unquantized experts)"
# hpc_ops 必须被每个 MoE 层选中，否则：
#   "would consume global ids as local ones and misroute tokens under EP>1,
#    so fail loudly at startup"
```

**最后这条是一个真实的正确性陷阱**：MoE kernel 用**局部专家 id**，
如果某些层用全局 id 就会路由错。SGLang 的选择是**启动时大声报错**。

---

## 3. `deep_gemm.py`：把量化融进 GEMM

```python
# moe_runner/deep_gemm.py
from sglang.kernels.ops.moe import (
    per_token_group_quant,        # per-token-group 量化
    silu_and_mul_clamp,           # 激活 + clamp
    silu_and_mul_masked_post_quant,   # 激活 + masked 后量化
    pad_expert_counts,
)
```

**量化、激活、padding 全部融进 GEMM 流水线**，而不是独立的 kernel。

配套的有 `deep_gemm_sm120.py`（SM120 变体）和
`use_symmetric_memory`（来自 `pynccl_allocator`，用于对称内存的 all-to-all）。

---

## 4. `token_dispatcher/`：EP 的 all-to-all

```python
# token_dispatcher/base.py
class DispatchOutput / CombineInput   # 各后端实现自己的协议
```

实现列表：

```
standard, deepep, deepep_v2, pplx, mooncake, nixl, mori, flashinfer, ascend_tp
```

```python
# moe/utils.py:34
class MoeA2ABackend(Enum):
    none, deepep, mooncake, nixl, mori, ascend_fuseep, ascend_tp,
    flashinfer, megamoe, deepep_v2, pplx, flashinfer_megamoe, customized
```

### `DispatcherBaseHooks`（`base.py:50`）

允许在 dispatch 前后插 hook。**TBO / SBO 就是靠这个实现的**：

```python
# batch_overlap/single_batch_overlap.py
```

**把通信和计算重叠，必然要在 dispatcher 前后插桩**——这是唯一能拿到
"通信开始/结束"这两个时间点的地方。

---

## 5. `TopK`：路由输出的四种格式

```python
# moe/topk.py:535
class TopK(BaseFusedOp):
    ...
```

`TopKConfig`（:247）携带：

```python
top_k, use_grouped_topk, topk_group, num_expert_group,
renormalize, num_fused_shared_experts,
correction_bias,              # DeepSeek 的 e_score_correction_bias
routed_scaling_factor,
scoring_func,                 # softmax / sigmoid
output_format,
```

### 四种输出格式（:322）

| 格式 | 类 | 说明 |
|---|---|---|
| `STANDARD` | `StandardTopKOutput(weights, ids, router_logits)` | 常规 |
| `BYPASSED` | `BypassedTopKOutput` | **不实体化路由结果**，让融合后端直接消费 |
| `TRITON_KERNEL` | `TritonKernelTopKOutput` | 给 triton_kernels runner |
| `PACKED` | `PackedTopKOutput` | — |

**`BYPASSED` 省掉的是**：`[tokens, num_experts]` 这个中间张量。
256 个专家 × 8192 个 token × 4 Byte = 8 MB，而且在每层都要生成一次。

### 分派（`forward_cuda`，:654）

按 runner backend 选。`forward_native` 设 `topk_config.torch_native = True`
并调 `select_experts`。

:190 有融合的 `fused_topk_deepseek`。

### 一个性能注释

```python
# _torch_compile_forward（:627）
# "torch.compile of the native TopK only pays off at bs=1;
#  for larger batches keep the current optimized dispatch"
```

**编译版本的 TopK 只在 batch=1 时有收益**——因为大批次下手写的融合 kernel 更好。

### `--enable-waterfill` 的共享专家均衡

```python
_simulate_balanced_routing_kernel（:421）
_apply_waterfill（:611）
```

---

## 6. 量化的两阶段契约

```python
# layers/quantization/base_config.py
class QuantizeMethodBase(ABC):
    def create_weights(self, layer, *weight_args, **extra_weight_attrs): ...
    @abstractmethod
    def apply(self, layer, *args, **kwargs) -> torch.Tensor: ...
    def process_weights_after_loading(self, layer: nn.Module) -> None: return

class LinearMethodBase(QuantizeMethodBase):
    """adds input_size_per_partition, output_partition_sizes"""

class FusedMoEMethodBase(QuantizeMethodBase):
    """adds create_moe_runner, get_triton_quant_info, get_moe_quant_info"""

class QuantizationConfig(ABC):
    weight_block_size: Optional[List[int]] = None
```

### 完整流程

```
1. QuantizationConfig.get_quant_method(layer, prefix)
   → 选一个 method（跳过的层返回 UnquantizedLinearMethod）

2. create_weights(layer, ...)                    ← 建模型时
   → 分配量化参数，按 output_partition_sizes 做 TP 切分

3. checkpoint 加载

4. process_weights_after_loading(layer)          ← 一次性重排
   → 转置、swizzle、block scale 布局、marlin interleave

5. 每次 forward：method.apply(layer, x, bias)
```

**这个设计的价值**：一份模型定义同时支持 fp8/awq/gptq/mxfp4/不量化，
模型代码里零分支。

---

## 7. `Fp8LinearMethod`：布局重排的具体例子

```python
# layers/quantization/fp8.py:250 Fp8Config / :477 Fp8LinearMethod
```

两种 flavor：

- **per-tensor**：`layer.weight_scale`
- **block quant**（DeepSeek 风格 128×128 块）：`layer.weight_scale_inv` +
  `self.weight_block_size`

`apply()`（:1123）分派到**初始化时就选好**的 linear backend：

```python
# :529
self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear(...)
# 候选：
#   deepgemm_w8a8_block_fp8_linear_with_fallback
#   aiter_w8a8_block_fp8_linear
#   torch_w8a8_block_fp8_linear
```

### MXFP8 视图的 scale 布局分派

```python
if backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl():
    weight_scale = layer.weight_scale_inv_swizzled
elif backend.is_flashinfer_trtllm():
    weight_scale = layer.weight_scale_inv_shuffled
elif backend.is_deep_gemm():
    weight_scale = layer.weight_scale_inv_deepgemm
    extra_kwargs["weight_scale_swizzled"] = layer.weight_scale_inv_swizzled
```

**同一份 checkpoint 被重排成所选 kernel 想要的 scale 布局，
只做一次**（`process_weights_after_loading_block_quant`，:721），
**热路径上零布局转换**。

还有按 token 数触发的 prefill autotune（`mxfp8_prefill_autotune_min_tokens`）。

---

## 8. 量化方法清单

`BASE_QUANTIZATION_METHODS`（`layers/quantization/__init__.py`），约 28 项：

```
fp8, mxfp8, blockwise_int8, modelopt, modelopt_fp8, modelopt_fp4, nvfp4_online,
modelopt_mixed, w8a8_int8, w8a8_fp8, awq, awq_marlin, bitsandbytes, gguf,
gptq_marlin, moe_wna16, compressed-tensors, w4afp8, petit_nvfp4, quark,
quark_mxfp4, auto-round, auto-round-int8, modelslim, quark_int4fp8_moe,
humming, mxfp_w4a8, unquant
```

平台条件性追加：

```python
# cpu/cuda/gfx95/xpu
"mxfp4" → Mxfp4Config
# NPU
"gptq"   → GPTQAscendConfig
"mxfp4"  → Mxfp4W4A4Config
```

### `Mxfp4Config`（`mxfp4.py:290`）

- OCP MXFP4：E2M1 元素 + E8M0 的 g32 scale
- `get_min_capability() == 80`
- 支持的激活 dtype：bf16 / fp16
- `is_checkpoint_mxfp4_serialized`：区分 `quant_method` 字符串里含 `mxfp4` 的 checkpoint
- **HIP 上要求 gfx95**（`is_gfx95_supported()`），否则报错

MoE 侧的实现分散在：`mxfp4_marlin_moe.py`、`mxfp4_flashinfer_cutlass_moe.py`、
`mxfp4_flashinfer_trtllm_moe.py`、`mxfp4_humming_moe.py`、`aiter_mxfp4_w4a8_moe.py`。

### GEMM runner 选择器

```python
FP8_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass",
    "flashinfer_deepgemm", "flashinfer_cutedsl", "cutlass", "triton", "aiter",
]
FP4_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto", "flashinfer_cudnn", "flashinfer_cutedsl", "flashinfer_cutlass",
    "flashinfer_trtllm", "marlin",
]
MXFP8_MOE_RUNNER_BACKEND_CHOICES = [
    "cutlass", "deep_gemm", "flashinfer_megamoe", "flashinfer_trtllm",
    "flashinfer_trtllm_routed",
]
```

对应 flag：`--fp8-gemm-backend`、`--fp4-gemm-backend`、`--moe-runner-backend`。

---

## 9. torch.compile 与 Piecewise CUDA Graph

### attention 是切分边界

```python
# compilation/compilation_config.py
SPLIT_OPS = []

def register_split_op(op_name: Optional[str] = None):
    def decorator(op_func: Callable):
        name = op_name or op_func.__name__
        SPLIT_OPS.append(f"sglang.{name}")
        return op_func
    return decorator
```

`RadixAttention.forward` 用了 `@register_split_op`。于是：

```
层结构：[norm + qkv + attn + o_proj + MoE] × 60
                    ↑ 切分点
torch.compile 把每两个 attention 之间的 op 编译成一个 piece
```

**为什么在 attention 处切？**

因为 attention 是**唯一不能编译的 op**：
- 它调用外部 kernel（flashinfer / FA3 / ...），torch.compile 无法 trace
- 它有动态形状（KV 长度）
- 它需要 `init_forward_metadata` 的主机侧逻辑

其他所有 op（norm、GEMM、MoE、all-reduce）都是静态形状、可编译的。

### horizontal fusion

```python
# configure_inductor()
import torch._inductor.config as inductor_config
# Horizontal fusion for sibling ops with different shapes,
# e.g. fusing q_norm + k_norm into a single triton kernel.
if hasattr(inductor_config, "combo_kernels"):
    inductor_config.combo_kernels = True
```

**`q_norm` 和 `k_norm` 形状不同，本来不能融合，开 `combo_kernels` 后可以**
（一个 triton kernel 里放两个独立的 tile）。

### `cuda_piecewise_backend.py` 的 `ConcreteSizeEntry`

```python
class ConcreteSizeEntry:
    need_to_compile      # 在 compile_sizes 里
    use_cudagraph        # 在 cudagraph_capture_sizes 里
    cudagraph
    output
    input_addresses      # "for cudagraph debugging, track the input addresses
                         #  during capture, and check that they are the same
                         #  during replay"
```

**`input_addresses` 是一个很实用的调试手段**：CUDA Graph 要求输入 buffer 地址
固定不变。记录地址并在 replay 时比对，能立刻定位"输入 buffer 被重新分配了"
这类 bug。

### flag

| Flag | 默认 | 说明 |
|---|---|---|
| `--enable-torch-compile` | False | "Optimize the model with torch.compile. Experimental feature." |
| `--enable-torch-compile-debug-mode` | False | |
| `--torch-compile-max-bs` | 32 | 只对小 batch 编译 |
| `--cuda-graph-tc-compiler` | `eager` | `eager` / `inductor` |

官方调参文档：

> torch.compile accelerates small models on small batch sizes.

---

## 10. 量化决策表

| 你的情况 | 建议 | 理由 |
|---|---|---|
| NVIDIA H100/H200，要通用 | `--quantization fp8` | 兼容性最好，decode 收益 ~1.5× |
| 显存极度紧张（INT4） | `awq` / `gptq_marlin` | 权重 1/4 |
| Blackwell，追极致 | `modelopt_fp4` / `nvfp4_online` | FP4 tensor core |
| DeepSeek 类（已有 FP8 checkpoint） | `fp8` + block quant | 直接用官方权重 |
| AMD MI300 | `quark` / `quark_mxfp4` | |
| 只想省显存不想重训 | `fp8`（在线量化） | 加载时转换 |
| **只想提并发** | **`--kv-cache-dtype fp8_e4m3`** | **最便宜，KV 减半** |

### 一个反直觉的点

**量化的收益是 decode-specific 的。**

| 阶段 | 量化收益 |
|---|---|
| Decode（memory-bound） | 权重字节减半 → 算术强度翻倍 → **近似线性加速** |
| Prefill（compute-bound） | **没有收益**，dequant 可能反而变慢 |

**所以如果你的负载是长 prompt 短输出（分类、抽取），量化帮助有限。**
长输出（对话、生成）才是量化的主战场。

---

## 11. 自检

1. 分组 GEMM 的两种范式各适合什么 batch 规模？
2. hpc_ops 为什么要求每个 MoE 层都选它？
3. `TopKOutputFormat.BYPASSED` 省掉了多少数据搬运？
4. `process_weights_after_loading` 为什么不能放到 forward 里做？
5. torch.compile 为什么必须在 attention 处切分？
6. `combo_kernels` 让什么变得可以融合？
7. 量化对 prefill 和 decode 的收益为什么不同？

---

下一页 → [12 投机解码](12_speculative_decoding.md)

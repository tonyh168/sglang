# 08 · 投机解码、量化与其他加速件

> 对应进阶 [11 MoE 与量化](../advanced/11_moe_and_quantization.md)、
> [12 投机解码](../advanced/12_speculative_decoding.md)。

---

## 8.1 投机解码：用额外算力换更少的串行步数

### 核心思想

Decode 是**串行**的：出 token 1 才能算 token 2。每次都要读一遍全部权重，
而 batch=1 时 GPU 算力利用率只有 0.3%。

**投机解码**：先用一个便宜的小模型（draft）猜出接下来 k 个 token，
再让大模型（target）**一次前向验证这 k 个 token**。

```
朴素 decode（4 步，4 次大模型前向）：
  大模型 → t1    大模型 → t2    大模型 → t3    大模型 → t4
  ████████████   ████████████   ████████████   ████████████

投机解码（1 次大模型前向 + 便宜的 draft）：
  draft: t1' t2' t3' t4'   （便宜，小模型）
  大模型: [验证 t1' t2' t3' t4'] → t1 t2 接受，t3 拒绝，修正为 t3
  ████████████
  一次前向产出 3 个 token
```

### 为什么可行

**验证 k 个 token 的成本 ≈ 生成 1 个 token 的成本**（batch 从 1 变 k+1，
但权重只读一遍，算力利用率从 0.3% 提升到 k×0.3%）。

所以只要**接受率**够高，就能获得接近 `1 + 接受长度` 倍的加速。

### SGLang 支持的算法

```python
# speculative/spec_info.py:33
class SpeculativeAlgorithm(Enum):
    DFLASH = auto()
    UNO = auto()
    DSPARK = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()
```

| 算法 | 说明 |
|---|---|
| `EAGLE` / `EAGLE3` | 最常用。draft 模型复用 target 的 hidden states，接受率高 |
| `NEXTN` | **EAGLE 的别名**（`_RESERVED_ALIASES`），即 MTP（Multi-Token Prediction） |
| `NGRAM` | 不用神经网络，用 n-gram 查表。零额外显存，适合有重复模式的文本 |
| `STANDALONE` | 独立的 draft 模型（如小号同族模型） |
| `DFLASH` / `DSPARK` / `UNO` / `FROZEN_KV_MTP` | SGLang 自研的变体 |

```bash
--speculative-algorithm EAGLE
--speculative-draft-model-path <draft-model>
--speculative-num-steps 3
--speculative-eagle-topk 8
--speculative-num-draft-tokens 16
```

### 在调度器里怎么接

新增了两个 `ForwardMode`：

```python
# forward_batch_info.py:177
TARGET_VERIFY       # 大模型验证 draft 猜的 token
DRAFT_EXTEND_V2     # draft 模型的扩展
```

`eagle_worker_v2.py:1393` 的 verify 流程：

```python
with plan_stream_ctx:                              # 1. 独立 stream 上准备 verify
    if plan_stream is not None:
        plan_stream.wait_stream(fwd_stream)
    verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(...)
# 2. 回主 stream 前修正依赖 draft 输出的量
torch.get_device_module(device).current_stream().wait_stream(plan_stream)
target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(...)
# 3. target verify forward
forward_batch_output = target_worker.forward_batch_generation(..., is_verify=True)
# 4. 采样 + 接受判定
predict, accept_lens, accept_index = eagle_sample(...)
```

**第 2 步的注释解释了必要性**：

```
custom_mask and position depend on the output of draft,
so the previous plan step used the wrong values
```

draft 的输出决定了 tree attention mask 和位置编码，所以 plan stream 上算的那版是错的，
必须用正确的值重算。

### 观测指标

```
spec_accept_length    ← 平均每步接受多少 token。这是唯一重要的数
spec_accept_rate
spec_num_steps / spec_num_draft_tokens   ← 自适应投机时当前使用的档位
```

**`spec_accept_length` < 2 时，投机解码基本是负收益**（额外算力抵不过收益）。
此时应该：
- 换 draft 模型
- 降低 `--speculative-num-steps`
- 或直接关掉

---

## 8.2 量化：用精度换显存和带宽

### 为什么对 decode 收益巨大

回顾第 01 章的 Roofline：**decode 是 memory-bound，算术强度 ≈ batch**。

```
FP16 → FP8：权重字节数减半 → 算术强度翻倍 → decode 近似线性加速
FP16 → INT4：字节数减到 1/4
```

对 prefill（compute-bound）**没有收益**，反而可能因为 dequant 开销变慢。

### 两套接口

```python
# layers/quantization/base_config.py
class QuantizeMethodBase(ABC):
    def create_weights(self, layer, *weight_args, **extra_weight_attrs): ...
    @abstractmethod
    def apply(self, layer, *args, **kwargs) -> torch.Tensor: ...
    def process_weights_after_loading(self, layer: nn.Module) -> None: return

class QuantizationConfig(ABC):
    weight_block_size: Optional[List[int]] = None
```

流程：

```
1. get_quant_method(layer, prefix)        → 选一个 method（跳过的层返回 Unquantized）
2. create_weights(...)                    → 建模型时分配量化参数（按 TP 切分）
3. process_weights_after_loading(layer)   → checkpoint 落地后的一次性重排
                                             （转置、swizzle、block scale 布局、marlin interleave）
4. 每次 forward: method.apply(layer, x, bias)
```

**这个设计的好处**：一份模型定义同时支持 fp8/awq/gptq/mxfp4/不量化，
模型代码里没有任何 `if quant_type == ...` 分支。

### `process_weights_after_loading` 里发生了什么

以 block FP8 为例（`fp8.py:721`）：

```python
if backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl():
    weight_scale = layer.weight_scale_inv_swizzled
elif backend.is_flashinfer_trtllm():
    weight_scale = layer.weight_scale_inv_shuffled
elif backend.is_deep_gemm():
    weight_scale = layer.weight_scale_inv_deepgemm
    extra_kwargs["weight_scale_swizzled"] = layer.weight_scale_inv_swizzled
```

**同一份 checkpoint 被重排成所选 kernel 想要的 scale 布局，只做一次**，
热路径上没有任何布局转换。

### 支持的方法（`--quantization` 的 choices，约 28 种）

```
fp8, mxfp8, blockwise_int8, modelopt, modelopt_fp8, modelopt_fp4, nvfp4_online,
modelopt_mixed, w8a8_int8, w8a8_fp8, awq, awq_marlin, bitsandbytes, gguf,
gptq_marlin, moe_wna16, compressed-tensors, w4afp8, petit_nvfp4, quark,
quark_mxfp4, auto-round, auto-round-int8, modelslim, quark_int4fp8_moe,
humming, mxfp_w4a8, unquant
```

常用选择：

| 场景 | 推荐 |
|---|---|
| NVIDIA Hopper/Blackwell，要通用 | `--quantization fp8` |
| 显存极度紧张 | `--quantization awq` 或 `gptq_marlin`（INT4） |
| Blackwell，追求极致 | `modelopt_fp4` / `nvfp4_online`（FP4） |
| AMD | `quark` / `quark_mxfp4` |
| 只想省钱不想重训 | `fp8`（在线量化，加载时转换） |

### KV cache 量化：最便宜的并发提升

```bash
--kv-cache-dtype fp8_e4m3
```

KV 字节数**直接减半** → `max_total_num_tokens` 翻倍 → 并发放大一倍。
代价只是少量精度损失。

源码在 `mem_cache/kv_cache_dtype.py` 和 `layers/quantization/kv_cache.py`。
注意 `BaseKVCacheMethod` 其实**不做计算**：

```python
def apply(self, layer):
    raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")
```

它只是个 **scale 载体**：建 `k_scale` / `v_scale` 参数，在
`process_weights_after_loading` 里填值，然后由 attention 层的 forward 消费。

### GEMM / MoE kernel 后端

```bash
--fp8-gemm-backend deep_gemm      # 或 flashinfer_trtllm / cutlass / triton / aiter
--fp4-gemm-backend flashinfer_cutedsl
--moe-runner-backend deep_gemm    # 或 triton / cutlass / flashinfer_trtllm / marlin ...
```

choices 见 `arg_groups/choices.py`。

---

## 8.3 MoE：为什么需要专门的 kernel

### 分组 GEMM 的两种做法

一个 MoE 层把每个 token 路由到 top-k 个专家（DeepSeek-V3 是 256 选 8）。
token→专家是数据相关的，所以：

**(a) 按专家排序 + 每个专家一个 GEMM**（grouped GEMM）
- 算术强度好，但小 batch 时每个专家的 token 数很少且参差不齐

**(b) 把所有 token 聚成一个连续 buffer + 一个"块稀疏" GEMM**（每块带 expert id）
- 小 batch 时更优

这正是 Triton `fused_moe` kernel（用 `moe_align_block_size` 构建
`sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`，然后用一个 kernel
按块迭代 + 逐块查专家）与 DeepGEMM/CUTLASS/TRT-LLM grouped-GEMM 变体的分野。

### 为什么 kernel 选择依赖 EP 模式

EP 改变了「每卡的 token 数」和「GEMM 能否与 all-to-all 重叠」：

| EP 模式 | 说明 |
|---|---|
| TP-only MoE（`MoeA2ABackend.NONE`） | 每卡持有全部专家，最后一次 all-reduce，不需要 dispatch |
| EP + DeepEP / DeepEP_v2 | token all-to-all 到持有专家的 rank；deep_gemm 直接消费 dispatch kernel 的 ragged 布局，不重新排列 |
| EP + MegaMOE / FlashInfer MegaMoE | dispatch + GEMM + combine 融成一个 megakernel，没有独立 dispatcher |
| PPLX / NIXL / Mooncake / Mori | 同一 all-to-all 的不同传输后端 |

### 路由输出的四种格式

```python
# moe/topk.py:322 TopKOutputFormat
STANDARD    → StandardTopKOutput(weights, ids, router_logits)
BYPASSED    → 不实体化路由结果，让融合后端直接消费
TRITON_KERNEL → TritonKernelTopKOutput
PACKED      → PackedTopKOutput
```

**`BYPASSED` 的意思是：连 `(weights, ids)` 都不生成**，直接让下游 kernel 消费
router logits。省掉一次 `[tokens, num_experts]` 的中间张量往返。

---

## 8.4 torch.compile 与 Piecewise CUDA Graph

```bash
--enable-torch-compile
--torch-compile-max-bs 32      # 默认 32，只对小 batch 编译
```

### 工作方式：attention 是切分边界

```python
# compilation/compilation_config.py
SPLIT_OPS = []

def register_split_op(op_name: Optional[str] = None):
    def decorator(op_func):
        SPLIT_OPS.append(f"sglang.{name}")
        return op_func
    return decorator
```

`RadixAttention.forward` 用了 `@register_split_op`。于是：

```
层结构：  [norm + qkv + attn + o_proj + MoE]  × 60
                   ↑ 切分点
torch.compile 把每两个 attention 之间的所有 op（norm、投影、MLP/MoE、all-reduce）
编译成一个 piece
```

这就是 `tc_piecewise` CUDA Graph 后端（`runner_backend/tc_piecewise_cuda_graph_backend.py`）
的含义：**图在 attention 层处被切开，每段内部再捕获子图**。

### 一个水平融合的例子

```python
# compilation_config.py configure_inductor()
# Horizontal fusion for sibling ops with different shapes,
# e.g. fusing q_norm + k_norm into a single triton kernel.
if hasattr(inductor_config, "combo_kernels"):
    inductor_config.combo_kernels = True
```

**q_norm 和 k_norm 形状不同，本来不能融合，但开 `combo_kernels` 后可以**
（一个 triton kernel 里两个 independent tile）。

### 什么时候用

官方调参文档：

> torch.compile accelerates small models on small batch sizes.

**大模型 / 大 batch 收益有限甚至为负**（编译后的 kernel 可能不如手写 kernel）。

---

## 8.5 其他特性一览

| 特性 | 参数 | 一句话 |
|---|---|---|
| **LoRA** | `--enable-lora` | 一个基座 + 多个 adapter，adapter 切换几乎零成本 |
| **多模态** | 自动检测 | VLM 的 ViT 部分也有 CUDA Graph（`vit_cuda_graph_runner.py`） |
| **结构化输出** | `--grammar-backend` | JSON schema 约束，用 FSM 在采样时 mask |
| **DP attention** | `--enable-dp-attention` | 见第 07 章 |
| **EPLB** | `--enable-eplb` | 专家负载均衡，动态调整专家分布 |
| **权重热更新** | `/update_weights_from_tensor` | RL 训练场景，不重启更新权重 |
| **确定性推理** | `--enable-deterministic-inference` | 关掉所有非确定性 kernel，用于调试/对齐 |
| **Session** | `/open_session` | 长会话的 KV 固定占用，避免被驱逐 |
| **Router** | `sgl-model-gateway` | 独立的网关，比 `--dp-size` 更好的 DP 方案 |

---

## 8.6 本章自检

1. 投机解码为什么能加速？`spec_accept_length` 低于多少就不值得？
2. 量化对 prefill 和 decode 的收益为什么不同？
3. `process_weights_after_loading` 做的是什么？为什么不在 forward 里做？
4. `TopKOutputFormat.BYPASSED` 省掉了什么？
5. torch.compile 在 SGLang 里在哪里切分？为什么在那里切？
6. 你的 KV 快满了，最便宜的扩容手段是什么？

---

下一页 → [09 性能测量与调参](09_benchmark_and_tuning.md)

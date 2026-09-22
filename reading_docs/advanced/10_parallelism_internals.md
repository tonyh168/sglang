# 进阶 10 · 并行策略内部实现

> 对应入门 [07 并行策略](../beginner/07_parallelism.md)。
> 源码：`distributed/parallel_state.py`、`layers/dp_attention.py`、
> `layers/communicator.py`、`disaggregation/`、`arg_groups/fields/parallel.py`

---

## 1. `GroupCoordinator`：collective 的路由中心

```python
# distributed/parallel_state.py:243
class GroupCoordinator:
    """Wrap a ProcessGroup and route each collective to the best implementation.

    use_pynccl / use_pymscclpp / use_custom_allreduce /
    use_torch_symm_mem_all_reduce / use_message_queue_broadcaster
    """
```

### 核心设计：CPU group 与 device group 分离

```python
self.cpu_group        # gloo，做控制面
self.device_group     # NCCL/RCCL，做数据面
```

**这个分离是让控制面广播（每步的 batch 元数据）不走 GPU 的原因。**

如果每步的"这个 batch 有多长"这类元数据都要走 NCCL，
就会在 GPU 上留下一个没必要的小 kernel，并且强制一次同步。

### 主要方法

```python
def all_reduce(self, input_) -> torch.Tensor: ...              # :660
def quant_all_reduce(self, input_) -> torch.Tensor: ...        # :772  FP8 all-reduce
def fused_allreduce_rmsnorm(self, ...): ...                    # :786  ★
def fused_allreduce_rmsnorm_quant_per_group(self, ...): ...    # :847
def reduce_scatter_tensor(self, output, input): ...            # :1099
def all_gather_into_tensor(self, output, input): ...           # :1320
def all_gatherv / reduce_scatterv(...): ...                    # :1405 / :1218
def broadcast_object / all_gather_object / send_object(...): ...
def send_tensor_dict / recv_tensor_dict / broadcast_tensor_dict(...): ...
```

**`v` 后缀的变体**（`all_gatherv` / `reduce_scatterv`）承载**变长数据**——
这正是 DP attention 需要的（每个 DP rank 的 token 数不同）。

---

## 2. `fused_allreduce_rmsnorm`：旗舰优化

```python
# distributed/parallel_state.py:786
def fused_allreduce_rmsnorm(self, input_, residual_, weight_, eps):
    """Attempt fused all-reduce + RMSNorm via custom all-reduce communicator. ROCm/HIP Only"""
    ca_comm = self.ca_comm
    if ca_comm is None or getattr(ca_comm, "disabled", True):
        return None
    ...
    # 1-stage vs 2-stage selection for fused AR+RMSNorm:
    # The 1-stage kernel launches one block per token and is capped at 80 tokens (kMaxBlocks).
    # Guard with a byte threshold so large prefill batches fall through to the 2-stage
    # kernel instead of hitting a runtime error.
    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        use_1stage_ar = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
    else:
        total_bytes = input_.numel() * input_.element_size()
        use_1stage_ar = total_bytes <= 128 * 1024
    fused_outputs = ca_comm.custom_fused_ar_rms(_, residual_inp_, weight_, eps, use_1stage_ar)
```

### 为什么这个融合重要

```
朴素 TP 层：
  all-reduce：
    读其他 rank 的数据 → 写 [tokens, hidden] 到本卡 HBM
  RMSNorm：
    从 HBM 读回 [tokens, hidden]
  ─────────────────────────────────
  一次完整的 HBM 往返（写 + 读）

融合：
  reduce 的结果直接落在 norm kernel 的寄存器里
  省掉一次 [tokens, hidden] 的读 + 写
```

### 具体数字

```
60 层 × 2 个 norm × 8192 token × 7168 hidden × 2 Byte
= 60 × 2 × 8192 × 7168 × 2
≈ 14 GB
```

**每次前向省掉 ~14 GB 的 HBM 流量。** 按 3 TB/s 算，约 4.7 ms。

### 1-stage vs 2-stage

**1-stage**：每个 token 一个 block，上限 80 个 token（`kMaxBlocks`）。
超过会**运行时错误**，所以必须用字节阈值守卫。

```python
total_bytes = input_.numel() * input_.element_size()
use_1stage_ar = total_bytes <= 128 * 1024
```

**128 KiB 阈值**，所以大 prefill 批自动落到 2-stage。

**源码注释里的设计原则值得学**：不是"1-stage 更快所以都用它"，
而是"1-stage 在它有寄存器容量的范围内更快，超了就走 2-stage"。

### 扩展到 FP8 量化

```python
# fused_allreduce_rmsnorm_quant_per_group，:847
# AR + RMSNorm + per-group FP8 quant 一个 kernel
# gfx95 上有个 tuning 例外：
# "the measured TP=8 K=7168 graph-replay crossover"（:905-912）
```

**这个 tuning 例外说明了什么**：性能优化的最优点往往由一个具体的
工程测量坐标决定（TP=8、K=7168、graph replay 模式），而不是一个通用公式。

---

## 3. `ScatterMode`：TP/DP 间的数据布局

```python
# layers/communicator.py:223
class ScatterMode:
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively

    SCATTERED:     [a, b, c, d]
    TP_ATTN_FULL:  [ab, ab, cd, cd]  — all ranks inside a TP attn group have full data
    FULL:          [abcd, abcd, abcd, abcd]
    MOE_FULL:      full within the MoE group (cp_per_moe CP chunks),
                   used when moe_dp_size < attn_cp_size
    """
```

### `LayerCommunicator`（:518）

```python
class LayerCommunicator:
    def __init__(self, ..., input_mode, residual_mode, ...):
        ...
```

**职责：用最少的 collective 在这些模式之间搬运 hidden states 和 residual。**

### `CommunicateWithAllReduceAndLayerNormFn`

按 `(input_mode, residual_mode, output_mode, residual_output_mode)` 四元组
选一个函数。标准情况（`_tp_attn_all_reduce_and_layernorm`，:1220）：

```python
def _tp_attn_all_reduce_and_layernorm(...):
    hidden_states = get_parallel().attn_tp_group.all_reduce(hidden_states)
    if hidden_states.shape[0] != 0:
        hidden_states, residual = layernorm(hidden_states, residual)
    return hidden_states, residual
```

**所以 4 种模式的组合 × 融合策略 = 一整套专化函数。**
这是 SGLang 里"用组合爆炸换零分支开销"的一个典型例子。

### 其他融合点

```python
attention_tensor_model_parallel_quant_all_reduce(...)   # FP8 all-reduce + per-group scale
_tp_all_reduce_with_scattered_residual(...)             # :1343  DP attention 用
apply_aiter_all_reduce_fusion(...)                      # :208   AMD
flashinfer_allreduce(...)                               # parallel_state.py:195
can_merge_post_experts_all_reduce / deferred_...        # :61-62  与下一层合并
```

最后这个值得注意：**把 MoE 之后的 all-reduce 与下一层的 all-reduce 合并**——
两次小的 all-reduce 合成一次大的，启动开销减半，带宽利用率更高。

---

## 4. DP attention：并行度解耦

### 核心洞察

```
Attention 层：权重小（KV 头少），激活大 → 通信量大
MoE 层：      权重大（几十 GB）        → 通信量小但计算量大
```

### 三个判据（`utils/common.py:3958-4050`）

```python
def require_mlp_tp_gather():
    """Check if the input of MLP is obtained by all-gather rather than all-reduce.
    This only happens when each MLP TP group contains multiple attention DP groups."""
    if get_parallel().enable_dp_attention:
        if get_exec().moe.elastic_ep_backend is not None and elastic_expanded_world_enabled():
            return True
        if get_parallel().moe_dense_tp_size is None:      # 有些 MoE 模型没有 dense 层
            return True
        elif not get_parallel().enable_dp_lm_head:
            return True
        elif get_moe_a2a_backend().is_none():
            return True
        elif get_moe_a2a_backend().is_flashinfer():
            return True
        else:
            return get_parallel().moe_dense_tp_size > get_parallel().tp_size // get_parallel().dp_size
    else:
        return False

def require_attn_tp_gather():
    """Check if the input of attention is scattered."""
    if get_parallel().disable_attn_tp_gather:
        return False
    if not get_moe_a2a_backend().is_none() or get_parallel().moe_dense_tp_size is not None:
        if get_parallel().enable_dp_attention:
            return get_parallel().dp_size < get_parallel().tp_size
        else:
            return True
    else:
        return False

def require_gathered_buffer():
    return require_mlp_tp_gather() or require_attn_tp_gather()

def require_mlp_sync():
    return get_parallel().enable_dp_attention or require_gathered_buffer()
```

### flashinfer 分支的长注释值得完整引用

```python
# :3983-3990
# FlashInfer MoE A2A needs a rank-invariant, DP-synchronized per-rank token count:
# MoeAlltoAll uses fixed-geometry buffers and the decode cuda-graph bucket must be
# identical across EP ranks, otherwise ranks replay different-sized graphs ->
# geometry mismatch -> illegal memory access (issue #30242). No literal MLP
# TP-gather happens here -- the MoE stays SCATTERED and the a2a op owns
# dispatch/combine -- but we reuse this flag's DP-sync bookkeeping.
```

**这解释了为什么这个 flag 名字有误导性但还是被复用了**：
它不是字面意义上的 "MLP TP gather"，而是复用了这个 flag 的
"DP 同步记账"语义。

**而这个语义是必需的**——因为 FlashInfer 的 MoE A2A 用固定几何的 buffer，
如果各 EP rank replay 不同尺寸的 CUDA Graph，就会 **illegal memory access**。

### gather/scatter 的开销（`layers/dp_attention.py:585`）

```python
get_parallel().attn_tp_group.reduce_scatter_tensor(...)
...
torch.distributed.all_gather_into_tensor(...)
```

### `DpPaddingMode`（:114-172）

三种模式，对应不同的通信原语：

| 模式 | 做法 | 代价 |
|---|---|---|
| pad 到 `max_len` | `all_gather_into_tensor` | 填充浪费 |
| pad 到 `sum_len` | zero-buffer `all_reduce` | 填充浪费 |
| **不 padding** | `all_gatherv`（:601, :704） | 无浪费 |

注释：

> The combine leg (reduce_scatterv) stays bf16: NCCL SUM cannot run on fp8.

**NCCL 的 SUM 不支持 FP8**，所以 combine 那一段必须保持 bf16。

### 静态 gather buffer

```python
# _DpGatheredBufferWrapper（:175-298）
# 元数据只设一次 → CUDA Graph 可捕获
```

### 一个控制面优化

```python
# arg_groups/fields/parallel.py:177
enable_dp_attention_local_control_broadcast: A[bool,
    "With DP-attention, send control messages to every DP group leader and
     broadcast within attn_tp_group instead of the full tp_group.
     Eliminates a costly all-ranks gloo sync on every scheduler iteration."]
```

**每个 scheduler 迭代省一次全 rank 的 gloo 同步。** 在 decode 的
每步 ~20ms 预算里，这类控制面同步是纯粹的成本。

---

## 5. PD 分离：两侧的生命周期

### Prefill 侧（`disaggregation/prefill.py:1-13`，原文）

```
1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation)
       has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
```

### Decode 侧（`disaggregation/decode.py:1-15`，原文）

```
1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
```

### 第 3 步是精髓

**decode 节点从不跑 prefill forward**，只构造一个 `PrebuiltExtendBatch` 来填充
`req_to_token` / `out_cache_loc` 元数据。

```
"prefill" 的代价 = 一次元数据填充，而不是一次完整前向
```

### 重叠点

- **1b（preallocation）和 2a（transfer polling）发生在 decode 批正在跑的时候**
- prefill 侧的 `maybe_send_cached_prefix_chunk`（`scheduler.py:4346`）在尾部
  forward 之前就把已缓存的 prefix KV 推过去——**与 forward 重叠**

### `--num-reserved-decode-tokens`

默认 512。decode 节点必须在传输完成**之前**就为请求预留 KV 槽位，所以要留预算。

```python
# arg_groups/fields/disagg.py:94-96
```

### KV 校验

```python
# disaggregation/checksum.py
disaggregation_enable_kv_checksum
```

对每个请求的 KV 页算 **Adler-32**，decode 侧验证。

**为什么需要？** 跨机传输没有 ECC 保证，静默错误会导致输出莫名其妙地错。
校验和让这类问题立刻暴露。

### 传输后端

`--disaggregation-transfer-backend` 的 choices（`arg_groups/choices.py:126`）：

```python
["mooncake", "nixl", "ascend", "fake", "mori", "mooncake_tcp"]
```

**默认是 `mooncake`**，不是 nixl。

`fake` 后端用于测试（不做真实传输）。

---

## 6. 并行度派生：attention 和 MoE 可以不同

```python
# arg_groups/fields/parallel.py:284-305
attn_tp_size
attn_dp_size
attn_dcp_size
moe_ep_size
moe_tp_size
```

**这是 SGLang 并行配置的核心抽象**：attention 和 MoE/FFN 的并行宽度是独立的。

### 完整参数表

| Flag | 默认 | 说明 |
|---|---|---|
| `--tp-size`（`--tensor-parallel-size`） | 1 | 张量并行 |
| `--pp-size`（`--pipeline-parallel-size`） | 1 | 流水线并行 |
| `--dp-size`（`--data-parallel-size`） | 1 | 数据并行 |
| `--ep-size`（`--expert-parallel-size`, `--ep`） | 1 | 专家并行 |
| `--moe-dp-size` | — | MoE 的数据并行（当 `moe_dp_size < attn_cp_size` 时用 MOE_FULL） |
| `--dcp-size`（`--decode-context-parallel-size`） | 1 | decode 的上下文并行 |
| `--attn-cp-size`（`--attention-context-parallel-size`） | 1 | attention 上下文并行 |
| `--enable-dp-attention` | False | attention DP + FFN TP |
| `--enable-dp-lm-head` | False | lm_head 也做 DP |
| `--enable-tp-lm-head-all-to-all` | False | lm_head 走 all-to-all |
| `--moe-dense-tp-size` | None | MoE 里 dense 层的 TP size |
| `--load-balance-method` | `auto` | `round_robin` / `follow_bootstrap_room` / `total_requests` / `total_tokens` |
| `--nnodes` / `--node-rank` | 1 / 0 | 多机 |
| `--nccl-init-addr` | — | 多机初始化 |
| `--gated-launch-port` | None | 延迟启动（把耗内存的阶段推迟到安全窗口） |
| `--enable-nccl-nvls` | False | NVLS（prefill 重的请求） |
| `--enable-symm-mem` | False | 对称内存 |

### `--gated-launch-port` 值得单独说

> When set, every rank blocks right after the distributed environment is
> initialized, before any sizable GPU allocation, until `POST /gate/activate`
> is sent to the port on the host of the first rank. This lets an external
> orchestrator defer the memory hungry part of startup to a safe window.

**用途**：Kubernetes 里同时启动多个实例时，避免所有实例同时申请显存导致
调度器抖动。编排器可以逐个放行。

---

## 7. 并行策略选择的影响链

改一个并行参数，会连锁影响：

```
--tp-size 8
  ↓
attn_tp_size = 8
  ↓
get_cuda_graph_batch_size_alignment() 里 alignment *= 8
  ↓
CUDA Graph 的 bucket 必须能被 8 整除 → bucket 列表变化
  ↓
可用的 batch size 粒度变粗 → padding 浪费变化
  ↓
显存占用变化 → 需要重调 --mem-fraction-static
```

**这就是为什么调并行参数之后一定要重新调 `--mem-fraction-static`。**

类似的链条还有：

```
--enable-dp-attention
  ↓
require_mlp_sync() = True
  ↓
每步一次全局 all-gather
  ↓
需要 PrefillDelayer 来避免空等（第 04 章）
  ↓
all-gather buffer 是 attn_tp_size 宽的 → CUDA Graph alignment 变化
```

---

## 8. 自检

1. `GroupCoordinator` 为什么要把 CPU group 和 device group 分开？
2. `fused_allreduce_rmsnorm` 省掉的是什么？1-stage 为什么需要字节阈值守卫？
3. `all_gatherv` / `reduce_scatterv` 的 `v` 是什么意思？谁需要它？
4. `require_mlp_tp_gather()` 里 flashinfer 分支为什么会导致 illegal memory access？
5. PD 分离下 decode 节点的 "prefill" 到底是什么？
6. `--num-reserved-decode-tokens` 为什么必要？
7. 改 `--tp-size` 为什么会连锁影响 `--mem-fraction-static`？

---

下一页 → [11 MoE 与量化](11_moe_and_quantization.md)

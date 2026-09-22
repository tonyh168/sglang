# 进阶 12 · 投机解码

> 对应入门 [08 投机解码](../beginner/08_speculative_and_advanced.md)。
> 源码：`speculative/`（40+ 文件）、`speculative/spec_info.py`、
> `speculative/eagle_worker_v2.py`、`speculative/spec_utils.py`

---

## 1. 数学与收益模型

### 为什么能加速

Decode 的算术强度 ≈ batch size（第 01 章）。batch=1 时 GPU 算力利用率约 0.3%。

**验证 k 个 token 的成本 ≈ 生成 1 个 token 的成本**：

```
生成 1 个 token：  读一遍权重（W bytes），算 [1, K] × [K, N]
验证 k+1 个 token：读一遍权重（同样 W bytes），算 [k+1, K] × [K, N]

字节数相同，算力利用率提升 (k+1) 倍
```

所以只要接受率够高，就能获得接近 `1 + accept_length` 倍的加速。

### 收益公式

```
设：
  α = 单步接受率（每个 draft token 被接受的概率）
  k = 每轮 draft 的 token 数
  c = draft 前向的成本 / target 前向的成本

期望接受长度 = (1 - α^(k+1)) / (1 - α)          （α < 1 时）
每轮时间     = target_forward × (1 + c)
吞吐比       = 期望接受长度 / (1 + c)
```

### 数值直觉

| α | k | 期望接受长度 | 需要 c < ? 才有正收益 |
|---|---|---|---|
| 0.8 | 4 | 3.36 | 2.36 |
| 0.7 | 4 | 2.77 | 1.77 |
| 0.6 | 4 | 2.18 | 1.18 |
| 0.5 | 4 | 1.94 | 0.94 |

**`spec_accept_length` < 2 时基本无收益**——因为 target forward 的形状变大了
（`[1, K]` → `[k+1, K]`），大 batch 下 target 本身就不是 memory-bound 了。

---

## 2. 算法清单

```python
# speculative/spec_info.py:33
class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and both
    expose the same ``is_*()`` / ``create_worker`` interface so callers dispatch
    uniformly without isinstance checks."""
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

| 算法 | draft 是什么 | 显存成本 | 典型接受率 |
|---|---|---|---|
| `EAGLE` / `EAGLE3` | 单层 decoder，复用 target 的 hidden states | 低 | 高（0.7–0.85） |
| `NEXTN`（= EAGLE） | DeepSeek 的 MTP 模块 | 低 | 高 |
| `NGRAM` | n-gram 查表，**无神经网络** | 零 | 依赖文本重复度 |
| `STANDALONE` | 独立的同族小模型 | 高 | 中 |
| `DFLASH` | block 级别的 draft | — | SGLang 自研 |
| `DSPARK` | gamma 个 draft token | — | SGLang 自研 |
| `UNO` | draft LoRA | 低 | — |
| `FROZEN_KV_MTP` | 冻结 KV 的 MTP | — | — |

### 插件注册

```python
@SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
def _factory(server_args):
    return MySpecWorker
```

`CustomSpecAlgo`（`spec_registry.py:36`）duck-type 了枚举值的接口，
所以调度器/ModelRunner 里的 `if spec_algorithm.is_eagle():` 分支不用改。

`_RESERVED_ALIASES` 处理别名（`NEXTN` → `EAGLE`）。

---

## 3. 新增的 `ForwardMode`

```python
# forward_batch_info.py:177
class ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()
    MIXED = auto()          # chunked prefill 时同时含 EXTEND 和 DECODE
    IDLE = auto()           # DP attention 下没活的 rank
    TARGET_VERIFY = auto()
    DRAFT_EXTEND_V2 = auto()
    ...
    DLLM_EXTEND = auto()

    def is_target_verify(self): ...
    def is_draft_extend_v2(self): ...
    def is_extend_without_speculative(self): ...
    def is_decode_or_idle(self): ...
```

调度器/ModelRunner 用这些谓词来分派，而**不是** `isinstance` 或字符串比较。

---

## 4. `EAGLEWorkerV2.forward_batch_generation`（:1393）

### prefill 分支

```python
# target prefill（CaptureHiddenMode.FULL）
# → on_publish(new_seq_lens)
# → draft prefill（_draft_extend_for_prefill）
```

注释里的一个关键约定：

> Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.

### decode 分支

```python
self.activate_step_by_batch(bs)          # 自适应投机：决定本轮的 steps
draft_output = self.draft(batch)         # → EagleVerifyInput
self.verify(...)                         # → on_publish → draft_extend
```

### `speculative_num_steps == 0` 的特殊处理

```python
_build_trivial_verify_input(...)
```

源码注释解释了为什么**大 batch 时仍然跑 draft_extend**：

> Drafting disabled (high batch size). `_draft_extend` below still runs,
> **keeping draft KV warm for when the batch shrinks.**

**这是个很细的工程考虑**：大 batch 时投机不划算（target 本身就不是
memory-bound 了），所以不做 draft；但如果完全不跑 draft_extend，
draft 的 KV 会变陈旧，等 batch 缩小需要投机时又要重新 prefill。

---

## 5. `verify`：五步流水线

`run_eagle_verify`（`eagle_worker_common.py:462-600`）：

```python
# 1. 独立 stream 上做 verify 准备
with plan_stream_ctx:
    if plan_stream is not None:
        plan_stream.wait_stream(fwd_stream)     # 依赖 draft 产出的 tree metadata
    verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(...)

# 2. 回主 stream 前修正依赖 draft 输出的量
if plan_stream:
    torch.get_device_module(device).current_stream().wait_stream(plan_stream)
    target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
        verify_input, decode_cuda_graph_runner.bs if can_run_cuda_graph else None)

grammar_tree = GrammarTree.from_device(...) if batch.has_grammar else None

# 3. target verify forward
forward_batch_output = target_worker.forward_batch_generation(
    batch=None, forward_batch=verify_forward_batch, is_verify=True, ...)

# 4. 采样 + 接受判定
predict, accept_lens, accept_index = eagle_sample(
    verify_input, batch, logits_output, grammar_mask, ...)
```

### 第 2 步为什么必须重算

源码注释：

> custom_mask and position depend on the output of draft, so the previous
> plan step used the wrong values.

**draft 的输出决定了 tree attention 的 mask 和位置编码。**

- 第 1 步在 plan stream 上做准备工作（可以提前做的部分）
- 但 mask/position 依赖 draft 的实际输出，所以第 1 步算的那版是**错的**
- 第 2 步回到主 stream，用正确的值重算

**这是一个"为了重叠而引入的错误中间态"** ——只有在明确知道哪些量会变的前提下
才能这么做。

### plan stream 的意义

```
plan_stream:  [准备下一轮的元数据]
fwd_stream:        [target verify]
                          ↑ 两者重叠
```

draft 在 plan stream 上准备下一轮，target verify 在 forward stream 上跑。

---

## 6. Tree Attention：为什么需要自定义 mask

EAGLE 的 draft 不是一条链，而是**一棵树**（`--speculative-eagle-topk` 控制分支数）。

```
draft 阶段 topk=2, steps=2：

     t0
    ╱  ╲
  t1a  t1b
  ╱ ╲  ╱ ╲
t2a t2b t2c t2d
```

验证时要把这棵树**拍平成一个 batch**，用一个自定义 mask 保证：

- 每个 token 只能看到它在树上的祖先
- 不同分支之间不能互相看见

这就是 `custom_mask` 的来源，也是为什么需要
`attn_backend.update_verify_buffers_to_fill_after_draft`。

### ragged verify

```python
# speculative/ragged_verify.py
ragged_verify_mode = (
    ragged_verify_compact_graphs_enabled(self.model_runner.spec_algorithm)
    and (self.capture_forward_mode == ForwardMode.TARGET_VERIFY)
    and not self.model_runner.is_draft_worker
)
self.capture_num_tokens = (
    self._build_ragged_verify_token_buckets() if self.ragged_verify_mode else None
)
```

**"ragged verify" 是按 num_tokens 分桶而不是按 batch size 分桶**——
因为树的大小是变化的（不同请求接受不同长度的历史），
按 token 数分桶能减少 padding 浪费。

`_captured_ragged_layouts` 里每个 tier 的 `verify_lens` / `qo_indptr`
**被烘进捕获的图里，每次 replay 时原地刷新**。

---

## 7. 调度器侧的深度集成

### publish 时机

```python
# scheduler.py:4373 run_batch
if not batch.spec_algorithm.is_none():
    fwd_kwargs["on_publish"] = partial(self.future_map.publish, future_indices)
    if batch.spec_algorithm.supports_grammar_overlap():
        fwd_kwargs["grammar_barrier"] = self._advance_pending_grammar
batch_result = self.model_worker.forward_batch_generation(batch, **fwd_kwargs)
if batch.spec_algorithm.is_none():
    self.future_map.publish(future_indices, batch.seq_lens + 1)
```

注释：

> Spec_v2 fires on_publish mid-worker (between verify and draft_extend) so
> schedule prep can overlap with draft_extend.
> Non-spec has no later work — scheduler publishes after return.

**投机路径把 publish 提前到 verify 结束、draft_extend 开始前**，
让 CPU 侧的调度准备与 draft_extend 的 GPU 时间重叠。

**只有 spec 路径能享受这个优化**——因为只有它才有"verify 之后还有工作"的结构。

### grammar 与 overlap

```python
# is_disable_overlap_for_batch（:2044）
need_grammar_sync = (
    batch
    and not batch.spec_algorithm.is_none()
    and batch.grammar_needs_sync()
    and batch.forward_mode.is_decode()
    and len(self.result_queue) > 0
)
```

但支持 `supports_grammar_overlap()` 的算法不受影响：

> Algorithms that support grammar overlap advance the FSM inside verify() via the
> grammar barrier (overlapping the target forward), which resolves whatever result
> is still pending in the queue — including the extend->decode boundary — so no
> grammar-specific overlap disable is needed.

**`grammar_barrier` 在 verify 内部推进 FSM**，所以不需要禁用 overlap。

### DP attention 下的约束

```python
# scheduler.py:3745
if (need_mlp_sync and not self.spec_algorithm.is_none()
        and not get_spec().speculative_skip_dp_mlp_sync):
    # NOTE: This branch makes sure prefill and decode batches will not be mixed
    # when spec and dp-attn is enabled.
    new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
    need_mlp_sync = new_batch is None
```

**spec + dp-attn 下不允许 prefill 和 decode 混批**——因为 verify 的形状
在两种模式下差异太大。

可用 `--speculative-skip-dp-mlp-sync` 关掉这个同步。

---

## 8. 自适应投机

```python
# speculative/adaptive_spec_params.py
# speculative/adaptive_runtime_state.py
```

相关指标：

```
spec_num_steps          ← 当前使用的 steps 档位
spec_num_draft_tokens   ← 当前使用的 draft token 数
spec_block_accept_length
```

**思想**：接受率随 workload 变化。文本重复度高时接受率高，可以多加 steps；
文本发散时接受率低，应该减 steps。

`--speculative-adaptive` 开启。调度器在每步根据实测的接受率调整档位。

### `--speculative-num-steps 0` 的语义

**不做 draft，但仍跑 draft_extend 保持 draft KV 新鲜**（见 §4）。
自适应策略在检测到"不划算"时会退到这个状态，而不是完全关掉投机。

---

## 9. 参数速查

| Flag | 说明 |
|---|---|
| `--speculative-algorithm` | `EAGLE` / `EAGLE3` / `NEXTN` / `STANDALONE` / `NGRAM` / `DFLASH` / `DSPARK` / `UNO` |
| `--speculative-draft-model-path`（`--speculative-draft-model`） | draft 权重路径 |
| `--speculative-draft-model-revision` | 版本 |
| `--speculative-draft-load-format` | `dummy` 可用于 profiling |
| `--speculative-num-steps` | draft 步数 |
| `--speculative-eagle-topk` | 每步采样的 token 数（树的宽度） |
| `--speculative-num-draft-tokens` | draft token 总数 |
| `--speculative-accept-threshold-single` / `-acc` | 接受阈值 |
| `--speculative-use-rejection-sampling` | 用拒绝采样保证分布正确 |
| `--speculative-draft-attention-backend` | draft 的 attention 后端 |
| `--speculative-draft-kv-cache-dtype` | draft 的 KV dtype |
| `--speculative-adaptive` | 自适应档位 |
| `--enable-multi-layer-eagle` | 多层 EAGLE |
| `--speculative-skip-dp-mlp-sync` | 关掉 spec + dp-attn 的同步 |
| `--speculative-dflash-block-size` | DFLASH 的 verify 窗口 |
| `--speculative-dspark-block-size` | DSPARK 的 gamma |
| `--speculative-dspark-sps-table-path` | 预 profile 的 SPS 成本表 |
| `--speculative-domino-candidate-pool-size` | Domino 的候选池（默认 2048） |

`DRAFT_ATTENTION_BACKEND_CHOICES` 单独一张表（`choices.py:107`）：

```python
["flashinfer", "fa3", "fa4", "triton", "ascend", "trtllm_mha", "intel_xpu"]
```

注释：

> trtllm_mha is valid for decode-only dense-MQA drafts. DFLASH rejects it
> earlier when its per-layer attention requirements are not met.

---

## 10. 什么时候不该用投机解码

| 情况 | 原因 |
|---|---|
| `spec_accept_length` < 2 | 负收益 |
| 大 batch（接近 compute-bound） | target 本身已经算力饱和，验证 k+1 个 token 的成本是线性的 |
| 输出高度发散（创意写作） | 接受率低 |
| 显存极度紧张 | draft 模型和它的 KV 池要额外显存（`_cell_size` 会被放大） |
| prefill 主导的负载 | 投机只优化 decode |

### 大 batch 的临界点

```
设 target forward 时间 = max(compute_time, memory_time)
  memory_time ≈ W / BW，与 batch 无关
  compute_time ≈ batch × FLOPs_per_token / peak_flops

临界 batch ≈ BW × FLOPs_per_token / (W × peak_flops) ... 简化为 ≈ I_ridge

H100 上约 295（实际受 M 维 GEMM 效率影响，通常 100–200 就开始收益递减）
```

**超过临界点后，验证 k+1 个 token 的成本变成 k+1 倍**，
而收益只有 `accept_length` 倍。所以大 batch 时投机收益迅速消失。

---

## 11. 与其他特性的交互

| 特性 | 交互 |
|---|---|
| **CUDA Graph** | verify 需要自己的图（按 num_tokens 分桶的 ragged verify） |
| **grammar** | 支持 `supports_grammar_overlap()` 的算法可以在 verify 内推进 FSM |
| **DP attention** | 默认不允许 prefill/decode 混批；有额外的 MLP sync |
| **PD 分离** | `eagle_disaggregation.py` / `dflash_disaggregation.py` 有专门实现 |
| **PD + DCP + trtllm_mla** | **不支持**（不转发 cyclic DCP 元数据，无 rank-local LSE） |
| **LoRA** | `uno_lora.py`；UNO 用 draft LoRA |
| **量化** | `--speculative-draft-kv-cache-dtype` 单独控制 draft 的 KV dtype |

---

## 12. 自检

1. 为什么"验证 k+1 个 token 的成本 ≈ 生成 1 个 token"？
2. `spec_accept_length` 低于多少就不值得？为什么？
3. verify 的第 2 步为什么必须用正确的 mask/position 重算？
4. `speculative_num_steps == 0` 时为什么还要跑 draft_extend？
5. ragged verify 为什么按 num_tokens 而不是 batch size 分桶？
6. spec 路径为什么能把 publish 提前？
7. 大 batch 时投机解码为什么失去收益？

---

下一页 → [13 调参方法论](13_tuning_playbook.md)

# 05 · ModelRunner 与一次前向

> 对应进阶 [06 ModelRunner 与 CUDA Graph](../advanced/06_model_runner_and_cuda_graph.md)、
> [07 Attention 后端与 Kernel](../advanced/07_attention_backends_and_kernels.md)。
> 源码：`python/sglang/srt/model_executor/model_runner.py`（2314 行，**frozen core file**）

---

## 5.1 ModelRunner 是什么

它是**每个 Scheduler 进程里唯一持有模型的角色**：

```
Scheduler
  └── TpModelWorker (tp_worker.py:317)
        └── ModelRunner (model_runner.py:296)
              ├── self.model            ← 真正的 nn.Module（几十/几百层）
              ├── self.sampler          ← 采样器
              ├── self.attn_backend     ← attention 后端（flashinfer/fa3/triton/...）
              ├── self.decode_cuda_graph_runner
              ├── self.token_to_kv_pool_allocator
              └── self.req_to_token_pool
Scheduler
  ├── self.tree_cache         ← RadixCache
  └── self.running_batch      ← 当前批
```

**职责边界很清楚**：
- Scheduler 管"跑什么、什么时候跑"（CPU 逻辑）
- ModelRunner 管"怎么跑"（GPU 执行）

---

## 5.2 `__init__` 的编排：顺序是有意义的

`ModelRunner.__init__`（:315）只做编排，每步一个命名方法。关键顺序：

```python
# 1. 解析参数、记录配置
self.model_config, self.server_args, self.is_draft_worker = ...
assert_published(server_args, role="scheduler")       # 运行上下文契约

# 2. 启动可观测性
self.init_startup_observability()                     # weight_load_time 等

# 3. 设备设置（必须在 TransferEngine 初始化之前）
torch.get_device_module(device).set_device(gpu_id)

# 4. 分布式初始化（注释：Get available memory before model loading）
self.init_torch_distributed()

# 5. 创建 forward stream —— overlap 调度用的那条 stream
self.forward_stream = <device>.Stream()
self.shared_read_done_event = None                    # WAR barrier 读的邮箱

# 6. 主力初始化
self.initialize()
```

而 `initialize()`（:643）里的顺序更是有讲究的：

```python
self.init_memory_saver_adapter()
self.maybe_init_remote_instance_transfer_engine()   # 必须在权重加载之前
self.maybe_init_eplb_manager()                      # 专家负载均衡
self.sampler = create_sampler()
self.load_model()                                   # ← 加载权重
prepare_moe_topk(...)
disable_routed_experts_capture_for_draft(self.model)  # 必须在 backend/graph 初始化之前
self.layer_info = resolve_layer_indices(...)        # PP 的层范围
configure_kv_cache_dtype()
```

源码注释解释了第 8 步为什么必须在这个位置：

```
# routed-experts capture must be disabled *before* backend/graph init
# so no draft graph records a capture-write kernel
```

> **读 SGLang 的一个通用技巧**：看到 `maybe_init_*` / `init_*` 方法时，
> 不要只看它做什么，要看它**为什么在这个位置**——几乎所有位置都有注释说明依赖。

### `load_model` 里的两个性能细节

```python
# model_runner.py:1178
torch.set_num_threads(1)      # "reduce thread conflicts and speed up weight loading"
...
reserve_rope_cache_for_long_sequences(...)   # 在 CUDA Graph 捕获之前预扩展 RoPE 缓存
```

第二个尤其重要：**如果 RoPE 的 cos/sin 表是懒扩展的，第一个长上下文请求会在
CUDA Graph 捕获区里做内存分配**，直接崩。所以必须在捕获前预留好。

---

## 5.3 一次前向：从 `ForwardBatch` 到 logits

```python
# model_runner.py:1682
def forward(self, forward_batch, ...):
    self.forward_pass_id += 1
    with (canary_ctx, step_span_ctx, ...):
        output = self._forward_raw(forward_batch, pp_proxy_tensors, ...)
```

`_forward_raw` 的核心分派：

```python
can_run_graph = bool(
    mode_check()
    and self.decode_cuda_graph_runner
    and self.decode_cuda_graph_runner.can_run_graph(forward_batch))

if can_run_graph:
    ret = self.decode_cuda_graph_runner.execute(forward_batch, pp_proxy_tensors=...)
    return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

self._prepare_eager_forward_batch(forward_batch)
...  # eager 路径
```

**一句话总结**：
- **Decode → CUDA Graph replay**
- **Prefill → eager**（或 piecewise prefill CUDA graph）
- `forward_idle` 是 DP 下没活的 rank 走的路

---

## 5.4 `ForwardBatch` 的字段含义

从 `ScheduleBatch` 转换而来（`ForwardBatch.init_new`）。最重要的几个：

```python
forward_mode: ForwardMode      # EXTEND / DECODE / IDLE / TARGET_VERIFY / ...
batch_size: int
input_ids: torch.Tensor        # [总 token 数]
req_pool_indices: torch.Tensor # 每个请求在 req_to_token_pool 的行号
seq_lens: torch.Tensor         # 每个请求的当前长度
out_cache_loc: torch.Tensor    # 本轮新 KV 写到池子的哪些槽位 ← 核心
seq_lens_sum: int
```

### `out_cache_loc` 是核心

```
ScheduleBatch 阶段：
    allocator.alloc_extend(...) → out_cache_loc     （第 06 章，Triton kernel）

ForwardBatch 阶段：
    attention kernel 拿到 out_cache_loc
    → 把本轮算出的 K/V 写到这些槽位
    → 同时用 req_to_token[req_pool_indices, 0:seq_len] 读历史 KV
```

**一条写入 + 一条读取的间接路径，就是"分页 KV cache"的全部。**

---

## 5.5 CUDA Graph：为什么它是 decode 的必需品

### 问题：kernel launch 开销

一个 decode step，在 TP=8、60 层的模型上：

```
每层：2 个 norm + qkv_proj + attention + o_proj + gate/up/down (MoE)
       + 2 个 all-reduce
≈ 每层 10+ 个 kernel
× 60 层 = 600+ 个 kernel
× 每个 kernel CPU 侧启动成本 2–5 μs
= 1.2 – 3 ms 纯 CPU 启动时间
```

而 batch=1 时 GPU 实际计算可能只有 1ms。**CPU 成了瓶颈。**

### 解法：把整个 decode 前向录成一个图

```python
# runner/base_cuda_graph_runner.py:157
@staticmethod
def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
    index = bisect.bisect_left(buckets, raw_size)
    return buckets[index]
```

核心思想：

1. **预先捕获一组固定 batch size 的图**（buckets），例如
   `[1, 2, 4, 8, 16, 24, 32, 40, 48, ..., 160, 256]`
2. 运行时把真实 batch **pad 到最近的 bucket**（37 → 40，多 3 行是填充）
3. replay 就是一次 `cudaGraphLaunch`，**600+ 次 kernel 启动变成 1 次**

### bucket 怎么选

```python
# runner/base_cuda_graph_runner.py:67 get_batch_sizes_to_capture
capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)
num_max_requests = model_runner.req_to_token_pool.size
mul_base = get_cuda_graph_batch_size_alignment()
alignment_width = captured_req_width
# 模型输入的 token 数 = bs * alignment_width，必须能被 attn_tp_size 整除
capture_bs = [bs for bs in capture_bs if bs * alignment_width % mul_base == 0]
capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
capture_bs = list(sorted(set(capture_bs)))
```

**为什么要有整除过滤？** 模型输入张量有 `bs × tokens_per_req` 行，
attention-TP 切分要求这个数能被整除。**不满足的 bucket 捕获出来的图永远 replay 不了**——
这是一个必须提前过滤的坑。

### 填充的安全性

**填充行需要一个安全的地方写。** 回顾第 06 章：

```
req_to_token 的第 0 行是 padding 行
cuda-graph padded batches default req_pool_indices to 0,
so dummy reads/writes land here harmlessly.
```

**填充行全部指向第 0 行，读写落在垃圾区。** 这就是为什么 `ReqToTokenPool` 要
`size + 1` 行。

### 什么情况下不能走 CUDA Graph

`can_run_graph`（`decode_cuda_graph_runner.py:646`）会拒绝：

- `raw_size > max_bs`（超出最大捕获尺寸）
- `replace_embeds is not None`（动态的 per-request embedding）
- per-request 宽度不等于 `captured_req_width`（投机解码的 batch 形状不匹配）
- encoder-decoder 批里有 `encoder_len == 0`
- TBO 不支持的批
- ngram 批的 `batch_size * captured_req_width != input_ids.numel()`

### 相关参数

| 参数 | 说明 |
|---|---|
| `--cuda-graph-max-bs-decode` | decode 捕获的最大 batch size（默认按模型自动） |
| `--cuda-graph-bs-decode` | 显式指定 bucket 列表 |
| `--disable-cuda-graph-padding` | 关掉填充（要求 batch size 精确匹配） |
| `--disable-decode-cuda-graph` | 关掉 decode 图 |
| `--cuda-graph-backend-decode` | `full`（默认）/ `breakable` / `tc_piecewise` / `disabled` |

官方调参建议（`docs/docs/advanced_features/hyperparameter_tuning.mdx`）：

> 默认 CUDA Graph 只对小 batch（<160 或 256）启用。但在大 TP size 下，
> 512 或 768 的 batch 也可能受益。可以尝试提高 `--cuda-graph-max-bs-decode`。
> 注意 CUDA Graph 更耗显存，可能需要同时降低 `--mem-fraction-static`。

---

## 5.6 Attention 后端：prefill 和 decode 用不同的 kernel

`python/sglang/srt/layers/attention/` 下有 40+ 个后端文件。注册方式：

```python
# attention_registry.py
ATTENTION_BACKENDS = {}

def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn
    return decorator
```

### 选择逻辑

`model_runner_components/attention_backend_setup.py`：

```python
resolve_attention_backend_strs(...)   # 从配置读 (prefill, decode) 两个后端名
_build_resolved_backend(...)          # decode != prefill 时组合成 HybridAttnBackend
```

**关键点：prefill 和 decode 可以指定不同的后端**：

```bash
--attention-backend flashinfer              # 统一指定
--prefill-attention-backend fa3             # 单独指定 prefill（优先级更高）
--decode-attention-backend flashinfer
```

### 后端职责分工

| 后端 | 典型用途 | 说明 |
|---|---|---|
| `flashinfer` | **NVIDIA 默认**（MHA） | prefill 用 `BatchPrefillWithPagedKVCacheWrapper`，decode 用 `BatchDecodeWithPagedKVCacheWrapper` |
| `fa3` / `fa4` | prefill 主力 | FlashAttention-3/4，SM90/SM80 |
| `triton` | **跨平台** | 所有厂商都能跑，是 fallback |
| `flashmla` | MLA decode | `sgl_kernel.flash_mla`，`PAGE_SIZE=64` |
| `trtllm_mla` | MLA decode | TRT-LLM GenMLA kernel |
| `cutedsl_mla` | MLA | 复用 trtllm_mla 的数据准备 |
| `aiter` | **ROCm/AMD** | AITER kernel 全家桶 |
| `torch_native` | 正确性对照 | 纯 PyTorch SDPA，**不是性能路径** |
| `dsa` / `nsa` | DeepSeek 稀疏注意力 | `nsa` 是 `dsa` 的废弃别名 |
| `dsv4` | DeepSeek-V4 | `compressed` 是别名 |

完整的 choices 列表在 `arg_groups/choices.py:74`。

### `RadixAttention`：薄封装

```python
# layers/radix_attention.py:98
class RadixAttention(nn.Module):
    """不持有任何 KV 存储，只有几何信息 + 可选的 KV 量化方法。"""

    def forward(self, q, k, v, forward_batch, ...):
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(...)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(...)
        else:
            return self.forward_extend(...)
```

**它本身不做计算**，只是：
1. 规范化 shape
2. 按 `forward_mode` 分派
3. 处理 CUDA Graph 需要的 padding/idle 清零

### `init_forward_metadata` 为什么拆成三个

```python
# base_attn_backend.py:36
def init_forward_metadata(fb):                    # eager 入口 = 下面两个相加
def init_forward_metadata_out_graph(fb):          # 在 graph.capture() 之外跑
def init_forward_metadata_in_graph(fb):           # 在捕获区之内跑，会被 replay 自动重放
```

**这个拆分纯粹是为了 CUDA Graph 友好**：
- 主机侧 / 动态形状 / 不可录制的逻辑 → `out_graph`
- 静态形状的 GPU op → `in_graph`，可以被录制进图里

---

## 5.7 三个必修的融合 kernel

### ① RMSNorm + 残差：`fused_add_rmsnorm`

```python
# layers/layernorm.py
if post_residual_addition is not None:
    residual = residual + post_residual_addition
fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
result = x, residual
```

`residual += x; h = rmsnorm(x)` **原地完成**——1 个 kernel 代替 (add)+(读 hidden)+(reduce)+(写)+(scale 写)。
residual 通过原地修改返回，下一层复用同一个 buffer。

### ② SwiGLU：`silu_and_mul`

```python
# layers/activation.py:145
out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device)
silu_and_mul(x, out)
return out
```

朴素实现是 3 个 kernel（silu、split、mul）加一个 2N 宽的中间张量。
融合后：**读一次 2N 宽的 gate‖up，算 `silu(gate) * up`，写 N**。

### ③ 采样：融合的 top-k/top-p

```python
# layers/sampler.py:140
if sampling_info.is_all_greedy:
    batch_next_token_ids = torch.argmax(logits, -1)      # 全 greedy 直接 argmax
else:
    logits.div_(sampling_info.temperatures)
    logits[:] = torch.softmax(logits, dim=-1)   # 原地，省内存
    probs = logits
    batch_next_token_ids, _ = self._sample_from_probs(probs, sampling_info, ...)
```

`_sample_from_probs` 在一个 kernel 里做 top-k 过滤 + top-p 过滤 + 多项式采样：

```python
top_k_top_p_sampling_from_probs(probs.contiguous(), top_ks, top_ps,
                                filter_apply_order="joint")
```

**为什么必须融合？** batch 有 B 个序列，每个都要在 V=128k 的词表上做 top-k，
然后 top-p（排序/扫描），然后采样。用 torch 原语要 5–8 次 kernel 启动，
每次都是 `[B, 128000]` 的完整往返（`B × 128000 × 4 Byte`）。融合后一趟搞定。

### 一个被刻意跳过的同步

```python
# sampler.py _sync_token_ids_across_tp
if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
    # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
    # This saves one all-reduce, but the correctness ... depends on the determinism of
    # the last all-reduce, the last lm_head matmul, and all sampling kernels.
    torch.distributed.all_reduce(batch_next_token_ids, op=ReduceOp.MIN, group=self.tp_sync_group)
```

**默认不做 TP 间的 token 同步**——省一次 all-reduce，代价是要求那几步 kernel 是确定性的。

---

## 5.8 延迟采样：把 kernel 启动藏起来

`tp_worker.py:700` 有个很有意思的技巧：

```python
if (self.enable_overlap and not self.enable_spec
        and (forward_batch.sampling_info.grammars is not None
             or (envs.SGLANG_ENABLE_DELAY_SAMPLE.get() and not forward_batch.is_prefill_only))):
    def sample_batch_func():
        batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
        return batch_result
    batch_result.delay_sample_func = sample_batch_func
    return batch_result
```

**把 sampling 打包成闭包返回，让 Scheduler 在下一轮异步触发。**
这样 sampling kernel 的启动开销就藏进了 CPU 调度时间里。

---

## 5.9 本章自检

1. Scheduler 和 ModelRunner 的职责边界在哪？
2. 为什么 CUDA Graph 对 decode 是必需品？对 prefill 呢？
3. bucket 为什么要按 `attn_tp_size` 整除过滤？
4. CUDA Graph 填充出来的假请求写到哪？为什么安全？
5. `RadixAttention` 自己算 attention 吗？
6. 融合 `silu_and_mul` 相比朴素实现省了什么？

---

下一页 → [06 显存与内存池](06_memory_and_kv_pool.md)

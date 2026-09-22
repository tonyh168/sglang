# 进阶 05 · Overlap 调度与延迟隐藏

> 对应入门 [04 调度器](../beginner/04_scheduler.md) 的 overlap 一节。
> 源码：`managers/scheduler.py:1862-2060`、`managers/overlap_utils.py`（FutureMap）、
> `batch_overlap/two_batch_overlap.py`、`batch_overlap/single_batch_overlap.py`

---

## 1. 为什么必须有 overlap

一个 decode step 的真实时间预算：

```
GPU: [========= forward 15ms =========]
CPU:                                  [取批 3ms][采样 2ms][后处理 4ms] = 9ms
                                               ↑ 这 9ms GPU 完全空转
```

如果 CPU 侧每步要 9ms 而 GPU 只要 15ms，那么单步时间 = 24ms，
**你永远压不到 15ms 以下**。TPOT 被 CPU 决定了。

overlap 的目标：**让这 9ms 藏进上一轮/下一轮的 GPU 计算里**。

---

## 2. `event_loop_overlap`（`scheduler.py:1948`）

```python
@DynamicGradMode()
def event_loop_overlap(self):
    self.result_queue: Deque[Tuple[ScheduleBatch, Union[...]]] = deque()

    def pop_and_process():
        # Process the results of the last batch
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    while True:
        if self.gracefully_exit:
            break

        # Receive requests
        self.ingest_requests()
        if self._engine_paused:
            self._record_scheduler_state_for_paused_engine()
            continue

        # Get the next batch to run
        plan = self.get_next_batch_to_run(running_batch=self.running_batch,
                                          last_batch=self.last_batch)
        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(
            batch, last_batch=self.last_batch)

        # 需要同步时先结算上一轮
        if disable_overlap_for_batch:
            pop_and_process()
            if self.enable_unified_memory:
                try:
                    self.token_to_kv_pool_allocator.flush_opportunistic()
                except Exception:
                    pass

        # Launch the current batch
        if batch:
            batch_result = self.run_batch(batch)      # 异步发起 GPU forward
            self._apply_war_barrier()
            self.result_queue.append((batch.copy(), batch_result))
        else:
            batch_result = None
            self._sched_idled = True

        # ★ 关键：CPU 处理上一轮结果，GPU 在跑本轮
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
        elif batch is None:
            self.on_idle()

        # 采样依赖上一轮结果（如 grammar），所以放在 pop_and_process 之后
        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result, batch)

        self.last_batch = batch
```

### 被重叠的 CPU 工作清单

`pop_and_process` → `process_batch_result` 里的所有 CPU 活：

| 工作 | 代价 |
|---|---|
| 解析结果、更新 `Req.output_ids`、判定 finish | 小 |
| **radix tree `cache_finished_req` / `cache_unfinished_req`** | **大**（插入 + 引用计数 + 可能的驱逐） |
| 采样参数更新 | 小 |
| **grammar FSM 推进** | 中 |
| metrics / 负载快照 / trace flush | 中 |
| `tree_cache.flush_pending_backups()` | 中（HiCache 的 KV 落盘） |

`process_batch_result` 的注释：

> Flush async trace ops here: in overlap mode this CPU work runs while the
> next batch's GPU forward is in flight, giving free overlap.

### 三个 stream 的分工

```
schedule_stream    ← 调度器的 CPU 侧操作（分配 KV、组批）
forward_stream     ← 模型 forward
copy_stream        ← D2H 拷贝（logprob / hidden states）
```

`run_batch` 里的编排：

```python
with self.forward_stream_ctx:
    self.forward_stream.wait_stream(self.schedule_stream)     # forward 等 schedule 的写入
    ...
    # D2H 放在独立 copy_stream 上
    self.copy_stream.wait_stream(self.forward_stream)
    with self.copy_stream_ctx:
        batch_result.copy_to_cpu(return_logprob=..., return_hidden_states=...)
```

**D2H 放在独立 stream 上**，这样它不会挡住 forward_stream 上的下一步计算。

### WAR 屏障（`_apply_war_barrier`，:1899）

```python
def _apply_war_barrier(self):
    # WAR: keep later schedule_stream writes behind this forward's shared reads.
    # Clearing matters: a phase that skips the publish then falls back to coarse.
    if not self._war_barrier_enabled:
        return
    runner = self.model_worker.last_shared_read_runner
    ev = runner.shared_read_done_event
    runner.shared_read_done_event = None
    if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
        self.schedule_stream.wait_event(ev)          # 细粒度：只等 shared read
    else:
        self.schedule_stream.wait_stream(self.forward_stream)  # 粗粒度：等整个 forward
```

**WAR = Write-After-Read 冒险**：

```
本轮 forward 读了共享 buffer（input_ids、seq_lens 等）
下一轮 schedule_stream 要写同一个 buffer
→ 必须保证"读"在"写"之前完成
```

**细粒度 vs 粗粒度**：
- 细粒度：只等 `shared_read_done_event`（forward 中读共享 buffer 的那一小段）
- 粗粒度：等整个 forward 完成

模型可以选择在读完共享 buffer 后立刻记录一个 event，这样调度器不用等整个
forward 结束就能开始写下一轮的 buffer——**省下的时间就是 forward 的后半段**。

`scheduler.py:1891` 的启用条件：

```python
self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
```

---

## 3. `FutureMap`：跨迭代的设备端 relay

这是 overlap 实现里最难的部分。

### 问题的本质

```
第 N 轮：forward → 采样出 token t_N
第 N+1 轮：input_ids 需要 t_N
```

但第 N 轮的 forward 是异步发起的。**CPU 不能 sync 去读 `t_N`，那会把 overlap 全毁掉。**

### 解法：设备上的 pool-indexed relay buffer

```python
# overlap_utils.py:248
class FutureMap:
    def __init__(self, ...):
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]
        self.new_seq_lens_buf = torch.zeros(self.req_pool_size, dtype=..., device=...)
        ...
```

**按 `req_pool_idx` 索引**，不是一个 FIFO 队列。这样：

1. CUDA Graph 把 batch pad 到捕获尺寸时，填充行读 slot 0 也是安全的
2. 同一个请求跨轮复用同一个槽位，不需要维护映射关系

`overlap_utils.py` 里 slot 0 镜像 KV padding 行的理由和 `req_to_token` 完全一致。

### `publish` / `resolve` 配对

```python
def publish(self, future_indices, new_seq_lens, confidence=None):
    self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
    if self.spec_algo.is_some():
        self.publish_ready.record()          # 只有 spec_v2 需要 event
        self._publish_fresh = True
```

```python
def resolve_seq_lens_cpu(self, batch):
    fi = draft_input.future_indices
    if self.publish_ready is not None:
        self.publish_ready.wait()            # 只等这一个 event，不等整个 forward
    batch.seq_lens = self.new_seq_lens_buf[fi]
    ...
    # D2H 走私有 stream，不阻塞 schedule stream
    self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
    with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
        self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
    self.fwd_prepare_d2h_stream.synchronize()
    batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
```

**两个关键设计**：

1. **非 spec 路径完全不需要 event**。因为 `batch.seq_lens + 1` 是 CPU 侧已知的
   （decode 每步只涨 1），直接算出来写进 buffer 就行。
2. **spec 路径需要 `publish_ready`**，因为接受长度是 GPU 算出来的。
   但只等**这一个 event**，不等整个 forward。

### 懒初始化的其他 buffer

```python
# _maybe_init_forward_bufs:323
def _maybe_init_forward_bufs(...):
    # topk_p_buf / topk_index_buf / hidden_states_buf / draft_probs_buf
    # / dsa_topk_indices_buf / accept_tokens_buf / accept_lens_buf
```

**按需懒初始化**——prefill 批不带 spec extras，所以等 decode 第一次携带时才分配。
这省掉了不用投机解码时的显存。

### `record_batch_in_overlap`：防 GC 提前释放

```python
# scheduler.py:4251
def record_batch_in_overlap(self, batch):
    """for all non-future tensors ... we shall keep its reference not being
    released during all the forwarding pass"""
```

对 batch 的所有 dataclass 字段做快照。**因为跨 stream 的 GPU tensor 如果在
CPU 侧失去了引用，torch 的 caching allocator 可能提前回收它**。

配套 `self.batch_record_buf` 双缓冲（`init_overlap:1683`）。

### `batch.input_ids = None`

```python
# run_batch:4439
batch.input_ids = None      # 见 resolve_forward_inputs 的注释
```

本轮的 `input_ids` 是临时的，下一轮从 relay 取。

---

## 4. 什么时候必须禁用 overlap

```python
# scheduler.py:2022
def is_disable_overlap_for_batch(self, batch, last_batch) -> bool:
    # In DP attention mode, use the globally synchronized is_extend_in_batch
    # so all DP ranks make the same overlap decision (avoiding deadlock).
    if self.require_mlp_sync:
        is_extend = lambda b: b and b.is_extend_in_batch
    else:
        is_extend = lambda b: b and b.forward_mode.is_extend()

    batch_is_extend = is_extend(batch)
    last_batch_is_extend = is_extend(last_batch)

    disable_overlap_for_batch = (
        envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
        and batch_is_extend
        and last_batch_is_extend
    )

    # Sync so the FSM advance lands before the next batch's bitmask.
    need_grammar_sync = (
        batch
        and not batch.spec_algorithm.is_none()
        and batch.grammar_needs_sync()
        and batch.forward_mode.is_decode()
        and len(self.result_queue) > 0
    )
    return disable_overlap_for_batch or need_grammar_sync
```

### 两种禁用原因

**① 连续两个 prefill 批**

> For two consecutive prefill batches, we disable overlap to improve the TTFT
> of the first batch. This might slightly hurt the throughput, so we use an
> environment variable to control it.

overlap 会让 prefill 的启动延后一个 iter（因为要先结算上一轮）。
连续 prefill 时延迟结算 = TTFT 变差，所以禁用。

**② grammar（结构化输出）**

注释里的 "Sync so the FSM advance lands before the next batch's bitmask"——
grammar 的 FSM 状态依赖上一轮的采样结果，而下一轮的 attention mask 又依赖 FSM。
**这条依赖链必须同步**。

**注意 spec 的例外**：支持 `supports_grammar_overlap()` 的算法改为在 `verify()`
内部通过 grammar barrier 推进 FSM，这样它和 target forward 重叠，
所以不需要禁用：

> Algorithms that support grammar overlap advance the FSM inside verify() via the
> grammar barrier (overlapping the target forward), which resolves whatever result
> is still pending in the queue — including the extend->decode boundary — so no
> grammar-specific overlap disable is needed.

**DP 模式下的额外约束**：用 `batch.is_extend_in_batch`（全局同步过的）
而不是本地的 `forward_mode`，**保证所有 DP rank 做同样的 overlap 决策**，
否则不同 rank 的迭代节奏不一致 → 死锁。

---

## 5. TBO：批内的计算-通信重叠

overlap 调度解决的是"CPU vs GPU"。**TBO（Two Batch Overlap）解决的是
"compute vs communication"**。

`--enable-two-batch-overlap`（`arg_groups/fields/exec_.py:877`）。

### 原理

把一个 micro-batch 拆成两个子批 A 和 B，然后**把两个子批的 op 序列错位交替执行**：

```python
# batch_overlap/operations.py:44
def execute_overlapped_operations(inputs_arr, operations_arr, delta_stages):
    inputs_a, inputs_b = inputs_arr
    stages_a = _convert_operations_to_stages(operations_a)
    stages_b = _convert_operations_to_stages(operations_b)
    executor_a = _StageExecutor("a", stages_a, inputs=inputs_a, child_ctx=child_ctx_a)
    executor_b = _StageExecutor("b", stages_b, inputs=inputs_b, child_ctx=child_ctx_b)

    for _ in range(delta_stage):
        executor_a.next()                                   # A 先跑 delta 个 stage
    for _ in range(executor_a.num_stages - delta_stage):
        executor_a.next(); executor_b.next()                # 然后逐 stage 交替
    for _ in range(delta_stage):
        executor_b.next()
```

**这是 kernel 级的流水线重排**。

### op 序列怎么来

`OperationsStrategy`（`operations_strategy.py`）为每种 decoder layer 解析出 op 序列
（`DeepseekV2DecoderLayer` / `Qwen3MoeDecoderLayer` / `MiMoV2DecoderLayer` ...）。

**主要用在 MoE 上**：让 A 的 dispatch/combine all-to-all 与 B 的 expert GEMM 重叠。

```
时间轴（无 TBO）：
A: [attn][dispatch A2A][expert GEMM][combine A2A]
B:                                    [attn][dispatch A2A][expert GEMM][combine A2A]
      ↑ A 的通信期间 B 在等

时间轴（有 TBO，delta=1）：
A: [attn][dispatch A2A][expert GEMM][combine A2A]
B:        [attn]      [dispatch A2A]         [expert GEMM] ...
      ↑ B 的计算填进了 A 的通信空隙
```

### CUDA Graph 侧

```python
# two_batch_overlap.py:340 TboCudaGraphRunnerPlugin
# capture 时算 tbo_split_seq_index 并调 TboForwardBatchPreparer.prepare_raw
# replay 时 replay_prepare 重算 split
# 注释："when two_batch_overlap is enabled, we only capture CUDA Graph for tbo=true"
```

**开启 TBO 后只捕获 tbo=true 的图**——因为没有 tbo 的路径不会被走到。

### 对批大小对齐的连锁影响

```python
# utils/common.py:4053 get_cuda_graph_batch_size_alignment
alignment = 1
if get_exec().overlap.enable_two_batch_overlap:
    alignment *= 2                    # ★ TBO 需要偶数
if require_gathered_buffer():
    alignment *= get_parallel().attn_tp_size
```

**TBO 要求 batch 能被 2 整除**（要拆成两个子批），所以 bucket 对齐翻倍。

### 相关参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--enable-two-batch-overlap` | False | TBO |
| `--enable-single-batch-overlap` | False | SBO |
| `--tbo-token-distribution-threshold` | 0.48 | 按 token 分布决定走 two-batch 还是 two-chunk；设 0 关闭 two-chunk |

---

## 6. SBO：单批内的计算-通信重叠

`--enable-single-batch-overlap`（`exec_.py:881`），实现在
`batch_overlap/single_batch_overlap.py`。

**不拆批**，而是在单个 micro-batch 内让计算与通信叠，
例如 `SboFlags.enable_combine_down_gemm_two_stream_overlap`——用两条 stream
让 `combine` 的 all-to-all 与 `down_proj` 的 GEMM 重叠。

**仅 cutedsl / deep_gemm 后端支持**（需要 kernel 层面的配合）。

---

## 7. 延迟采样：把 kernel 启动藏起来

`tp_worker.py:700`：

```python
if (self.enable_overlap and not self.enable_spec
        and (forward_batch.sampling_info.grammars is not None
             or (envs.SGLANG_ENABLE_DELAY_SAMPLE.get()
                 and not forward_batch.is_prefill_only))):
    def sample_batch_func():
        batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
        return batch_result
    batch_result.delay_sample_func = sample_batch_func
    return batch_result
```

**把 sampling 打包成闭包返回，由 Scheduler 在下一轮异步触发。**

效果：sampling kernel 的启动开销藏进了 CPU 调度时间里。
`SGLANG_ENABLE_DELAY_SAMPLE` 控制是否默认开启。

注意条件 `not self.enable_spec`——投机解码的采样与 verify 深度耦合，不能延迟。

---

## 8. 用 profiler 验证 overlap 是否生效

```bash
curl http://localhost:30000/start_profile
# 跑请求
curl http://localhost:30000/stop_profile
```

在 trace 里看：

| 现象 | 含义 |
|---|---|
| GPU kernel 之间有空白 gap | CPU 侧太慢，overlap 没覆盖住 |
| gap 出现在 prefill 之后 | 连续 prefill 触发了 `is_disable_overlap_for_batch` |
| NCCL kernel 占比 > 30% | 通信瓶颈，考虑减 TP 或开 TBO |
| `copy_to_cpu` 与 compute 重叠 | D2H 的 copy_stream 生效了 |
| forward_stream 上有长等待 | 检查 `_apply_war_barrier` 是不是走了粗粒度路径 |

---

## 9. 自检

1. overlap 调度解决的是什么问题？如果不做，TPOT 的下限由谁决定？
2. `FutureMap` 为什么按 `req_pool_idx` 索引而不是用 FIFO 队列？
3. 非 spec 路径为什么不需要 `publish_ready` event？
4. `record_batch_in_overlap` 防的是什么？
5. 连续两个 prefill 为什么禁用 overlap？
6. TBO 和 overlap 调度的区别是什么？
7. TBO 为什么会改变 CUDA Graph 的 bucket 对齐要求？

---

下一页 → [06 ModelRunner 与 CUDA Graph](06_model_runner_and_cuda_graph.md)

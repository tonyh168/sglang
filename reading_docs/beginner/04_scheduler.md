# 04 · 调度器：连续批处理、分块预填充与调度策略

> 对应进阶 [04 调度器内部](../advanced/04_scheduler_internals.md)、
> [05 Overlap 与延迟隐藏](../advanced/05_overlap_and_latency_hiding.md)。
> 源码：`python/sglang/srt/managers/scheduler.py`（6126 行）、
> `python/sglang/srt/managers/schedule_policy.py`（1524 行）、
> `python/sglang/srt/managers/schedule_batch.py`（3894 行）

---

## 4.1 主循环只有 15 行

把 6000 行剥掉，调度器的心脏是 `event_loop_normal`（`scheduler.py:1913`）：

```python
while True:
    self.ingest_requests()                        # ① 收请求
    plan = self.get_next_batch_to_run(...)        # ② 决定这一轮跑什么
    batch = plan.batch_to_run
    if batch:
        result = self.run_batch(batch)            # ③ 发起 GPU forward
        self.process_batch_result(batch, result)  # ④ 处理结果
    else:
        self.on_idle()
    self.last_batch = batch
```

整个 SGLang 的调度逻辑都在这四步里。本章就按这四步讲。

---

## 4.2 ① 收请求：`ingest_requests`

从 ZMQ 收 `TokenizedGenerateReqInput`，用 `TypeBasedDispatcher` 一次查表分派
（`scheduler.py:1733`），生成类请求走 `handle_generate_request`（`:2750`）：

- 建 `Req` 对象
- **`tree_cache.match_prefix()` 查前缀命中**（第 03 章）
- 加入 `waiting_queue`

这一步没有 GPU 工作，是纯 CPU。

---

## 4.3 ② 组批：`get_next_batch_to_run`

这是调度器最复杂的方法（`scheduler.py:3622`，约 150 行）。按顺序做四件事：

### (a) 把上一轮的 prefill 批合并进 `running_batch`

```python
if last_batch and last_batch.forward_mode.is_extend():
    last_batch.filter_batch(chunked_req_to_exclude=...)   # 摘掉已完成的
    if not last_batch.is_empty():
        if running_batch.is_empty():
            running_batch = last_batch
        else:
            running_batch.merge_batch(last_batch)         # ← continuous batching 的实现
```

**这 5 行就是 "continuous batching" 的全部**：
- `filter_batch()` 把生成完的请求摘掉（它们立刻让出 KV）
- `merge_batch()` 把刚 prefill 完的请求拼进 decode 池

每轮循环都重做一次，所以批是"连续流动"的，不是静态的。

### (b) 尝试拿一个新 prefill 批

```python
prefill_plan = self.get_new_batch_prefill(running_batch)
new_batch = prefill_plan.batch_to_run
```

内部先排序（`policy.calc_priority`），再用 `PrefillAdder` 逐个试算预算。

### (c) **有 prefill 就优先跑 prefill**

```python
if new_batch is not None:
    ret = new_batch                                    # Run prefill first if possible
else:
    running_batch = self.update_running_batch(running_batch)
    ret = running_batch if not running_batch.is_empty() else None
```

**这是个关键的语义**：默认情况下 **prefill 批和 decode 批不混**。

为什么？因为 prefill 和 decode 的计算形状差异巨大（`[8192, 4096]` vs `[64, 4096]`），
混在一起 GEMM 效率会下降。

代价是什么？**一个 8K 的长 prefill 会让所有 decode 请求卡住**——TPOT 抖动。
这就引出了下面的 chunked prefill 和 mixed chunk。

### (d) DP attention 的同步

```python
ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)
```

DP attention 下每步要 all-gather 各 rank 的 token 数，保证所有 rank 的 batch 形状一致
（否则 collective 会挂）。见第 07 章。

---

## 4.4 Chunked Prefill：把长 prefill 切碎

### 问题

```
时间轴（无 chunked prefill，8K prompt）：
decode: ███ ███ ███ [————— 800ms 的 prefill —————] ███ ███
                    ↑ 所有 decode 请求卡了 800ms
```

### 解法：切块 + 与 decode 混批

状态只有一个变量：`self.chunked_req`——**同一时刻至多一个被切片的请求**。

```python
# schedule_policy.py:1022 add_chunked_req
cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
_rem_tokens = self.memory_budget.fit_chunk(extend_input_len=cand_extend_input_len, ...)
truncated = cand_extend_input_len > _rem_tokens
new_len = min(cand_extend_input_len, _rem_tokens)
req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
...
return req if truncated else None      # 返回 req = 还没切完，下轮继续
                                       # 返回 None = 这个请求做完了
```

每个 chunk 结束后，已经算出的 KV 用 `stash_chunked_request` 挂进 radix tree
（`scheduler.py:3516`），这样即使中断也能复用。

### 混批：`--enable-mixed-chunk`

```python
# scheduler.py:4104
if (self.is_mixed_chunk and not running_batch.is_empty() and ...):
    running_batch.prepare_for_decode()
    new_batch.mix_with_running(running_batch)      # ← 同一个 forward 里 extend + decode
    new_batch.decoding_reqs = running_batch.reqs
    running_batch = ScheduleBatch(reqs=[], ...)
```

`forward_mode` 变成 `MIXED`。收益是 TPOT 平滑；代价是 attention 要处理混合 mask，
且 `return_logprob` 路径不支持。

### PP 下的动态 chunk：`--enable-dynamic-chunking`

PP 下每个 micro-batch 必须等所有 stage 对齐。固定 chunk size 会让长 context 的
后段 chunk 越来越慢（attention 是 O(L²)），造成 pipeline bubble。

`dynamic_chunk_sizer.py` 的做法：

1. 启动时跑 128 个合成 prefill，同步计时，得到 `(seq_len, latency_ms)` 样本
2. 各 rank 用 `np.linalg.lstsq` 拟合同一个二次模型 `f(l) = a·l² + b·l + c`
3. 运行时解 `f(L+x) − f(L) = T`，取正根得到下一个 chunk 大小

**目标：让每块的执行时间恒定，而不是 token 数恒定。**

---

## 4.5 调度策略：先排谁？

`--schedule-policy` 的取值（`arg_groups/fields/schedule.py:89`）：

```python
choices=["lpm", "random", "fcfs", "dfs-weight", "lof",
         "priority", "routing-key", "hrrn", "shortest-prefill-first"]
# 默认是 "fcfs"
```

| 策略 | 排序依据 | 什么时候用 |
|---|---|---|
| `fcfs`（默认） | 先到先服务 | 通用，开销最低 |
| `lpm` | **最长前缀命中优先** | 大量共享前缀（few-shot / multi-turn / RAG） |
| `dfs-weight` | radix 树的 DFS 顺序 | 同上，但用树结构做局部性排序 |
| `hrrn` | 已有 token 数 / 未缓存 token 数 | 防短请求被饿死 |
| `shortest-prefill-first` | 未缓存 token 数升序 | 最小化平均 TTFT |
| `lof` | `max_new_tokens` 降序 | 长输出先跑，减少尾延迟 |
| `routing-key` | 同 key 的请求聚在一起 | 网关路由下让同会话聚实例 |
| `random` | 随机 | 基准对比 |
| `priority` | 显式优先级 | 需要 `--enable-priority-scheduling` |

### 一个重要的自适应降级

```python
# schedule_policy.py:330
if self.policy in (CacheAwarePolicy.LPM, CacheAwarePolicy.HRRN) and len(waiting_queue) > 128:
    # Turn off the expensive prefix matching and sorting when the #queue is large.
    return CacheAgnosticPolicy.FCFS
```

**队列超过 128 时，LPM/HRRN 自动退化成 FCFS。** 因为前缀匹配是
O(队列长度 × 前缀长度) 的纯 CPU 开销，队列一大，排序收益就被 CPU 成本吃掉了。

> 这是 SGLang 里很典型的风格：**每个优化都配一个"什么时候该关掉它"的守卫**。

### batch 内前缀去重：最巧妙的一段

```python
# schedule_policy.py:372 _compute_prefix_matches
self.waiting_queue_radix_tree.reset()
for r in waiting_queue:
    match_result = self.waiting_queue_radix_tree.match_prefix(...)
    if len(match_result.device_indices) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:  # 32
        temporary_deprioritized.add(r.rid)     # 队列里已有人共享这段前缀 → 让别人先跑
    else:
        self.waiting_queue_radix_tree.insert(...)  # 只插入没命中的
```

`waiting_queue_radix_tree` 是一棵 **只有 token、没有 KV 的影子树**。
它的作用是判断"本批内请求之间是否共享前缀"。

如果队列里 A 和 B 共享 500 token 前缀，那就**只调度 A**，让 B 等下一轮——
那时 B 就能直接命中 A 刚写进 cache 的 KV，而不是两个都从头算一遍。

**这是"批内命中率"优化，收益常常被忽略。**

### `--schedule-conservativeness`

默认 1.0。它直接缩放 `new_token_ratio` 的初值（见 4.7）。
**调大 = 更保守 = 少 retract**。日志里频繁看到 retract 就调到 1.3。

---

## 4.6 `PrefillAdder`：预算器

拿到排序后的队列，`PrefillAdder` 逐个试算"这个请求塞进来会不会超预算"
（`schedule_policy.py:1232 add_one_req`）。

三个独立预算：

```python
# schedule_policy.py:848 _update_prefill_budget
self.memory_budget.reserve(extend_input_len, max_new_tokens, ...)
self.rem_input_tokens -= compute_charge      # 受 --max-prefill-tokens 限制（默认 16384）
self.rem_chunk_tokens -= compute_charge      # 受 --chunked-prefill-size 限制
# rem_total_tokens 由 memory_budget 管（KV 总量）
```

> 源码注释点明了一个容易踩的坑：**compute budget 和 KV budget 必须用同一个计数口径**，
> 否则四舍五入的余量会被其中一个先耗尽，导致奇怪的行为。

### 并发上限

```python
# scheduler.py:3770 get_num_allocatable_reqs
pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
available = self.req_to_token_pool.available_size()
res = min(pp_budget, available)
```

**并发上限 = min(PP 微批上限, req 槽位空闲数)**。
不是 KV 容量！KV 容量在 `PrefillAdder` 的预算里单独检查。

---

## 4.7 Retraction：KV 满了怎么办

### 触发

```python
# scheduler.py:4179 update_running_batch
if not batch.check_decode_mem():
    ... retract_decode() ...
```

`check_decode_mem` 问的是："下一次 decode 需要的 KV 还放得下吗？"

### 撤回谁

```python
# schedule_batch.py:3318 _get_decode_retraction_order
def length_key(req):
    return (len(req.output_ids), -len(req.origin_input_ids))
# 排序后从尾部弹 → 优先撤回「已生成少 + 输入长」的请求
```

**为什么是这个顺序？** 沉没成本：

- 已经生成很多的请求快结束了，撤回它等于浪费所有已算的 KV
- 长输入短输出的请求，重新 prefill 时还能大量命中 radix cache

循环直到 `check_decode_mem` 通过——**所以一次可能撤回多个**，不是固定的一个。
但至少保留一个请求（`if len(sorted_indices) == 1: break`）。

`--retraction-policy priority` 时改为按优先级撤回。

### `new_token_ratio` 的自适应

这是 SGLang 一个很聪明的设计。核心问题：**用户声明的 `max_new_tokens` 通常远大于实际输出。**

如果调度器按声明预分配 KV，并发会被压得极低。
如果不预留，decode 中途就会 OOM。

解法：维护一个自适应的比例 `new_token_ratio`：

```python
# scheduler_components/new_token_ratio_tracker.py
init = min(SGLANG_INIT_NEW_TOKEN_RATIO * schedule_conservativeness, 1.0)
min_ratio = min(init * SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)
decay = (init - min_ratio) / SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS
```

- **无压力时**：每轮 `decay_step()` 缓慢下降 → 越来越敢接纳请求
  （因为实测证明请求不会跑满 `max_new_tokens`）
- **发生 retract 时**：立刻用实测值重估

```python
new_estimate_ratio = (total_decoded_tokens + SGLANG_RETRACT_DECODE_STEPS * len(reqs)) \
                     / (total_max_new_tokens + 1)
```

分子是"已生成 + 还预计再生成几步"，分母是声明的总预算 ——
**用真实的长尾分布替换用户的悲观声明**。

日志里会看到：

```
KV cache pool is full. Retract requests. #retracted_reqs: 1,
  #new_token_ratio: 0.9998 -> 1.0000
```

偶尔一次（约每分钟一次）是健康的。频繁出现就调大 `--schedule-conservativeness`。

---

## 4.8 Overlap 调度：让 CPU 藏进 GPU 里

### 问题

一个 decode step：

```
GPU: [===== forward 15ms =====]
CPU:                           [调度 3ms][采样 2ms][后处理 4ms]
                                        ↑ 这 9ms GPU 在空转
```

如果 CPU 侧每步要 9ms 而 GPU 只要 15ms，**你永远压不到 15ms 以下**。

### 解法：`event_loop_overlap`（`scheduler.py:1948`）

```python
def pop_and_process():
    tmp_batch, tmp_result = self.result_queue.popleft()
    self.process_batch_result(tmp_batch, tmp_result)   # 处理上一轮结果

while True:
    ...
    if batch:
        batch_result = self.run_batch(batch)           # 异步发起 GPU forward
        self.result_queue.append((batch.copy(), batch_result))

    # ★ 关键：CPU 处理上一轮结果，GPU 在跑本轮
    if self.last_batch and not disable_overlap_for_batch:
        pop_and_process()
```

**被重叠的 CPU 工作**：
- 解析结果、更新 `Req.output_ids`、判定 finish
- radix tree 插入 + 引用计数（这一项很重）
- 采样参数更新、grammar FSM 推进
- metrics / 负载快照 / trace flush

源码注释说得很直白：

```
# in overlap mode this CPU work runs while the next batch's GPU forward is in flight,
# giving free overlap
```

### 难点：下一轮的输入依赖本轮的结果

下一轮 decode 的 `input_ids` 是**本轮采样出来的 token**。但本轮 GPU 还没跑完，
CPU 不能 sync 去读。

解法是 `FutureMap`（`managers/overlap_utils.py:248`）——**设备上的 relay buffer**：

```python
# publish：forward 结束后把新 seq_lens 写进设备 buffer
def publish(self, future_indices, new_seq_lens, confidence=None):
    self.new_seq_lens_buf[indices] = new_seq_lens.to(...)

# resolve：下一轮调度时从 buffer 取回，按 req_pool_idx 索引
def resolve_seq_lens_cpu(self, batch):
    fi = draft_input.future_indices
    if self.publish_ready is not None:
        self.publish_ready.wait()          # 只等这一个 event，不等整个 forward
    batch.seq_lens = self.new_seq_lens_buf[fi]
    ...
```

buffer 按 `req_pool_idx` 索引，而且 **slot 0 镜像 KV 的 padding 行**——
这样 CUDA Graph 把 batch pad 到捕获尺寸时，填充行读 slot 0 也是安全的。

> 这是 SGLang overlap 实现的精髓：**用设备内存做跨迭代通信，避免任何 sync**。

### 什么时候必须同步

```python
# scheduler.py:2022 is_disable_overlap_for_batch
disable_overlap_for_batch = (envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
                             and batch_is_extend and last_batch_is_extend)
need_grammar_sync = (... grammar_needs_sync() and ...)
```

两个例外：

1. **连续两个 prefill 批**：禁用 overlap 以改善第一个批的 TTFT
2. **grammar（结构化输出）**：状态机依赖上一轮结果，必须同步推进

`--disable-overlap-schedule` 可以完全关掉（默认开启）。

---

## 4.9 两个"延迟"机制

### PrefillDelayer（`managers/prefill_delayer.py`）

**场景**：DP attention 下，某 rank 有 prefill 可做、别的 rank 只有 decode。
如果立刻放行，少数 rank 做长 prefill，其余 rank 空等（因为每步都要等 collective）。

**做法**：让有 prefill 的 rank **主动等几轮**，直到所有 rank 都能一起 prefill，
把碎片化的 prefill 合并成大批。

```python
tp0_info = self._gather_info(local_prefillable=..., ...)
global_prefillable = tp0_info[:, 0]
if global_prefillable.min().item() > 0:      prefillable_status = "all"
elif global_prefillable.max().item() == 0:   prefillable_status = "none"
else:                                        prefillable_status = "mixed"
```

**决策必须全局一致**（否则 rank 间 batch 形状不一致会 collective hang），
所以每轮做一次 all-gather。

三道防饿死阀门：`--prefill-delayer-token-usage-low-watermark`（KV 空闲就立刻放行）、
`--prefill-delayer-queue-min-ratio`、`--prefill-delayer-max-delay-ms`（默认 5000ms 墙钟上限）。

### MinFreeSlotsDelayer（`managers/min_free_slots_delayer.py`）

```python
def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
    return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
```

区别：PrefillDelayer 等的是"**别的 rank**"，MinFreeSlotsDelayer 等的是"**本 rank 的 req 槽位**"。
目的是把零散的一次一个 prefill 攒成一次批量 prefill。**本地决策，不需要通信**。

---

## 4.10 本章自检

1. `merge_batch` 那一行实现了什么？为什么说它就是 continuous batching？
2. `chunked_req` 为什么只能有一个？
3. 为什么队列 >128 时 LPM 要退化成 FCFS？
4. retraction 为什么优先撤回"已生成少 + 输入长"的请求？
5. `new_token_ratio` 解决的是什么问题？
6. Overlap 调度的核心技术难点是什么？`FutureMap` 怎么解决的？

---

下一页 → [05 ModelRunner 与一次前向](05_model_runner.md)

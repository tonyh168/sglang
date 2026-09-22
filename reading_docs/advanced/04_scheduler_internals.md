# 进阶 04 · 调度器内部实现

> 对应入门 [04 调度器](../beginner/04_scheduler.md)。
> 源码：`managers/scheduler.py`（6126 行）、`managers/schedule_policy.py`（1524 行）、
> `managers/schedule_batch.py`（3894 行）、`managers/scheduler_components/`

---

## 1. `Scheduler.__init__` 的编排

`Scheduler.__init__`（`scheduler.py:447`）是 `large-class-style` skill 管辖的对象，
风格是"每个 `init_*` 一件事"。关键顺序：

```python
self.init_model_config()
self.init_metrics_collector(...)
self.init_ipc_channels(port_args)              # IPC socket（只有 rank0 有）
self.init_idle_sleeper()
self.init_tokenizer()                          # 需要 tokenizer 算 max_req_input_len
self.init_tp_model_worker()                    # ← 建 ModelRunner
self.maybe_init_draft_worker()                 # 投机解码的 draft worker
self.init_memory_pools()                       # KV 池 + allocator
self.init_all_attention_backends()
self.init_all_cuda_graphs()
self.init_model_worker()
self.init_running_status()                     # running_batch / waiting_queue
self.init_chunked_prefill()                    # chunked_req = None
self.maybe_init_dynamic_chunk_sizer()
self.init_schedule_policy()
self.init_overlap()                            # FutureMap / result_queue
...
self.init_request_dispatcher()                 # TypeBasedDispatcher
```

### `init_schedule_policy`（:1385）

```python
self.policy = SchedulePolicy(
    self.tree_cache, self.schedule_policy, self.page_size, ...
)
```

`SchedulePolicy` **持有 tree_cache 的引用**——因为 `_compute_prefix_matches`
需要用它来做"批内前缀去重"。

### `init_overlap`（:1630）

```python
self.future_map = FutureMap(...)
```

`FutureMap` 的 buffer 大小是 `req_to_token_pool.req_to_token.shape[0]`
（= `max_running_requests + 1`）。见第 05 章。

### 初始状态

```python
self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
self.waiting_queue: List[Req] = []
self.chunked_req: Optional[Req] = None          # ★ 至多一个
self.last_batch: Optional[ScheduleBatch] = None
self.chunked_req_to_exclude: set = set()
```

---

## 2. `get_next_batch_to_run` 逐段拆解

这是调度器最复杂的方法（`scheduler.py:3622`，约 150 行）。按执行顺序：

### 段 1：清理（:3624-3640）

```python
self.process_pending_chunked_abort()      # 处理 chunked 请求的 abort
self._process_hicache_events()            # HiCache 的异步事件
if self.dllm_config is not None:
    self.dllm_manager.filter_finished_reqs()
```

### 段 2：处理上一轮的 chunked_req（:3643-3678）

```python
chunked_req_to_exclude = set()

if self.chunked_req is not None:
    # Move the chunked request out of the batch so that we can merge
    # only finished requests to running_batch.
    chunked_req_to_exclude.add(self.chunked_req)

    # Stash (cache) the previous chunk only when it produced new KV
    # beyond what is already cached. A parked chunk (add_chunked_req
    # hybrid-SWA early-return) leaves extend_range.end == len(prefix_indices),
    # so there is nothing new to cache and stashing would be a no-op.
    if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
        self.stash_chunked_request(self.chunked_req)
```

**两个精确的细节**：

1. `chunked_req` 必须从 `last_batch` 里排除，否则它会被当成"已完成"合并进
   `running_batch`，然后下一轮又要做 chunk——状态就乱了。
2. `stash_chunked_request` 有条件：**只有真的产生了新 KV 才 stash**。
   混合 SWA 模型的一个早返回路径会让 `extend_range.end == len(prefix_indices)`，
   此时 stash 是空操作。

### 段 3：合并上一轮的 prefill 批（:3679-3712）

```python
if (not self.enable_hisparse
        and last_batch
        and last_batch.forward_mode.is_extend()):
    if last_batch.chunked_req is not None:
        # In the context pipeline parallelism, after the last chunk, the current
        # microbatch still track outdated chunked_req. We need to discard it.
        chunked_req_to_exclude.add(last_batch.chunked_req)

    last_bs = last_batch.batch_size()
    last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
    if last_batch.batch_size() < last_bs:
        running_batch.batch_is_full = False        # ← 有请求退出，解除"满"标记

    if not last_batch.is_empty():
        if running_batch.is_empty():
            running_batch = last_batch
        else:
            running_batch.merge_batch(last_batch)
```

**`running_batch.batch_is_full = False`** 这一行很关键：只要有请求退出，
就解除"批已满"标记，让下一轮能接纳新的 prefill。

### 段 4：prefill 或 decode（:3739-3766）

```python
if self.dllm_config is not None:
    new_batch = self.get_new_batch_dllm(running_batch)
elif self._should_defer_prefill():          # --prefill-decode-interval
    new_batch = None
else:
    prefill_plan = self.get_new_batch_prefill(running_batch)
    new_batch = prefill_plan.batch_to_run
    running_batch = prefill_plan.running_batch

need_mlp_sync = self.require_mlp_sync
if (need_mlp_sync and not self.spec_algorithm.is_none()
        and not get_spec().speculative_skip_dp_mlp_sync):
    new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
    need_mlp_sync = new_batch is None

if new_batch is not None:
    ret = new_batch                          # Run prefill first if possible
else:
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        running_batch = self.update_running_batch(running_batch)
        ret = running_batch if not running_batch.is_empty() else None
    else:
        ret = None
```

### 段 5：DP 与收尾（:3762-3769）

```python
ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)
converted = self.dp_attn_adapter.maybe_convert_decode_to_extend(ret)
ret = converted
self._arm_prefill_decode_interval(ret)
ret = self.ngram_embedding_manager.prepare_for_forward(ret, chunked_req=self.chunked_req)
if ret:
    set_schedule_time_batch(ret)
return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)
```

---

## 3. `_get_new_batch_prefill_raw`：预算与准入

`get_new_batch_prefill`（:3790）是薄封装，实际逻辑在 `_get_new_batch_prefill_raw`
（:3817-4137）。骨架：

```python
# 1. 快速退出
if ((running_batch.batch_is_full or len(self.waiting_queue) == 0)
        and self.chunked_req is None):
    return None, running_batch

# 2. req 槽位检查
if self.get_num_allocatable_reqs(running_bs, ...) <= 0 and self.chunked_req is None ...:
    running_batch.batch_is_full = True
    return None, running_batch

# 3. 排序（见 §4）
self.policy.calc_priority(self.waiting_queue, running_batch,
                          processed_tokens=self.processed_tokens_counter)

# 4. 建预算器
adder = PrefillAdder(self.page_size, self.tree_cache, self.token_to_kv_pool_allocator,
                     running_batch, self.new_token_ratio, self.max_prefill_tokens,
                     self.chunked_prefill_size, ...)

# 5. 先续跑上一轮的 chunk
if self.chunked_req is not None:
    self.chunked_req = adder.add_chunked_req(self.chunked_req)

# 6. 逐个加请求
for req in self.waiting_queue:
    ...
    res = adder.add_one_req(req, has_chunked_req=..., truncation_align_size=...)
    if res != AddReqResult.CONTINUE:
        break

# 7. 组批
new_batch = ScheduleBatch.init_new(can_run_list, ...)
new_batch.prepare_for_extend()
```

### `get_num_allocatable_reqs`（:3770）

```python
def get_num_allocatable_reqs(self, running_bs, beam_width=None, running_batch=None):
    pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
    available = self.req_to_token_pool.available_size()
    active_batch = running_batch or self.running_batch
    available = max(available - self.beam_coordinator.pending_member_rows(active_batch), 0)
    res = min(pp_budget, available)
    if beam_width is not None:
        res = min(res, available // beam_width)
    return res
```

**并发上限 = min(PP 微批上限, req 槽位空闲数)**。
注意这里**没有 KV 容量的检查**——KV 在 `PrefillAdder` 里单独算。

---

## 4. 排序：`SchedulePolicy.calc_priority`

```python
# schedule_policy.py:259
def calc_priority(self, waiting_queue, running_batch, processed_tokens=None):
    policy = self._determine_active_policy(waiting_queue)
    if self.policy == CacheAgnosticPolicy.FCFS:
        if self.enable_priority_scheduling:
            SchedulePolicy._sort_by_priority_and_fcfs(waiting_queue, self.priority_sign)
        return
    if isinstance(policy, CacheAwarePolicy):
        temporary_deprioritized = self._compute_prefix_matches(waiting_queue, policy)
        if policy == CacheAwarePolicy.LPM:
            SchedulePolicy._sort_by_longest_prefix(waiting_queue, temporary_deprioritized)
        elif policy == CacheAwarePolicy.DFS_WEIGHT:
            SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
        elif policy == CacheAwarePolicy.HRRN:
            SchedulePolicy._sort_by_hrrn(waiting_queue, processed_tokens, ...)
        elif policy == CacheAwarePolicy.SHORTEST_PREFILL_FIRST:
            SchedulePolicy._sort_by_shortest_prefill(waiting_queue)
```

### 自适应降级（:330）

```python
def _determine_active_policy(self, waiting_queue):
    if self.policy in (CacheAwarePolicy.LPM, CacheAwarePolicy.HRRN) and len(waiting_queue) > 128:
        # Turn off the expensive prefix matching and sorting when the #queue is large.
        return CacheAgnosticPolicy.FCFS
    return self.policy
```

**队列 >128 时 LPM/HRRN 退化为 FCFS。** 前缀匹配是 O(queue × prefix_len)
的纯 CPU 开销，队列一大，排序收益被 CPU 成本吃掉。

### HRRN：token 粒度的老化

经典 HRRN 是 `1 + wait_time / service_time`。SGLang 把**时间换成已处理的 prefill token 数**：

```python
# :495
@staticmethod
def _sort_by_hrrn(waiting_queue, processed_tokens, ...):
    def key(r):
        waited_tokens = max(0, processed_tokens - r.arrival_processed_tokens)
        uncached = r.origin_input_len - r.num_matched_prefix_tokens
        ratio_delta = waited_tokens / max(uncached, 1)
        return (-ratio_delta, r.rid)
    waiting_queue.sort(key=key)
```

`processed_tokens_counter` 在 `run_batch:4331` 累加 `batch.extend_num_tokens`。

**为什么用 token 数而不是墙钟时间？**
在 GPU 上"时间"不是均匀流逝的——batch 大时一步慢，batch 小时一步快。
用 token 数衡量"服务量"更贴近真实的资源消耗。

注意注释里的退化说明：调用方不传 `processed_tokens` 时（dllm、disagg decode）
会退化成 rid 的字典序。

### shortest-prefill-first：带反饥饿

```python
# :439
def _sort_by_shortest_prefill(waiting_queue):
    waiting_queue.sort(key=lambda r: (SchedulePolicy._shortest_prefill_work(r), r.rid))

@staticmethod
def _shortest_prefill_work(r: Req) -> int:
    return max(1, len(r.full_untruncated_fill_ids) - len(r.prefix_indices))
```

**优先调度未缓存 token 最少的请求**——最小化平均 TTFT。

配套的反饥饿机制 `shortest_prefill_chunk_limit`（:451）：

```python
remaining = len(chunked_req.full_untruncated_fill_ids) - len(chunked_req.prefix_indices)
reserved = 0
for req in waiting_queue:
    work = self._shortest_prefill_work(req)
    charge = _ceil_div(work, page_size) * page_size
    if work >= remaining or reserved + charge > budget - page_size:
        break
    reserved += charge
return (budget - reserved) // page_size * page_size
```

**给正在跑的 chunk 设上限，为队列里更短的请求预留 token。**
否则一个超长 chunk 会把后面所有短请求全饿死。

### `_compute_prefix_matches`：批内去重

```python
# :372
def _compute_prefix_matches(self, waiting_queue, policy):
    if get_server_args().disable_radix_cache:
        return set()
    self.waiting_queue_radix_tree.reset()
    temporary_deprioritized = set()
    for r in waiting_queue:
        prefix_ids = r.origin_input_ids + r.output_ids
        match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids, include_req=True)
        # NOTE: the prefix_indices must always be aligned with page_size
        r.prefix_indices = match_result.device_indices
        r.extend_input_len = len(prefix_ids) - len(r.prefix_indices)
        if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:   # 32
            match_result = self.waiting_queue_radix_tree.match_prefix(...)
            if len(match_result.device_indices) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:  # 32
                temporary_deprioritized.add(r.rid)
            else:
                # Not necessary to insert the filled prefix and prefix_indices
                self.waiting_queue_radix_tree.insert(...)
    return temporary_deprioritized
```

`waiting_queue_radix_tree` 是 `RadixCache.create_simulated()`（:257）——
**一棵只有 token、没有 KV 的影子树**。

逻辑：
1. 对队列里每个请求查真实 cache 的命中
2. 如果命中很少（≤32），再查影子树：**队列里已经有人共享这段前缀？**
3. 是 → 标记为 `temporary_deprioritized`，这一轮不调度它
4. 否 → 插进影子树，供后面的请求查

**效果**：A 和 B 共享 500 token 前缀时只调度 A，B 等下一轮直接命中 A 刚写进 cache 的 KV。
两个请求都省了 500 token 的 prefill。

常量（均可用环境变量覆盖）：

```
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD        = 32
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = 32
SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION          = 4096
```

---

## 5. `PrefillAdder`：三个独立预算

```python
# schedule_policy.py:848 _update_prefill_budget
extend_input_len = self.ceil_paged_tokens(extend_input_len)     # 按 page 向上取整
self.memory_budget.reserve(extend_input_len, max_new_tokens,
                           extra_tokens=mamba_gap_reserve,
                           chunk_limit=self.rem_chunk_tokens, ...)
self.rem_input_tokens -= compute_charge
if self.dllm_config is not None:
    self.rem_dllm_tokens -= compute_charge
elif self.rem_chunk_tokens is not None:
    self.rem_chunk_tokens -= compute_charge
```

| 预算 | 来源 | 含义 |
|---|---|---|
| `memory_budget` | KV 池容量 | 能否放下这个请求的 prefill + 预估的生成 |
| `rem_input_tokens` | `--max-prefill-tokens`（默认 16384） | 本批 prefill 的 token 上限 |
| `rem_chunk_tokens` | `--chunked-prefill-size` | 单个 chunk 的上限 |

源码注释里的警告：

> **compute budget 和 KV budget 必须用同一个计数口径**，否则四舍五入的余量
> 会被其中一个先耗尽。

`ceil_paged_tokens` 就是"按 page 向上取整"——分页场景下 1 个 token 也要占一整页。

### `_check_prefill_budget`（:792）

```python
def _check_prefill_budget(self, req, extend_input_len, max_new_tokens):
    if self.rem_total_tokens < ... :
        return False
    ...
```

---

## 6. Retraction 的完整逻辑

### `check_decode_mem`（`schedule_batch.py:3229`）

```python
def check_decode_mem(self, selected_indices=None):
    num_tokens = self.new_tokens_required_next_decode(selected_indices)
    return self.token_to_kv_pool_allocator.check_decode_capacity(
        num_tokens=num_tokens, tree_cache=self.tree_cache,
        requests=requests, spec_algorithm=self.spec_algorithm)
```

**职责划分很清晰**：allocator 负责容量判定（含 eviction），batch 只负责问。

### `retract_decode`（`schedule_batch.py:3248`）

```python
sorted_indices = self._get_decode_retraction_order(self.reqs)     # 最想保留的在前
while first_iter or (not self.check_decode_mem(selected_indices=sorted_indices)):
    if len(sorted_indices) == 1:
        break                                    # Always keep at least one request
    idx = sorted_indices.pop()                   # 从尾部弹 = 最不想要的
    req = self.reqs[idx]
    if req.beam_group is not None:
        req.to_finish = FINISH_ABORT("Beam search group aborted: ...")
        self.release_req(idx, len(sorted_indices), offload_kv=False)
        continue
    if self.release_req(idx, len(sorted_indices)):
        retracted_reqs.append(req)               # 成功 → 回 waiting_queue
    else:
        req.to_finish = FINISH_ABORT("Retraction host KV pool exhausted. ...")
self.filter_batch(keep_indices=sorted_indices)
new_estimate_ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(self.reqs)
return retracted_reqs, new_estimate_ratio, reqs_to_abort
```

**注意是 `while` 循环不是 `if`**——一次可能 retract 多个，直到 `check_decode_mem` 通过。

### 撤回顺序（:3318）

```python
@staticmethod
def _get_decode_retraction_order(reqs):
    def length_key(req):
        return (len(req.output_ids), -len(req.origin_input_ids))
    return [i for i, _ in sorted(enumerate(reqs), key=lambda x: length_key(x[1]), reverse=True)]
```

`reverse=True` 后从尾部弹 → **优先撤回「已生成 token 少 + 输入长」的请求**。

**理由（沉没成本）**：
- 已生成很多的请求快结束了，撤回 = 浪费全部已算的 KV
- 长输入短输出的请求，重新 prefill 时能大量命中 radix cache（输入前缀已缓存）

`--retraction-policy priority` 时改为按优先级撤回。

### `new_token_ratio` 的自适应

```python
# scheduler_components/new_token_ratio_tracker.py
init = min(SGLANG_INIT_NEW_TOKEN_RATIO * schedule_conservativeness, 1.0)
min_ratio = min(init * SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)
decay = (init - min_ratio) / SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS
```

- **无压力**：每轮 `decay_step()` 缓慢下降 → 越来越敢接纳请求
- **retract 后**：

```python
@staticmethod
def estimate_new_token_ratio_after_retract(reqs):
    total_decoded_tokens = sum(len(r.output_ids) for r in reqs)
    total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in reqs)
    return (total_decoded_tokens + SGLANG_RETRACT_DECODE_STEPS * len(reqs)) \
           / (total_max_new_tokens + 1)
```

**这是全系统最有工程智慧的一处**：
用户声明的 `max_new_tokens` 通常远大于实际输出。按声明预分配 → 并发被压死；
不预留 → decode 中途 OOM。SGLang 的答案是"乐观接纳 + 满时回滚 + 用实测替换声明"。

---

## 7. 抢占（`preempt_to_schedule`，:1454）

只在 `--enable-priority-scheduling` 且
`priority_diff > --priority-scheduling-preemption-threshold`（默认 10）时触发：

```python
min_tokens_to_remove = (len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
                        + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                        - self.rem_total_tokens)
for running_req in sorted_valid_running_reqs:
    priority_diff = (req.priority - running_req.priority) * (-priority_sign)
    if priority_diff > self.priority_scheduling_preemption_threshold:
        preemptible_reqs.append(running_req)
        min_tokens_to_remove -= self._get_running_request_total_token_offset(running_req)
        if min_tokens_to_remove <= 0:
            break
    else:
        break                          # 优先级单调，可以 early-exit
```

`break` 而不是 `continue` 是**利用了排序的单调性做 early-exit**。

被抢占的请求进 `preempt_list` → 回 waiting_queue（**不是 abort**），
KV 通过 `release_req` 尽力写回 host/tree。

---

## 8. 延迟机制

### `PrefillDelayer`（`managers/prefill_delayer.py`）

**场景**：DP attention 下，部分 rank 有 prefill、部分只有 decode。
立即放行会让少数 rank 做长 prefill，其余 rank 空等（每步都要等 collective）。

```python
# _negotiate_should_allow_prefill_pure:170
tp0_info = self._gather_info(local_prefillable=..., local_token_watermark_force_allow=...,
                             running_batch=..., max_prefill_bs=..., waiting_queue_len=...)
global_prefillable = tp0_info[:, 0]
if global_prefillable.min().item() > 0:      prefillable_status = "all"
elif global_prefillable.max().item() == 0:   prefillable_status = "none"
else:                                        prefillable_status = "mixed"
```

**决策必须全局一致**（否则 rank 间 batch 形状不同 → collective hang），
所以每轮做一次 all-gather。

三道防饿死阀门：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--prefill-delayer-token-usage-low-watermark` | None | KV 使用率低于此 → GPU 明显空闲，立即放行 |
| `--prefill-delayer-queue-min-ratio` | None | `waiting_queue >= min(running * ratio, prefill_max_requests)` 时放行 |
| `--prefill-delayer-max-delay-ms` | 5000 | 墙钟上限，**界定最坏 TTFT** |
| `--prefill-delayer-max-delay-passes` | 30 | forward 数上限 |

文档字符串的 tagline 精确概括：

> Enable prefill delayer for DP attention to reduce idle time.

### `MinFreeSlotsDelayer`（`managers/min_free_slots_delayer.py`）

```python
def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
    return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
```

**区别**：PrefillDelayer 等的是"别的 rank"，这个是等"本 rank 的 req 槽位"。
目的是**把零散的一次一个 prefill 攒成一次批量 prefill**（"batching them into one
admission instead of one at a time"）。**per-rank 本地决策，不需要通信**。

### `--prefill-decode-interval`

反向操作：一次 prefill 后强制跑 N 轮 decode 再调下一次 prefill。

```python
# scheduler.py:1350
def _should_defer_prefill(self) -> bool:
    if self._prefill_decode_interval_remaining == 0:
        return False
    self._prefill_decode_interval_remaining -= 1
    return True
```

**用途**：给刚进 decode 的请求一段不受 prefill 干扰的时间窗（保护 TPOT 稳定性）。
DP 下跨 rank 同步计数。

---

## 9. `run_batch`：三段结构（:4311）

```python
def run_batch(self, batch, ...):
    # 段 1：PD prefill 提前发 KV —— 与尾部 forward 重叠
    self.maybe_send_cached_prefix_chunk(batch)

    # 段 2：forward
    with self.forward_stream_ctx:
        self.forward_stream.wait_stream(self.schedule_stream)
        resolve_forward_inputs(batch, self.future_map)          # 从 relay 取 input_ids
        batch_result = self.model_worker.forward_batch_generation(batch, **fwd_kwargs)
        if batch.spec_algorithm.is_none():
            self.future_map.publish(future_indices, batch.seq_lens + 1)
        # D2H 放在独立 copy_stream 上
        self.copy_stream.wait_stream(self.forward_stream)
        with self.copy_stream_ctx:
            batch_result.copy_to_cpu(return_logprob=..., return_hidden_states=...)

    # 段 3：WAR 屏障
    self._apply_war_barrier()
```

### spec 的 publish 时机不同（:4373）

```python
if not batch.spec_algorithm.is_none():
    fwd_kwargs["on_publish"] = partial(self.future_map.publish, future_indices)
    if batch.spec_algorithm.supports_grammar_overlap():
        fwd_kwargs["grammar_barrier"] = self._advance_pending_grammar
batch_result = self.model_worker.forward_batch_generation(batch, **fwd_kwargs)
if batch.spec_algorithm.is_none():
    self.future_map.publish(future_indices, batch.seq_lens + 1)
```

注释是关键：

> Spec_v2 fires on_publish mid-worker (between verify and draft_extend) so
> schedule prep can overlap with draft_extend. Non-spec has no later work —
> scheduler publishes after return.

**投机路径把 publish 提前到 verify 结束、draft_extend 开始之前**，
让 CPU 侧的调度准备与 draft_extend 的 GPU 时间重叠。

### `process_batch_result`（:4738）

```python
flush_trace_batch(batch.reqs)
snapshot = self.publish_load_snapshot(force=batch.forward_mode.is_extend())
self.load_publisher.publish_load_stat(...)
if batch.forward_mode.is_decode():
    self.batch_result_processor.process_batch_result_decode(batch, result)
elif batch.forward_mode.is_extend():
    self.batch_result_processor.process_batch_result_prefill(batch, result)
...
self.tree_cache.flush_pending_backups()
```

注释点明 overlap 的意义：

> Flush async trace ops here: in overlap mode this CPU work runs while the
> next batch's GPU forward is in flight, giving free overlap.

---

## 10. 相关 CLI 速查

| Flag | 默认 | 作用 |
|---|---|---|
| `--schedule-policy` | `fcfs` | 9 种排队策略 |
| `--schedule-conservativeness` | 1.0 | 缩放 `new_token_ratio` 初值 |
| `--chunked-prefill-size` | 自动 | 单 chunk token 上限；`-1` 关闭 |
| `--enable-mixed-chunk` | False | chunk 与 decode 同批 |
| `--max-prefill-tokens` | 16384 | 单 prefill 批 token 上限 |
| `--enable-dynamic-chunking` | False | PP 下按拟合模型定 chunk |
| `--prefill-decode-interval` | 自动 | prefill 后强制 N 轮 decode |
| `--max-running-requests` | 自动 | 并发上限 |
| `--disable-overlap-schedule` | False | 关掉 CPU/GPU overlap |
| `--retraction-policy` | `length` | `length` / `priority` |
| `--enable-priority-scheduling` | False | 优先级调度 |
| `--priority-scheduling-preemption-threshold` | 10 | 抢占阈值 |
| `--enable-prefill-delayer` | False | DP 下延迟 prefill |
| `--min-free-slots-delay` | 自动 | 攒批 prefill |
| `--num-continuous-decode-steps` | 1 | **已废弃（无读取点）** |
| `--scheduler-recv-interval` | 1 | 降低收包轮询开销 |

### 环境变量

```
SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
SGLANG_FORCE_COARSE_WAR_BARRIER
SGLANG_ENABLE_WAR_BARRIER
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR
SGLANG_INIT_NEW_TOKEN_RATIO / SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR
SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS / SGLANG_RETRACT_DECODE_STEPS
SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION          (默认 4096)
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD        (默认 32)
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD (默认 32)
```

> ⚠️ `--num-continuous-decode-steps` 只存在于参数声明和 `field_order.py:75`，
> **没有任何读取点**——已废弃，不要用它。

---

## 11. 自检

1. `_determine_active_policy` 里 128 这个阈值背后的权衡是什么？
2. `shortest_prefill_chunk_limit` 为什么必要？不设会怎样？
3. 批内前缀去重（影子树）省掉了什么？
4. `retract_decode` 为什么用 `while` 而不是 `if`？
5. `new_token_ratio` 的分子分母分别代表什么？
6. 抢占循环里的 `break` 为什么可以 early-exit？
7. spec 路径为什么要把 publish 提前到 draft_extend 之前？

---

下一页 → [05 Overlap 与延迟隐藏](05_overlap_and_latency_hiding.md)

# 进阶 02 · 进程拓扑、IPC 与异步桥接

> 对应入门 [02 架构总览](../beginner/02_architecture_overview.md)。
> 源码：`entrypoints/engine.py:1051`、`managers/tokenizer_manager.py`、
> `managers/detokenizer_manager.py`、`managers/scheduler_components/ipc_channels.py`

---

## 1. 进程编排的顺序

`Engine._launch_subprocesses`（`engine.py:1051`）的顺序是刻意的：

```python
configure_logger(server_args)
server_args.resolve_once()
_set_envs_and_config(server_args)
load_plugins()
server_args.check_server_args()
# 解析 "auto" 的 reasoning/tool parser（需要 tokenizer 和 chat template）
parsers = resolving_view(server_args)
if parsers.reasoning_parser == "auto" or parsers.tool_call_parser == "auto":
    resolve_auto_parsers(server_args)

context_before_publish = snapshot_context()      # ← 失败可回滚的运行上下文快照
publish(server_args, role="tokenizer")

try:
    port_args = PortArgs.init_new(server_args)
    ...
except BaseException:
    restore_context(context_before_publish)      # 回滚
    raise

scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(...)
...
detoken_procs, detoken_names = cls._launch_detokenizer_subprocesses(...)
tokenizer_manager, template_manager = init_tokenizer_manager_func(server_args, port_args)
...
scheduler_init_result.wait_for_ready()           # 阻塞等模型加载完
```

三个值得注意的设计：

1. **`snapshot_context` / `restore_context`**：SGLang 把全局配置收拢成
   `RuntimeContext`（`runtime_context.py`），启动失败时可以整体回滚，
   而不是留下半初始化的全局状态。这对"同一个进程里先后启动两个 Engine"的场景很重要。

2. **`publish(server_args, role="tokenizer")`**：配置是**分阶段发布**的。
   注释解释了一个顺序要求：

   > Not read-only: the LoRA checks normalize adapter paths through late
   > resolution, which a published config refuses. Hence before publish.

3. **Scheduler 先起，TokenizerManager 后起**，然后才 `wait_for_ready()`。
   为什么？因为 TokenizerManager 里有个 bootstrap server，需要在 scheduler 的
   `get_init_info()` 里登记端口。

### GPU 分配

```python
gpu_id = base_gpu_id + (pp_rank % pp_size_per_node) * tp_size_per_node \
                    + (tp_rank % tp_size_per_node) * gpu_id_step
```

TP 优先填满一个节点，再填下一个 PP stage——这样 TP 组的通信永远在机内。

### 进程数

```
Scheduler 进程数 = tp_size × pp_size        （dp_size > 1 时经 DP Controller 再乘）
Detokenizer 进程数 = 1（--detokenizer-worker-num > 1 时 1 router + N workers）
DP Controller = 1（仅 dp_size > 1）
```

---

## 2. 五组 ZMQ 端点

`PortArgs` 在 `server_args.py:798`（**不在** io_struct.py）：

```python
@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str           # Detokenizer/Scheduler → TokenizerManager
    scheduler_input_ipc_name: str     # TokenizerManager → Scheduler(rank0)
    detokenizer_ipc_name: str         # Scheduler → Detokenizer
    nccl_port: int
    rpc_ipc_name: str                 # Engine ↔ Scheduler
    metrics_ipc_name: str
    tokenizer_worker_ipc_name: str | None
    decoupled_spec_ipc_config: DecoupledSpecIpcConfig | None
```

`init_new` 在 `enable_dp_attention` 为 False 时用 `ipc://` 临时文件（Unix domain socket，
同机零拷贝语义），否则用 TCP。

### Socket 类型（权威来源：`scheduler_components/ipc_channels.py:34`）

| 端点 | 发起方 | 类型 |
|---|---|---|
| `scheduler_input_ipc_name` | TokenizerManager **PUSH connect** | Scheduler rank0 **PULL bind** |
| `tokenizer_ipc_name` | Scheduler **PUSH connect**；Detokenizer **PUSH connect** | TokenizerManager **PULL bind** |
| `detokenizer_ipc_name` | Scheduler **PUSH connect** | Detokenizer **PULL bind** |
| `rpc_ipc_name` | Scheduler **DEALER connect** | Engine 侧 bind |
| `metrics_ipc_name` | Scheduler **PUSH connect** | metrics server bind |

**只有 `attn_tp_rank==0 && pp_rank==0 && attn_cp_rank==0` 的 rank0 scheduler 持有这些
socket**（`scheduler.py:789 is_rank_zero`）。其他 rank 是 `None`，靠 NCCL 汇聚到 rank0。

> 这是 TP 下 IPC 开销不随 TP 增长的原因：**只有 rank0 做 IPC，其余 rank 只做 NCCL**。

### 两个容易忽略的分支

```python
# ipc_channels.py:56
if skip_tokenizer_init:
    # send_to_detokenizer 直接连到 tokenizer_ipc_name，跳过 Detokenizer 进程
```

纯 token-id 场景（RL rollout、离线评测）可以省掉一个进程跳。

`recv_from_rpc` 用 **DEALER**（不是 PUSH）——因为 Engine 与 Scheduler 之间
需要双向 req/rep，PUSH/PULL 是单向的。

---

## 3. 序列化：msgspec + 两种逃生舱

### 主路径

```python
# io_struct.py:83
class BaseReq(msgspec.Struct, tag=True, array_like=True):
```

- **`array_like=True`**：按**位置**编码而不是"字段名: 值"，去掉所有字段名
- **`tag=True`**：接收端能按 tag 直接反序列化成具体类型，`TypeBasedDispatcher` 靠它分派

在一个每 token 都要发一次 IPC 的系统里，这两个是必须的。

### 逃生舱 ①：`PickleWrapper`

```python
# io_struct.py:2430
def wrap_as_pickle(...) / unwrap_from_pickle(...)
```

复杂对象（`time_stats`、`customized_info`、tensor list）绕过 msgpack，
用一个 `PickleWrapper` 结构体包一层。

### 逃生舱 ②：整体切换后端

```python
# io_struct.py:2467
SGLANG_USE_PICKLE_IPC=1     # 整体切成 pickle
```

### 发送侧的优化

`output_sender.py:11 SenderWrapper.send_output()` 会回填
`recv_obj.http_worker_ipc`（多 tokenizer worker 模式下的回程路由）。

`output_streamer.py:270` 附近有一个具体的例子：

```python
# Stacked:     [stacked_tensor(N, ...)] — len 1, N > 1 requests
# Non-stacked: [tensor_0, tensor_1, ...] — len == N
# Stacking reduces N pickle/__reduce_ex__ calls to 1.
```

**pooled hidden states 用 `torch.stack` 打成一个 tensor 过一次 pickle**，
避免 N 次 `__reduce_ex__`。

---

## 4. TokenizerManager：async/sync 的桥

`tokenizer_manager.py:413`。这是整个系统里最难的一段异步代码。

### `__init__` 的编排

```python
self.init_model_config()
self.init_tokenizer_and_processor()
self.init_ipc_channels(port_args)          # zmq.asyncio.Context(2)  ← 注意是 asyncio
self.init_running_status()                 # self.rid_to_state = {}
self.init_request_logging_and_dumping()
self.init_weight_update()
self.init_lora()
self.init_disaggregation(...)
self.init_metric_collector_watchdog()
self.init_request_dispatcher()
self.cuda_vmm_feature_transport = CudaVmmFeatureTransport(...)
```

```python
# :622 init_ipc_channels
context = zmq.asyncio.Context(2)           # ← 不是 zmq.Context
self.recv_from_detokenizer = get_zmq_socket(context, zmq.PULL, port_args.tokenizer_ipc_name, True)
self.send_to_scheduler = get_zmq_socket(context, zmq.PUSH, port_args.scheduler_input_ipc_name, True)
```

**用 `zmq.asyncio` 是为了让 `await sock_recv(...)` 直接挂起协程**，
而不是阻塞整个事件循环。

### `rid → asyncio.Event` 的映射

```python
# :235
class ReqState:
    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]
    time_stats: APIServerReqTimeStats
    last_completion_tokens: int = 1
    ttft_observed: bool = False
    dispatched: bool = False
    abort_sent: bool = False
    last_output_offset: int = 0
    text: str = ""
    text_chunks: List[str] = dataclasses.field(default_factory=list)
```

`text_chunks` + `append_text()` 是**增量流式的优化**：

> Accumulate text lazily so incremental streaming can emit the incoming delta
> directly without rebuilding the full output prefix.

### 三个关键方法

**`generate_request`（:848）** —— async generator，整条链路的"客户端桩"：

```python
self.auto_create_handle_loop()
obj.normalize_batch_and_arguments()
self._init_req_state(obj, request)          # 建 ReqState + asyncio.Event
try:
    async with self.model_update_lock.reader_lock:      # ← 读写锁
        await self._validate_and_resolve_lora(obj)
        if obj.is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            await self._send_one_request(tokenized_obj)
            async for response in self._wait_one_response(obj, request):
                yield response
```

`model_update_lock.reader_lock` 让**权重更新（writer）与请求（reader）互斥，
但请求之间可并发**。

**`_send_one_request`（:1665）** —— PUSH 后立刻返回：

```python
self._dispatch_to_scheduler(tokenized_obj)   # PUSH，非阻塞
self._mark_state_dispatched(rid)
# 不 await 结果
```

**`_stream_one_response`（:1828）** —— 核心等待循环：

```python
while True:
    if request is None:
        await state.event.wait()                       # Engine 路径
    else:
        await asyncio.wait_for(state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT)
    # Drain all pending outputs atomically.
    out_list = state.out_list
    state.out_list = []
    finished = state.finished
    state.event.clear()
```

三个要点：

1. **先取走 `out_list` 再 `clear`**——原子 drain，每次唤醒把累积的 chunk 一次拿走。
2. **HTTP 路径有超时**，超时后检查 `await request.is_disconnected()`，客户端断了就
   `abort_request(rid)` 并抛 `ValueError`。**这是防止断连请求泄漏 GPU KV 的关键机制。**
3. **非流式路径下 `_handle_batch_output` 会把中间 chunk 的 `"text"` 设为 `None`**：

   ```
   avoid O(n) string rebuild per step (O(n^2) total)
   ```

   只在最后 `state.get_text()` 拼一次。**这是个很容易被忽略但很严重的 O(n²) 陷阱。**

### 后台接收循环

```python
# :2317 handle_loop
async def handle_loop(self):
    while True:
        with self.soft_watchdog.disable():
            recv_obj = await async_sock_recv(self.recv_from_detokenizer)
        if isinstance(recv_obj, (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput)):
            await self._handle_batch_output(recv_obj)
        else:
            self._result_dispatcher(recv_obj)      # 控制类响应走这里
        self.last_receive_tstamp = real_time()
        self.soft_watchdog.feed()
```

**`recv_from_detokenizer` 这个名字有历史误导性**——它同时接收 detokenizer 的
正常输出和 scheduler 直发的控制响应。

### `batch_notify_size`：一个性能旋钮

```python
# :2332 _handle_batch_output
if out_dict is not None:
    state.out_list.append(out_dict)
    pending_notify[rid] = state
    if len(pending_notify) >= batch_notify_size:
        for s in pending_notify.values():
            s.event.set()
        pending_notify = {}
        await asyncio.sleep(0)      # 让出控制权，让 waiter 跑
```

- **太大** → TTFT 变差（等攒够才唤醒）
- **太小** → 每 token 一次 context switch

`await asyncio.sleep(0)` 这一句很重要：它把控制权交给事件循环，
让所有刚 `set()` 的 waiter 有机会跑。

---

## 5. DetokenizerManager：为什么必须是独立进程

`detokenizer_manager.py:102`。

### 理由

detokenize 是 CPU 密集的 Python/HuggingFace tokenizer 调用：

- 放在 Scheduler 进程里 → **阻塞调度循环**，TTFT 直接劣化
- 放在 TokenizerManager（主进程）里 → **阻塞 HTTP 事件循环**

独立成进程，三方天然并行：**GPU 在算、scheduler 在调度、detokenizer 在解码文本**。

### `DecodeStatus`：增量解码的状态

```python
class DecodeStatus:
    """Store the status of incremental decoding."""
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    sent_offset: int = 0
    decoded_text_len: int = dataclasses.field(init=False)
    decoded_text_chunks: List[str] = dataclasses.field(default_factory=list)
```

存在 `self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)`——
**有容量上限，LRU 淘汰**，防内存无界增长。

### 主循环极简

```python
def event_loop(self):
    while True:
        with self.soft_watchdog.disable():
            recv_obj = sock_recv(self.recv_from_scheduler)
        output = self._request_dispatcher(recv_obj)
        if output is not None:
            sock_send(self.send_to_tokenizer, output)
        self.soft_watchdog.feed()
```

**同步（非 asyncio）ZMQ**——这里不需要异步并发，单进程单线程足够。

### surrogate context 技巧：最核心的一段

```python
# :283 _decode_batch_token_id_output
read_ids.append(self.trim_matched_stop(
    s.decode_ids[s.surr_offset:], ...))
surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])
```

- `surr_ids` = "上次读到的位置之前的全部 token"（**surrogate context**）
- `read_ids` = "到现在为止的全部 token"

两者都送进 tokenizer 做 decode，然后**相减取差**得到真正的新增文本。

**为什么需要这样？** BPE 的 token 不保证 UTF-8 对齐——一个中文字可能横跨两个 token，
单独 decode 第二个 token 会得到乱码。用 surrogate context 让 tokenizer 看到完整
上下文，再减掉已输出的部分，既不会切碎多字节字符，也不需要重解码整个前缀。

### 三个叠在一起的批处理优化

```python
# :242 _grouped_batch_decode
if not all(ids_list):
    keep_idx = [i for i, ids in enumerate(ids_list) if ids]     # (a) 过滤空 span
    ...
if not getattr(self.tokenizer, "is_fast", False):
    decoded = [decode_without_hf_kwargs(...) ...]
else:
    first_skip, first_space = skip_list[0], space_list[0]
    if all(s == first_skip and sp == first_space for s, sp in zip(skip_list, space_list)):
        decoded = self.tokenizer.batch_decode(ids_list, ...)     # (b) fast tokenizer 走 batch_decode
    else:
        groups: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)   # (c) 按 flag 分组
```

(a) 过滤空 span（高并发流式下 per-row 开销会累积）
(b) fast tokenizer 走 `batch_decode`（一次 Rust 调用而不是 N 次）
(c) 按 `(skip_special_tokens, spaces_between_special_tokens)` 分组，同组共享一次调用

`trim_matched_stop`（:189）在这里做 stop string / stop token 截断——
只有拿到文本才能找到 stop string 的位置。

---

## 6. Scheduler 侧的接收与发送

### 分派

```python
# scheduler.py:2090 process_input_requests
def process_input_requests(self, recv_reqs):
    for recv_req in recv_reqs:
        output = self._request_dispatcher(recv_req)
        if output is not None:
            self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)
```

`TypeBasedDispatcher` 按 msgspec tag 一次查表分派到 `handle_generate_request` /
`abort_request` / `flush_wrapper.handle` / …

**生成类请求无返回值**（进 waiting_queue）；**控制类请求有返回值**则直接回
`send_to_tokenizer`——**不经过 detokenizer**。

### 输出发送

`output_streamer.py:45 SchedulerOutputStreamer`，在 :213：

```python
self.send_to_detokenizer.send_output(payload)
```

---

## 7. HTTP 层

`http_server.py:206` 模块级 `_global_state`，在 `launch_server` 里
`set_global_state(...)` 挂上 tokenizer_manager / template_manager / scheduler_info。

```python
# :918
async def generate_request(obj: GenerateReqInput, request: Request):
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    if obj.stream:
        async def stream_results() -> AsyncIterator[bytes]:
            async for out in _global_state.tokenizer_manager.generate_request(obj, request):
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"
        return StreamingResponse(stream_results(), media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj))
    else:
        ret = await _global_state.tokenizer_manager.generate_request(obj, request).__anext__()
        return orjson_response(ret)
```

两个细节：

1. **`background=create_abort_task(obj)`**：流结束（正常或异常）后触发清理。
2. **`orjson_response`**：用 orjson 而不是标准 json。序列化 `[B, 128000]` 的
   logprob 时差别很大。

---

## 8. 性能优化清单（本章汇总）

| # | 优化 | 位置 | 收益 |
|---|---|---|---|
| 1 | msgspec `array_like=True` + `tag=True` | `io_struct.py:83` | 去掉字段名 + O(1) 类型分派 |
| 2 | `array('q')` 而非 `List[int]` | `schedule_batch.py` | 避免逐 int 装箱 |
| 3 | `TypeBasedDispatcher` dict 查表 | `scheduler.py:1733` | 避免长 if-elif 链 |
| 4 | `batch_notify_size` 批量唤醒 | `tokenizer_manager.py:2638` | 摊薄 asyncio context switch |
| 5 | 非流式 `text=None` 延迟拼接 | `tokenizer_manager.py:1876` | **避免 O(n²) 字符串重建** |
| 6 | surrogate-diff 增量解码 | `detokenizer_manager.py:283` | 每步只 decode 新增 span |
| 7 | `_grouped_batch_decode` | `detokenizer_manager.py:242` | 空过滤 + batch_decode + 分组 |
| 8 | PHS `torch.stack` 合并 | `output_streamer.py:270` | N 次 pickle → 1 次 |
| 9 | `PickleWrapper` 逃生舱 | `io_struct.py:2430` | 复杂对象不走 msgpack |
| 10 | 延迟采样 `delay_sample_func` | `tp_worker.py:700` | 采样启动开销藏进 CPU 调度 |
| 11 | 只有 rank0 持 IPC socket | `ipc_channels.py` | IPC 开销不随 TP 增长 |
| 12 | `skip_tokenizer_init` 旁路 detokenizer | `ipc_channels.py:56` | 省一个进程跳 |
| 13 | `asyncio.sleep(0)` 让出控制权 | `tokenizer_manager.py:2646` | 批量唤醒不饿死 waiter |

---

## 9. 自检

1. 为什么 `recv_from_rpc` 用 DEALER 而其他用 PUSH/PULL？
2. 为什么 `zmq.asyncio.Context` 而不是 `zmq.Context`？
3. 独立 Detokenizer 进程的收益具体是什么？
4. surrogate context 技巧解决什么问题？不用它会怎样？
5. 非流式请求为什么要把中间 chunk 的 `"text"` 设为 `None`？
6. 为什么只有 rank0 需要持有 IPC socket？

---

下一页 → [03 Radix Cache 内部实现](03_radix_cache_internals.md)

# 02 · 架构总览与请求生命周期

> 对应进阶 [02 进程拓扑与 IPC](../advanced/02_process_topology_and_ipc.md)。
> 源码入口：`python/sglang/srt/entrypoints/engine.py:1051 _launch_subprocesses`

---

## 2.1 SGLang 由四个角色组成

启动 `python -m sglang.launch_server` 之后，系统里有四种进程：

```
┌──────────────────────────────────────────────────────────────────┐
│  主进程（无 GPU）                                                 │
│  ┌─────────────────────┐   ┌──────────────────────┐              │
│  │  HTTP Server        │──▶│  TokenizerManager    │              │
│  │  (FastAPI/uvicorn)  │   │  · 分词              │              │
│  │  /generate          │   │  · rid → asyncio.Event│             │
│  │  /v1/chat/...       │   │  · 流式响应组装       │              │
│  └─────────────────────┘   └──────────┬───────────┘              │
└────────────────────────────────────────┼─────────────────────────┘
                                         │ ZMQ PUSH
                    ┌────────────────────▼──────────────────────┐
                    │  Scheduler × (tp_size × pp_size)          │
                    │  · 排队 / 组批 / 前缀匹配 / 分配 KV        │
                    │  · 驱动 ModelRunner 做 forward            │
                    │  · 采样出 token                            │
                    └───────┬───────────────────────┬───────────┘
                            │ ZMQ PUSH              │ ZMQ PUSH
                            │ (生成结果)             │ (控制响应，直达)
                            ▼                       │
          ┌─────────────────────────────┐           │
          │  Detokenizer（独立进程）      │           │
          │  · token id → 文本           │           │
          │  · 增量解码 / stop string    │           │
          └──────────────┬──────────────┘           │
                         │ ZMQ PUSH                  │
                         └──────────┬────────────────┘
                                    ▼
                        TokenizerManager.recv_from_detokenizer
```

### 为什么拆成四个进程

| 角色 | 拆出去的理由 |
|---|---|
| **HTTP Server + TokenizerManager** | 分词是纯 CPU 的 Python 调用。放在 scheduler 里会阻塞调度循环，TTFT 直接劣化。 |
| **Scheduler** | 唯一持有 GPU 的角色。每个 TP rank 一个进程（GIL 隔离，且 CUDA context 必须每进程一个）。 |
| **Detokenizer** | 反分词也是 CPU 密集的 Python 调用。独立进程让它与 GPU forward **天然并行**。 |

**一句话总结**：GPU 在算的时候，CPU 必须能同时分词、反分词、组下一批。
这就是全部四个进程的理由。

### 进程数量

```
Scheduler 进程数 = tp_size × pp_size          （dp_size > 1 时再乘以 dp_size）
Detokenizer 进程数 = 1（--detokenizer-worker-num > 1 时 1 个 router + N 个 worker）
```

TP/PP 的 rank 都在 engine.py:880 附近分配 GPU：

```python
gpu_id = base_gpu_id + (pp_rank % pp_size_per_node) * tp_size_per_node \
                    + (tp_rank % tp_size_per_node) * gpu_id_step
```

---

## 2.2 请求生命周期：14 步走查

以 `POST /generate`（`stream=False`）为例：

| # | 位置 | 设备 | 动作 |
|---|---|---|---|
| 1 | `http_server.py:918 generate_request` | CPU | FastAPI 反序列化 JSON → `GenerateReqInput` |
| 2 | `tokenizer_manager.py:848 generate_request` | CPU | 分配 `rid`，建 `ReqState`（含 `asyncio.Event`），用 HF tokenizer 分词 |
| 3 | `tokenizer_manager.py:1665 _send_one_request` | CPU | **PUSH 发送后立刻返回，不等待**；协程挂起在 `await state.event.wait()` |
| 4 | `scheduler.py:2090 process_input_requests` | CPU | 收到、分派到 `handle_generate_request`，建 `Req`，进 `waiting_queue` |
| 5 | `scheduler.py:3790 get_new_batch_prefill` | CPU | **前缀匹配 + 预算检查 + 组批**，产出 `ScheduleBatch`（EXTEND） |
| 6 | `scheduler.py:4311 run_batch` → `model_runner.py` | **GPU** | prefill forward：把新 token 的 KV 写进池子 |
| 7 | `model_runner.sample()` | GPU | 采样出第一个 token |
| 8 | `scheduler.py:4738 process_batch_result` | CPU | 写回 `req.output_ids`；`tree_cache.cache_unfinished_req()` 把 KV 挂进 radix tree |
| 9 | `output_streamer.py:213` | CPU | 组 `BatchTokenIDOutput`，PUSH 给 Detokenizer |
| 10 | 每步重复 | **GPU** | `update_running_batch` → DECODE → **CUDA Graph replay** 出 1 个 token |
| 10′ | `scheduler.py:1948 event_loop_overlap` | CPU | **overlap**：上一批的 CPU 后处理与当前批的 GPU forward 并行 |
| 11 | `detokenizer_manager.py:179` | CPU | 增量反分词 + stop string 截断 → `BatchStrOutput` |
| 12 | `tokenizer_manager.py:2317 handle_loop` | CPU | 塞进 `state.out_list`，攒够 `batch_notify_size` 后 `event.set()` |
| 13 | `tokenizer_manager.py:1828 _stream_one_response` | CPU | 醒来 drain，finished 则 `yield` |
| 14 | `http_server.py:918` | CPU | `orjson` 序列化成 HTTP 响应 |

**关键状态切片**：

- 第 3 步之后 HTTP 协程立刻挂起，主进程去处理别的请求 → **请求之间天然并发**。
- **GPU 只在第 6、10 步忙**。其余全是 CPU。这就是为什么"CPU 侧的每一步都很重要"。
- 第 8/9 步（上一批的后处理）与第 10 步（当前批的 GPU 计算）在 overlap 模式下并行。

---

## 2.3 两个关键数据结构

### `Req`：调度器眼中的请求（`schedule_batch.py:977`）

```python
class Req:
    rid: str
    origin_input_ids: List[int]        # prompt
    output_ids: array("q")             # 已生成的 token，append-only
    full_untruncated_fill_ids: array("q")   # prompt + output，全序列
    extend_range: Optional[Range]      # 本轮 chunked prefill 要算的区间
    prefix_indices: torch.Tensor       # 前缀命中得到的 KV 槽位（第 03 章）
    kv: ReqKvInfo                      # 该请求的 KV 索引信息
    sampling_params: SamplingParams
    ...
```

`array("q")` 而不是 `list[int]`——避免每个整数都装箱成 Python 对象。
在每步都要遍历几万个 token 的热路径上，这个差别很大。

### `ForwardBatch`：ModelRunner 眼中的一批（`forward_batch_info.py`）

从 `ScheduleBatch` 转换而来，是**真正喂给模型的张量包**：

```python
class ForwardBatch:
    forward_mode: ForwardMode      # EXTEND / DECODE / IDLE / TARGET_VERIFY / ...
    batch_size: int
    input_ids: torch.Tensor        # [总 token 数]
    req_pool_indices: torch.Tensor # 每个请求在 req_to_token_pool 里的行号
    seq_lens: torch.Tensor         # 每个请求的当前长度
    out_cache_loc: torch.Tensor    # 本轮新 KV 写到池子的哪些槽位 ← 核心
    ...
```

`ForwardMode` 是理解 SGLang 的另一个钥匙：

| 模式 | 含义 |
|---|---|
| `EXTEND` | prefill（或 chunked prefill 的一块） |
| `DECODE` | 逐 token 解码 |
| `IDLE` | DP attention 下这个 rank 没事干（**必须有**，否则 DP 组会死锁） |
| `TARGET_VERIFY` | 投机解码的目标模型验证 |
| `DRAFT_EXTEND_V2` | 投机解码的 draft 模型扩展 |

---

## 2.4 IPC：五条 ZMQ 通道

端点在 `PortArgs`（`server_args.py:798`）：

```python
@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str        # Scheduler/Detokenizer → TokenizerManager
    scheduler_input_ipc_name: str  # TokenizerManager → Scheduler
    detokenizer_ipc_name: str      # Scheduler → Detokenizer
    nccl_port: int
    rpc_ipc_name: str              # Engine ↔ Scheduler（双向，用 DEALER）
    metrics_ipc_name: str
    tokenizer_worker_ipc_name: str | None
```

| 端点 | 发起方 | Socket 类型 |
|---|---|---|
| `scheduler_input_ipc_name` | TokenizerManager **PUSH** connect | Scheduler rank0 **PULL** bind |
| `tokenizer_ipc_name` | Scheduler **PUSH** / Detokenizer **PUSH** | TokenizerManager **PULL** bind |
| `detokenizer_ipc_name` | Scheduler **PUSH** | Detokenizer **PULL** bind |
| `rpc_ipc_name` | Scheduler **DEALER** connect | Engine 侧 bind |

**只有 rank0 scheduler 持有这些 socket**（`is_rank_zero`，`scheduler.py:789`）。
其他 rank 靠 NCCL 把结果汇聚到 rank0。

两个设计细节：

1. **`ipc://` 而不是 `tcp://`**（同机时）。走 Unix domain socket，跳过 TCP 栈。
2. **序列化用 msgspec 且 `array_like=True`**（`io_struct.py:83`）：

```python
class BaseReq(msgspec.Struct, tag=True, array_like=True):
```

`array_like=True` 让消息按**位置**而不是"字段名: 值"编码，去掉所有字段名；
`tag=True` 让接收端能按 tag 直接反序列化成具体类型（`TypeBasedDispatcher` 靠它分派，
一次 dict 查表，没有 if-elif 链）。

> 在一个每 token 都要发一次 IPC 的系统里，这两条优化是必须的。

---

## 2.5 本章小结与自检

**记住这张图就够了**：

```
HTTP → TokenizerManager → Scheduler(GPU) → Detokenizer → TokenizerManager → HTTP
              ↑                                    ↑
          分词是 CPU 活                        反分词是 CPU 活
          所以必须拆出去                        所以必须拆出去
```

自检：

1. 为什么 Detokenizer 要单独一个进程？合并到 TokenizerManager 会怎样？
2. 一个请求在 scheduler 里总共要"跑"几次 GPU forward？（提示：prefill 1 次 + decode N 次）
3. `out_cache_loc` 是干什么的？（第 06 章会展开）
4. 为什么 `ForwardMode` 需要 `IDLE`？

---

下一页 → [03 RadixAttention](03_radix_attention.md)

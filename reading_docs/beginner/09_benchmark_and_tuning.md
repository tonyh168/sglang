# 09 · 性能测量与调参手册（动手）

> 对应进阶 [13 调参方法论](../advanced/13_tuning_playbook.md)。
> 官方参考：`docs/docs/advanced_features/hyperparameter_tuning.mdx`（82 行，**必读**）

---

## 9.1 先学会看日志

SGLang 每 2 秒打印一次运行状态。这一行是你的"体温计"：

```
Decode batch. #running-req: 233, #token: 370959, token usage: 0.82,
  cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317
```

| 字段 | 含义 | 健康值 |
|---|---|---|
| `#running-req` | 正在跑的请求数（就是 batch size B） | 越高越好（受 token_usage 限制） |
| `#token` | 所有 running 请求的 token 总数 | — |
| `token usage` | **KV 池占用率** | **> 0.9** |
| `cuda graph` | 是否走 CUDA Graph | `True` |
| `gen throughput` | 集群级生成吞吐（token/s） | 越高越好 |
| `#queue-req` | 排队请求数 | **100 – 2000** |

### 三条诊断规则（来自官方文档）

**① `#queue-req: 0` 频繁出现** → 客户端提交太慢。提高请求速率。

**② `token usage < 0.9` 且 `#queue-req > 0`** → 服务端太保守，不敢收请求。

```
原因：用户声明了很大的 max_new_tokens，但请求因为 EOS / stop string 很早就结束了
动作：降低 --schedule-conservativeness 到 0.3
```

**③ `token usage` 很高 + 频繁看到 retract 警告** → 服务端太激进

```
KV cache pool is full. Retract requests. #retracted_reqs: 1,
  #new_token_ratio: 0.9998 -> 1.0000
```

```
动作：提高 --schedule-conservativeness 到 1.3
注意：每分钟出现一次左右是正常的，不用管
```

---

## 9.2 基准测试工具

### 在线服务基准（最常用）

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --model Qwen/Qwen3-8B \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 1000 \
    --request-rate 20 \
    --max-concurrency 128
```

输出包含 **TTFT / TPOT / ITL 的 P50/P90/P99**，以及总吞吐。这是评估用户体验的唯一方式。

常用参数：

| 参数 | 说明 |
|---|---|
| `--request-rate` | 每秒请求数；`inf` 表示不限速（压满） |
| `--max-concurrency` | 最大并发（客户端侧限制） |
| `--dataset-name` | `random` / `sharegpt` / `speed_bench` / 自定义 |
| `--random-input-len` / `--random-output-len` | 合成负载的输入/输出长度 |
| `--random-range-ratio` | 长度的随机范围比例 |
| `--output-file` | 把每条请求的明细写 JSONL（含 `--output-details`） |
| `--cache-report` | 报告前缀缓存命中情况 |
| `--disable-stream` | 非流式（会掩盖 TTFT 的真实分布，一般不推荐） |

### 离线吞吐基准

```bash
python -m sglang.bench_offline_throughput --model-path <model> \
    --num-prompts 1000 --random-input-len 1024 --random-output-len 256
```

### 单 batch 延迟分解

```bash
python -m sglang.bench_one_batch --model-path <model> --batch-size 32 --input-len 1024
```

适合快速验证一个 kernel 改动，不涉及调度。

### Profiler

```bash
# 服务端开启并指定输出目录
python -m sglang.launch_server --model-path <model> --profile-dir /tmp/prof

# 另开终端
curl http://localhost:30000/start_profile
# ... 跑一段时间请求 ...
curl http://localhost:30000/stop_profile
```

生成 Chrome trace，用 `chrome://tracing` 或 Perfetto 打开。

> **看 trace 的顺序建议**：先看 GPU kernel 的空白（bubble）在哪；
> 再看是 CPU 侧太慢（有 gap 但 GPU 空闲）还是通信等待（NCCL kernel 占比高）。

### Prometheus 指标

`--enable-metrics` 后暴露。关键指标（定义见
`python/sglang/srt/observability/metrics_collector.py:76 SchedulerStats`）：

| 指标 | 含义 | 对应章节 |
|---|---|---|
| `sglang:num_running_reqs` | 正在跑的请求数 | 04 |
| `sglang:num_queue_reqs` | 排队数 | 04 |
| `sglang:token_usage` | KV 池占用（max of full/swa/mamba） | 06 |
| `sglang:kv_available_tokens` | 空闲槽位 | 06 |
| `sglang:kv_evictable_tokens` | 可驱逐槽位 | 03 |
| `sglang:cache_hit_rate` | 前缀命中率 | 03 |
| `sglang:gen_throughput` | 生成吞吐 | — |
| `sglang:num_retracted_reqs` | 撤回次数 | 04 |
| `sglang:new_token_ratio` | 自适应比例 | 04 |
| `sglang:spec_accept_length` | 投机接受长度 | 08 |
| `sglang:hicache_host_used_tokens` | HiCache 主机池占用 | 06 |
| `sglang:per_stage_req_latency_seconds` | 分阶段延迟 | 进阶 01 |

---

## 9.3 调参顺序（重要）

**不要一次改多个参数。** 按下面顺序，每步只改一个：

### 第 1 步：把 KV 池调到最大

```bash
# 以 0.01 为步长提高，直到接近 OOM
--mem-fraction-static 0.90
```

**判据**：启动日志里的 `available_gpu_mem`

```
max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384,
max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

- `available_gpu_mem` 在 **5–8 GB** → 健康
- 10–20 GB → 太保守，继续加 `--mem-fraction-static`
- < 5 GB → 危险，可能 OOM

### 第 2 步：确认 token_usage 打满

跑 benchmark，看日志：
- `token usage > 0.9` ✓
- `#queue-req` 在 100–2000 之间 ✓

不满足就调 `--schedule-conservativeness`：
- `token usage` 低 + 队列有货 → **降到 0.3**
- 频繁 retract → **升到 1.3**
- `#queue-req: 0` 频繁 → 客户端问题，不是服务端问题

### 第 3 步：按负载类型选优化

| 你的负载 | 优化 |
|---|---|
| 大量共享前缀（RAG / few-shot / 多轮） | `--schedule-policy lpm` |
| 前缀几乎不共享 | `--disable-radix-cache` 试试（可能有意外收益） |
| 长 prompt（> 8K） | 调 `--chunked-prefill-size`，必要时开 `--enable-mixed-chunk` |
| 长输出 | 考虑量化（decode 收益大）+ 投机解码 |
| 需要低 TTFT | 降 `--chunked-prefill-size`；用 `--schedule-policy shortest-prefill-first` |
| 需要低 TPOT 抖动 | `--enable-mixed-chunk`，`--prefill-decode-interval` |

### 第 4 步：显存不够时的选项

```
1. KV 量化：--kv-cache-dtype fp8_e4m3        ← 最便宜，KV 减半
2. 模型量化：--quantization fp8               ← 权重减半
3. HiCache：--enable-hierarchical-cache       ← 用 CPU 内存换
4. 降 --chunked-prefill-size 到 4096/2048     ← 省激活显存
5. 降 --max-running-requests                  ← 最后手段
```

### 第 5 步：CUDA Graph 扩容

```bash
# 默认只对小 batch 捕获；大 TP 下可以调大
--cuda-graph-max-bs-decode 512
# 注意会多吃显存，可能要同时降 --mem-fraction-static
```

### 第 6 步：并行策略

```bash
# 显存够 → 优先 DP（吞吐最好）
# 显存不够 → TP
--tp-size 8
# MoE 模型 → 考虑 EP / DP attention
--enable-dp-attention --dp-size 8
--ep-size 8
# TTFT 和 TPOT 都要好 → PD 分离
```

---

## 9.4 常见问题诊断表

| 症状 | 可能原因 | 动作 |
|---|---|---|
| TTFT 高、TPOT 正常 | prefill 排队 / chunk 太大 | `--schedule-policy shortest-prefill-first`，降 `--chunked-prefill-size` |
| TPOT 抖动大 | prefill 打断 decode | `--enable-mixed-chunk`，`--prefill-decode-interval` |
| 吞吐上不去、token_usage 低 | 调度太保守 | 降 `--schedule-conservativeness` 到 0.3 |
| 吞吐上不去、token_usage 高 | KV 到顶 | KV 量化 / HiCache / 加卡 |
| 频繁 retract | 预估太乐观 | 升 `--schedule-conservativeness` 到 1.3 |
| cache_hit_rate ≈ 0 | 负载无共享前缀 | `--disable-radix-cache` 对比一下 |
| GPU 利用率低、有 bubble | CPU 调度慢 | 确认 overlap 开启（默认开）；看 trace |
| NCCL 占比高 | TP 过大 | 换 DP，或减 TP |
| `spec_accept_length` < 2 | 投机收益为负 | 关掉投机或换 draft 模型 |
| prefill OOM | 单次激活太大 | 降 `--chunked-prefill-size` 到 4096/2048 |
| decode OOM | 并发太高 | 降 `--max-running-requests` |

---

## 9.5 一个完整的调参例子

**目标**：Qwen3-8B，8 卡，长 prompt（2K）+ 中等输出（512），要求高吞吐。

```bash
# 第 1 版：基线
python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --tp-size 8 --port 30000

# 观察：token usage 0.6, #queue-req 500, gen throughput 3000 tok/s
# 诊断：太保守

# 第 2 版：放开调度
... --schedule-conservativeness 0.3
# 观察：token usage 0.85, gen throughput 4500 tok/s ✓

# 第 3 版：加大 KV 池
... --mem-fraction-static 0.88
# 观察：available_gpu_mem 6.2 GB ✓, token usage 0.9, throughput 5200

# 第 4 版：KV 量化，进一步扩并发
... --kv-cache-dtype fp8_e4m3
# 观察：max_total_num_tokens 翻倍, throughput 6100

# 第 5 版：CUDA Graph 扩容
... --cuda-graph-max-bs-decode 512 --mem-fraction-static 0.86
# 观察：throughput 6500

# 最后用 bench_serving 验证 TTFT/TPOT 分位数没有劣化
python -m sglang.bench_serving --backend sglang --model Qwen/Qwen3-8B \
    --dataset-name random --random-input-len 2048 --random-output-len 512 \
    --num-prompts 500 --request-rate 20
```

**每一步只改一个参数，每次都记录 token_usage 和 throughput。**

---

## 9.6 本章自检

1. `token usage` 0.6、`#queue-req` 500，你改什么？
2. `available_gpu_mem` 是 15 GB，说明什么？
3. KV 快满了，按成本从低到高的扩容手段依次是什么？
4. `spec_accept_length` 1.3，你应该做什么？
5. 为什么调参要一次只改一个？

---

下一页 → [10 术语表与源码地图](10_glossary_and_source_map.md)

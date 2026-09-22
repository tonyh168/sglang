# 进阶 01 · Roofline 模型与 SGLang 的延迟分解

> 对应入门 [01 为什么需要推理引擎](../beginner/01_why_inference_engine.md)。
> 目标：把"直觉"变成"公式"，让你能定量判断一个优化值不值得做。

---

## 1. Roofline：一切优化的判据

Roofline 模型把任何 kernel 的性能上限写成：

```
可达性能 = min(峰值算力, 算术强度 × 内存带宽)

算术强度 I = FLOPs / Bytes_moved
```

机器的**拐点**（ridge point）是：

```
I_ridge = peak_flops / peak_bandwidth
```

对 H100 SXM（FP16/BF16 稠密）：`I_ridge ≈ 989e12 / 3.35e12 ≈ 295 FLOP/Byte`

| 阶段 | 算术强度 | 位置 | 优化方向 |
|---|---|---|---|
| Prefill（T=4096） | ~4096 FLOP/Byte | 远在拐点右侧 | 提升算力利用率：更大 tile、更好的 GEMM kernel |
| Decode（batch=1） | ~1 FLOP/Byte | 远在拐点左侧 | **减少字节搬运**：量化、GQA/MLA、batch 增大 M 维 |

### 关键推论

**Decode 的算术强度 ≈ batch_size。** 因为：

```
I = (2 × B × K × N) / (2 × K × N) = B        # 权重只读一次，B 行输入
```

所以 **batch 从 1 → 32 直接把算术强度提升 32 倍**，而 H100 从 memory-bound 转为
compute-bound 需要在 batch ≈ 295 附近（实际受 M 维 GEMM 效率影响，通常在 100–200
就开始收益递减）。

这解释了 SGLang 日志里的核心指标：

```
Decode batch. #running-req: 233, #token: 370959, token usage: 0.82,
  cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317
```

- `#running-req: 233` —— 就是你上面说的 B。它是吞吐的直接来源。
- `token usage: 0.82` —— KV 池占用率。**它才是 B 的上限**（见第 08 章）。
- `#queue-req: 317` —— 队列里有货，说明还没饿着，说明 token_usage 可以再往上压。

---

## 2. 为什么量化对 decode 收益巨大

FP16 → FP8 权重：字节数减半，`I` 翻倍。对 memory-bound 的 decode 是**近似线性加速**。
FP16 → INT4：字节数减到 1/4。

但对 prefill（compute-bound）**没有收益**，反而可能因为 dequant 开销变慢。

这就是为什么 SGLang 的 `--quantization fp8` 在 decode 主导的负载（长输出、小 batch）
上收益明显，在 prefill 主导的负载（长 prompt、短输出）上收益有限。

> **实践判据**：先看你的负载是 prefill-bound 还是 decode-bound，再决定量化级别。

---

## 3. 延迟分解：一个请求的时间都花在哪

SGLang 的 `Req` 会记录分段耗时（`python/sglang/srt/observability/req_time_stats.py`），
prometheus 暴露为 `sglang:per_stage_req_latency_seconds`（按 stage 打标签）。
理解这张表比看任何博客都有用：

```
TTFT = 排队时间 + 分词时间 + prefill 前向时间 + 采样时间 + 反分词时间
       └─queue─┘  └tokenize┘  └─── forward ───┘

TPOT = (decode 前向时间 + 采样 + 调度开销) / (1 + 投机的接受 token 数)
                     ↑                    ↑
              主要成本在这里        spec_accept_length 是这里的杠杆
```

### 逐项拆解与优化手段

| 时间项 | 归谁管 | 优化手段 | 章节 |
|---|---|---|---|
| queue（排队） | 调度器 | 提高 token_usage、DP 扩容、router 分流 | [04](../beginner/04_scheduler.md), [10](10_parallelism_internals.md) |
| tokenize | TokenizerManager | 多 tokenizer 进程、增量解码 | [02](02_process_topology_and_ipc.md) |
| prefill forward | ModelRunner + attention kernel | chunked prefill、FA3、cache hit 率 | [06](06_model_runner_and_cuda_graph.md), [07](07_attention_backends_and_kernels.md) |
| decode forward | ModelRunner | CUDA Graph、量化、batch 大小 | [06](06_model_runner_and_cuda_graph.md) |
| sampling | Sampler kernel | 融合采样 kernel | [07](07_attention_backends_and_kernels.md) |
| 调度开销（CPU） | Scheduler | Overlap 调度 | [05](05_overlap_and_latency_hiding.md) |
| detokenize | DetokenizerManager | 增量反分词 | [02](02_process_topology_and_ipc.md) |

> **关键洞察**：如果 GPU 前向只花 15ms，而 CPU 侧的调度 + 采样 + 反分词 + IPC
> 花了 8ms，那么你**永远无法把 TPOT 压到 15ms 以下**，除非做 overlap。
> 这就是第 05 章存在的理由。

---

## 4. 一个好用的"天花板"估算流程

在动手调参之前，先算三个数：

### 4.1 显存能放多少 KV

```
可用 KV 显存 = 总显存 × mem_fraction_static − 权重 − 激活预留
max_total_num_tokens = 可用 KV 显存 / (每 token 每层的 KV 字节 × 层数)
```

Qwen3-8B（36 层、GQA 8 KV heads、head_dim=128、FP16）：

```
每 token 每层 = 2 × 8 × 128 × 2 = 4096 Byte = 4 KB
每 token 全部层 = 4 KB × 36 = 144 KB
可用 KV 显存 40 GB → max_total_num_tokens ≈ 40e9 / 144e3 ≈ 277,000 tokens
```

SGLang 启动日志会直接打印这个数：

```
max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384,
max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

### 4.2 平均每请求占用

```
每请求 KV = (prompt_len + 生成长度) × 144 KB
```

4096 上下文 → 每请求 ~590 MB。277,000 / 4096 ≈ **67 个并发请求**。

### 4.3 理论吞吐上限

```
decode 吞吐上限 ≈ (显存带宽 / 权重字节) × batch × 接受率
                ≈ (3.35e12 / 16e9) × 67 ≈ 14,000 token/s
```

实测通常能到 50–70%。**如果你的实测远低于这个数，说明有问题**——去第 13 章
调参手册排查。

---

## 5. 三个容易搞错的量

### 5.1 `token usage` 不是越高越好

`token usage > 0.9` 是好状态，但**长期贴着 1.0 会触发 retraction**
（见第 04 章），反而增加抖动。日志里出现

```
KV cache pool is full. Retract requests. #retracted_reqs: 1,
  #new_token_ratio: 0.9998 -> 1.0000
```

说明需要提高 `--schedule-conservativeness`（如 1.3）。

### 5.2 `gen throughput` 包含了所有 running 请求

它是集群级吞吐，不是单请求速度。要评估用户体验要看 TTFT / TPOT 分位数（P99），
benchmark 用 `python -m sglang.bench_serving` 才有。

### 5.3 `cache_hit_rate` 依赖 workload

没有共享前缀的 workload（每个请求都是独特的长文本），命中率天然接近 0，
此时 RadixAttention 的开销（radix tree 查找 + 锁）是纯成本。
可以用 `--disable-radix-cache` 关掉对比一下。

---

## 6. 本章小结

- **算术强度决定优化方向**：I < I_ridge 就去减字节（量化、压缩 KV），I > I_ridge 就去提算力利用率。
- **Decode 的 I ≈ batch**，所以 batch 是 decode 吞吐的第一杠杆，而 batch 的上限是 KV 容量。
- **TPOT 的下限由 CPU 侧开销决定**，不做 overlap 就压不下去。
- 先算天花板（4.x 的三个公式），再看实测，差距就是你的优化空间。

---

下一页 → [02 进程拓扑与 IPC](02_process_topology_and_ipc.md)

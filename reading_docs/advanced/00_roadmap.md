# 进阶教程：SGLang 内部结构与性能优化

> 前置：读完 `../beginner/`。本教程**逐章对应**入门教程的每一节，把它放大到源码级。
> 每个文件的结构固定为：**问题 → 数据结构 → 关键代码 → 性能分析 → 调参建议**。
>
> 读法：不要顺序读。挑一个你现在最关心的性能问题，找到对应章节。

---

## 章节映射

| 进阶章 | 展开入门章 | 主题 | 核心矛盾 |
|---|---|---|---|
| [01](01_roofline_and_latency_model.md) | beginner 01 | Roofline 模型与 SGLang 的延迟分解 | 算术强度决定优化方向 |
| [02](02_process_topology_and_ipc.md) | beginner 02 | 进程拓扑、ZMQ IPC、异步桥接 | 并行度 vs IPC 开销 |
| [03](03_radix_cache_internals.md) | beginner 03 | Radix tree 实现、匹配/插入/驱逐 | 命中率 vs 内存/CPU 开销 |
| [04](04_scheduler_internals.md) | beginner 04 | 调度器内部：批组装、预算、retract | TTFT vs TPOT vs 吞吐 |
| [05](05_overlap_and_latency_hiding.md) | beginner 04 | Overlap 调度、FutureMap、TBO | CPU 下发时间必须被 GPU 藏住 |
| [06](06_model_runner_and_cuda_graph.md) | beginner 05 | ModelRunner 初始化、CUDA Graph 捕获/回放 | 启动开销 vs 灵活性 |
| [07](07_attention_backends_and_kernels.md) | beginner 05 | Attention/GEMM/Sampler 的 kernel 选择 | prefill 与 decode 需要不同 kernel |
| [08](08_kv_pool_and_allocators.md) | beginner 06 | 内存池布局、分页分配器、容量计算 | 碎片 vs 分配速度 |
| [09](09_hicache_and_multitier.md) | beginner 06 | HiCache 三级存储、写策略、预取 | 复用半径 vs 传输开销 |
| [10](10_parallelism_internals.md) | beginner 07 | TP/DP/EP/PP/PD 的通信原语与实现 | 通信量 vs 并行收益 |
| [11](11_moe_and_quantization.md) | beginner 08 | MoE 分组 GEMM、EP 分发、量化管线 | 精度 vs 带宽 |
| [12](12_speculative_decoding.md) | beginner 08 | 投机解码原理与 EAGLE/MTP 实现 | 接受率 vs 额外算力 |
| [13](13_tuning_playbook.md) | beginner 09 | 端到端调参方法论与故障诊断 | 所有权衡的最终汇总 |

---

## 阅读前你需要知道的代码约定

SGLang 的许多目录有自己的契约文档，**读代码前先读它们**：

| 目录 | 文档 |
|---|---|
| `python/sglang/srt/mem_cache/` | `README.md`（分层契约，强烈建议先读） |
| `python/sglang/srt/mem_cache/unified_cache/` | `components/README.md` |
| `docs/` | Mintlify 站点，`docs/docs/advanced_features/` 下有官方调参指南 |

另外注意 `python/sglang/srt/runtime_context.py`：SGLang 把全局配置收拢成
`RuntimeContext`（分层、命名空间化的配置袋），所以你在代码里会经常看到
`get_exec()`、`get_parallel()`、`get_model()`、`get_serving()` 这类访问器，
而不是直接读 `server_args.xxx`。看到它们时，去 `server_args.py` 找对应的字段即可。

---

## 性能分析工具箱

写优化建议之前，先确保你能测量。这些是教程里会反复引用的工具：

```bash
# 1. 端到端 benchmark（离线吞吐 / 在线延迟）
python -m sglang.bench_serving --backend sglang --model <model> \
    --num-prompts 1000 --request-rate 20

# 2. 单 batch 延迟分解
python -m sglang.bench_one_batch --model-path <model> --batch-size 32

# 3. Profiler（会在 SGLang 启动的机器上生成 trace）
#    在服务端开：curl http://localhost:30000/start_profile
#    跑一会请求后：curl http://localhost:30000/stop_profile
#    trace 输出到 --profile-dir 指定的目录（默认 /tmp）

# 4. 关键指标（Prometheus，--enable-metrics 时暴露）
#    sglang:num_running_reqs / num_queue_reqs
#    sglang:token_usage / kv_available_tokens / kv_evictable_tokens
#    sglang:gen_throughput / cache_hit_rate
#    sglang:num_retracted_reqs / new_token_ratio
#    sglang:spec_accept_length
#    定义见 python/sglang/srt/observability/metrics_collector.py:76 (SchedulerStats)
```

`SchedulerStats`（`observability/metrics_collector.py:76`）是理解 SGLang 运行状态的
字典：`token_usage`、`kv_evictable_tokens`、`new_token_ratio`、`spec_accept_length`
每一个都对应后面某一章的一个机制。

---

下一页 → [01 Roofline 模型与延迟分解](01_roofline_and_latency_model.md)

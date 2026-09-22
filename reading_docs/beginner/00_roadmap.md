# 入门教程：一晚上看懂 SGLang

> 目标读者：会写 Python、知道 Transformer 大概长什么样、但没读过任何推理引擎源码的人。
> 时长预算：**4–6 小时**。每章 20–40 分钟。
> 读完你能做到：画出 SGLang 的进程拓扑图、说清一个请求从 HTTP 进来到底经过了哪些对象、
> 知道每个性能开关大概在优化什么。

---

## 这套教程的读法

不要一上来就打开 `scheduler.py`——它有 6000 多行，第一次读只会迷路。

正确顺序是：**先建立"地图"，再进入"街区"，最后看"建筑内部"**。

| 阶段 | 章节 | 你获得的东西 |
|---|---|---|
| 地图 | 01 → 02 | 推理引擎要解决什么问题；SGLang 由哪几个进程组成 |
| 街区 | 03 → 06 | 四大组件：RadixAttention、调度器、ModelRunner、显存管理 |
| 外环 | 07 → 08 | 并行策略、投机解码、量化这些"可选加速件" |
| 动手 | 09 | 跑 benchmark、读日志、调参 |

进阶教程 `../advanced/` 是对应每一章的"放大镜"：同一个主题，深入到数据结构、
kernel 选择、以及每一处性能优化的取舍。

**建议**：入门教程读完后，挑你最感兴趣的两三章去读进阶版，不要一次全读。

---

## 章节列表

| # | 文件 | 一句话 | 核心源码入口 |
|---|------|--------|--------------|
| 00 | [本文件](00_roadmap.md) | 路线图 | — |
| 01 | [01_why_inference_engine.md](01_why_inference_engine.md) | 为什么需要推理引擎：从 `model.generate()` 到连续批处理 | `python/sglang/srt/entrypoints/engine.py` |
| 02 | [02_architecture_overview.md](02_architecture_overview.md) | 进程拓扑 + 一个请求的完整生命周期 | `entrypoints/`、`managers/` |
| 03 | [03_radix_attention.md](03_radix_attention.md) | RadixAttention：用一棵树自动复用前缀 KV | `mem_cache/radix_cache.py` |
| 04 | [04_scheduler.md](04_scheduler.md) | 调度器：continuous batching、chunked prefill、调度策略 | `managers/scheduler.py`、`managers/schedule_policy.py` |
| 05 | [05_model_runner.md](05_model_runner.md) | ModelRunner 与一次前向：ForwardBatch → 采样 | `model_executor/model_runner.py` |
| 06 | [06_memory_and_kv_pool.md](06_memory_and_kv_pool.md) | 显存怎么分：内存池、分配器、HiCache 分层 | `mem_cache/` |
| 07 | [07_parallelism.md](07_parallelism.md) | TP / DP / EP / PP / PD 全景图 | `distributed/parallel_state.py` |
| 08 | [08_speculative_and_advanced.md](08_speculative_and_advanced.md) | 投机解码、CUDA Graph、量化、多模态速览 | `speculative/`、`layers/quantization/` |
| 09 | [09_benchmark_and_tuning.md](09_benchmark_and_tuning.md) | 性能测量与调参手册（动手） | `benchmark/`、`docs/docs/advanced_features/hyperparameter_tuning.mdx` |
| 10 | [10_glossary_and_source_map.md](10_glossary_and_source_map.md) | 术语表 + 源码地图 + 下一步读什么 | — |

---

## 环境准备（15 分钟）

### 只读代码（推荐先这样）

```bash
cd <sglang-repo>
git log -1 --format='%H %ad %s'   # 记住你读的是哪个 commit
python -c "import sglang; print(sglang.__version__)"
```

### 想跑起来

```bash
# 启动一个服务
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 30000

# 另开一个终端，看日志
# 日志里这一行是后面所有章节的"体温计"：
# Decode batch. #running-req: 233, #token: 370959, token usage: 0.82,
#   cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317
```

### 会反复用到的三个文档

- `docs/docs/advanced_features/hyperparameter_tuning.mdx` —— 官方调参指南，短小精悍
- `python/sglang/srt/mem_cache/README.md` —— KV cache 目录的分层契约，写得非常清楚
- `python/sglang/srt/server_args.py` —— 所有 CLI 参数的**唯一真相**（971 行，值得通读一遍）

---

## 一句话预告每一章的核心矛盾

理解一个系统最快的方式，是理解它在**权衡什么**：

| 章 | 核心矛盾 |
|---|---|
| 01 | 吞吐 vs 延迟：batch 越大吞吐越高，但每个请求的延迟越差 |
| 02 | 进程拆分：拆得越细越能并行，但 IPC 开销越大 |
| 03 | 缓存复用 vs 显存容量：留得越多命中率越高，但能并发的请求越少 |
| 04 | prefill 优先 vs decode 优先：谁先跑决定了 TTFT 还是 TPOT 变差 |
| 05 | CPU 下发 vs GPU 计算：GPU 快但 CPU 喂不饱它 |
| 06 | 权重 / KV / 激活三者抢显存 |
| 07 | 通信 vs 计算：并行度越高单卡算得越少，但通信占比越大 |
| 08 | 用额外算力换更少的串行步数（投机解码）、用额外显存换更少的启动开销（CUDA Graph） |

带着这些问题去读，每一章的源码都会变得有目的。

---

下一页 → [01 为什么需要推理引擎](01_why_inference_engine.md)

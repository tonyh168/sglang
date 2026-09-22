# 10 · 术语表、源码地图与下一步

---

## 10.1 术语表

### 计算阶段

| 术语 | 英文 | 说明 |
|---|---|---|
| **Prefill / Extend** | 预填充 | 一次性处理整个 prompt，产出首个 token。compute-bound |
| **Decode** | 解码 | 逐 token 生成。memory-bound |
| **TTFT** | Time To First Token | 首 token 延迟，由 prefill + 排队决定 |
| **TPOT** | Time Per Output Token | 逐 token 延迟，由 decode 决定 |
| **ITL** | Inter-Token Latency | 等价于 TPOT |
| **Roofline** | — | `性能 = min(算力, 算术强度 × 带宽)` |
| **算术强度** | Arithmetic Intensity | `FLOPs / Bytes`。decode 时 ≈ batch size |

### 内存与缓存

| 术语 | 说明 |
|---|---|
| **KV Cache** | 注意力层的 Key/Value 缓存，随上下文和并发线性增长 |
| **RadixAttention** | SGLang 的前缀缓存，用 radix tree 索引 KV |
| **Radix Tree / 基数树** | 压缩前缀树。每条边存一段 token 序列 |
| **Prefix Cache** | 前缀缓存，RadixAttention 的通用叫法 |
| **Page / 页** | KV 分配的粒度（`--page-size`，默认 1 个 token） |
| **`req_to_token`** | 请求行号 → KV 物理槽位的间接表 |
| **`out_cache_loc`** | 本轮新 KV 要写入的槽位列表 |
| **`lock_ref`** | 树节点的引用计数，>0 表示不可驱逐 |
| **Evictable / Protected** | 可驱逐 / 受保护的 KV 槽位 |
| **HiCache** | 三级 KV 存储（GPU → Host → SSD/远端） |
| **`max_total_num_tokens`** | KV 池的总容量（token 数） |
| **`token_usage`** | KV 池占用率 |
| **Retraction** | KV 满时撤回部分 decode 请求，让它们重新排队 |
| **`new_token_ratio`** | 自适应估算"请求还会再生成多少 token"的比例 |

### 调度

| 术语 | 说明 |
|---|---|
| **Continuous Batching** | 每个 step 重组 batch：完成的退出，排队的补入 |
| **Chunked Prefill** | 把长 prefill 切块，与 decode 混合执行 |
| **Mixed Chunk** | `--enable-mixed-chunk`，prefill chunk 与 decode 同一批 |
| **Overlap Scheduling** | CPU 调度与 GPU 计算重叠 |
| **`FutureMap`** | overlap 用的设备端 relay buffer |
| **PrefillAdder** | 预算器：逐个试算请求能否加入本批 |
| **Retraction Policy** | 撤回顺序策略（`length` / `priority`） |
| **Prefill Delayer** | DP 下主动延迟 prefill，等所有 rank 对齐 |
| **TBO** | Two Batch Overlap，两个子批的计算与通信重叠 |

### 并行

| 术语 | 说明 |
|---|---|
| **TP** | Tensor Parallelism，切权重矩阵 |
| **PP** | Pipeline Parallelism，按层切 |
| **DP** | Data Parallelism，切请求 |
| **EP** | Expert Parallelism，切 MoE 专家 |
| **DP Attention** | attention 用 DP、FFN 用 TP/EP 的混合方案 |
| **PD 分离** | Prefill 和 Decode 部署在不同节点 |
| **All-reduce / All-gather / Reduce-scatter** | 集合通信原语 |
| **All-to-all** | EP 的 token 分发原语 |
| **`GroupCoordinator`** | SGLang 对 ProcessGroup 的封装 |
| **`attn_tp_size` / `moe_ep_size`** | 派生并行度（attention 和 MoE 可以不同） |

### 执行

| 术语 | 说明 |
|---|---|
| **CUDA Graph** | 把整个前向录成一个图，replay 只发一次 launch |
| **`capture_bs` / bucket** | 预捕获的 batch size 列表 |
| **Padding** | 把真实 batch pad 到最近的 bucket |
| **`ForwardBatch`** | 喂给模型的张量包 |
| **`ForwardMode`** | EXTEND / DECODE / IDLE / TARGET_VERIFY / DRAFT_EXTEND_V2 |
| **Piecewise CUDA Graph** | 在 attention 处切开的 torch.compile 图 |
| **`split_op`** | torch.compile 的切分边界（= attention） |

### 模型相关

| 术语 | 说明 |
|---|---|
| **MoE** | Mixture of Experts |
| **Grouped GEMM** | 按专家分组的 GEMM |
| **MLA** | Multi-head Latent Attention（DeepSeek），压缩 KV |
| **GQA** | Grouped Query Attention，多个 Q 头共享 KV 头 |
| **MTP** | Multi-Token Prediction，DeepSeek 的投机解码 |
| **EAGLE** | 一种投机解码算法，draft 复用 target 的 hidden states |
| **Speculative Decoding** | 投机解码 |
| **`spec_accept_length`** | 平均每步接受的 token 数 |

---

## 10.2 源码地图

### 按目录

```
python/sglang/
├── launch_server.py                    # CLI 入口
├── bench_serving.py                    # 已废弃，改指 benchmark/serving.py
├── benchmark/                          # benchmark 实现
│   ├── serving.py                      # 在线服务基准（TTFT/TPOT）
│   ├── offline_throughput.py           # 离线吞吐
│   └── one_batch.py                    # 单 batch 延迟
└── srt/                                # "SGLang Runtime"
    ├── server_args.py                  # ★ 所有 CLI 参数的唯一真相
    ├── arg_groups/fields/*.py          # 参数分组（schedule/memory/parallel/spec/exec_...）
    ├── arg_groups/choices.py           # ★ 所有枚举选项列表
    ├── runtime_context.py              # 全局配置袋（get_exec/get_parallel/...）
    ├── environ.py                      # SGLANG_* 环境变量
    │
    ├── entrypoints/
    │   ├── engine.py                   # ★ _launch_subprocesses：进程编排
    │   ├── http_server.py              # FastAPI 路由
    │   └── EngineBase.py
    │
    ├── managers/
    │   ├── tokenizer_manager.py        # ★ 分词 + rid→asyncio.Event
    │   ├── detokenizer_manager.py      # ★ 独立进程，增量反分词
    │   ├── scheduler.py                # ★★ 6126 行，调度器核心
    │   ├── schedule_policy.py          # ★ 排序 + 预算（PrefillAdder）
    │   ├── schedule_batch.py           # ★ Req / Batch 数据结构
    │   ├── io_struct.py                # ★ 所有 IPC 消息结构
    │   ├── tp_worker.py                # TpModelWorker
    │   ├── overlap_utils.py            # ★ FutureMap
    │   ├── prefill_delayer.py
    │   ├── min_free_slots_delayer.py
    │   └── scheduler_components/       # 调度器的分解模块（20 个文件）
    │
    ├── mem_cache/
    │   ├── README.md                   # ★★ 分层契约，先读这个
    │   ├── radix_cache.py              # ★★ RadixAttention 实现
    │   ├── hiradix_cache.py            # HiCache
    │   ├── memory_pool.py              # ★★ 5904 行，KV 池
    │   ├── allocator/                  # 分配器（paged/token/page_interleave）
    │   ├── evict_policy.py             # 驱逐策略
    │   ├── pool_host/                  # L2 host 池
    │   └── storage/                    # L3 后端
    │
    ├── model_executor/
    │   ├── model_runner.py             # ★★ 2314 行，frozen core
    │   ├── forward_batch_info.py       # ★ ForwardBatch / ForwardMode
    │   ├── runner/                     # cuda graph runner 实现
    │   ├── runner_backend/             # full/breakable/tc_piecewise
    │   └── model_runner_components/    # 分解模块
    │
    ├── layers/
    │   ├── attention/                  # ★★ 40+ 个 attention 后端
    │   ├── moe/                        # ★ MoE kernel 与 dispatcher
    │   ├── quantization/               # ★ 28 种量化方法
    │   ├── layernorm.py                # fused_add_rmsnorm
    │   ├── activation.py               # silu_and_mul
    │   ├── sampler.py                  # ★ 融合采样
    │   ├── linear.py                   # TP 线性层
    │   ├── communicator.py             # ★ TP 通信编排
    │   └── dp_attention.py             # DP attention
    │
    ├── distributed/
    │   └── parallel_state.py           # ★★ GroupCoordinator
    │
    ├── speculative/                    # ★ 投机解码（40+ 文件）
    │   └── spec_info.py                # SpeculativeAlgorithm 枚举
    │
    ├── disaggregation/                 # PD 分离
    │   ├── prefill.py / decode.py      # 两侧的生命周期
    │   └── nixl/ mooncake/ mori/       # 传输后端
    │
    ├── compilation/                    # torch.compile 集成
    ├── observability/                  # metrics / trace
    ├── model_loader/                   # 权重加载
    ├── models/                         # 各模型定义
    └── configs/model_config.py         # 模型配置
```

### 按"我想搞懂 X"索引

| 想搞懂 | 读这个 |
|---|---|
| 所有 CLI 参数 | `server_args.py` + `arg_groups/fields/*.py` |
| 一个请求怎么走完全程 | `entrypoints/engine.py:1051` → 第 02 章的表 |
| 前缀缓存怎么实现 | `mem_cache/radix_cache.py` |
| 调度策略怎么排序 | `managers/schedule_policy.py:259` |
| 批怎么组 | `managers/scheduler.py:3622` |
| chunked prefill | `scheduler.py:3790` + `schedule_policy.py:1022` |
| retraction | `schedule_batch.py:3248` |
| KV 池怎么分配 | `mem_cache/allocator/paged.py` |
| 显存怎么算 | `mem_cache/kv_cache_configurator.py:2214` |
| CUDA Graph 怎么捕获 | `model_executor/runner/base_cuda_graph_runner.py:67,157` |
| attention 后端怎么选 | `layers/attention/attention_registry.py` |
| MoE 怎么跑 | `layers/moe/moe_runner/runner.py` |
| 量化怎么接入 | `layers/quantization/base_config.py` + `fp8.py` |
| TP 通信怎么编排 | `layers/communicator.py` + `distributed/parallel_state.py` |
| 投机解码怎么验证 | `speculative/eagle_worker_v2.py` |
| PD 分离怎么传 KV | `disaggregation/decode.py` |

### 四个"先读文档再读代码"的地方

1. `python/sglang/srt/mem_cache/README.md` —— 分层契约
2. `docs/docs/advanced_features/hyperparameter_tuning.mdx` —— 调参
3. `python/sglang/srt/server_args.py` 顶部的注释 —— 文件组织约定
4. 各 `__init__.py` 的模块文档字符串 —— 往往有设计意图说明

---

## 10.3 下一步读什么

### 如果你想做性能优化

1. 先跑通 benchmark，建立**基线**
2. 用 `start_profile` / `stop_profile` 拿一个 trace
3. 按第 09 章的顺序调参（一次一个）
4. 深入到进阶教程对应的章节

### 如果你想贡献代码

1. 读 `CONTRIBUTING.md` 和 `.claude/rules/modify-component-must-read.md`
2. **注意 frozen core file**：`model_runner.py` 修改前必须先读
   `.claude/skills/large-class-style/SKILL.md`
3. 改配置项前读 `env-var-conventions` skill
4. 改投机解码相关代码前读 `speculative-naming` skill

### 如果你想读更多系统

| 系统 | 差异 |
|---|---|
| **vLLM** | PagedAttention 的原创者；V1 架构更接近 SGLang |
| **TensorRT-LLM** | NVIDIA 官方，kernel 最优化，但灵活性差 |
| **LMDeploy** | 国内，TurboMind 引擎 |

**横向对比的价值点**：看同一个问题（如前缀缓存）各家怎么用不同的数据结构解决。

---

## 10.4 十句话总结这套教程

1. **Prefill 是 compute-bound，decode 是 memory-bound**；decode 的算术强度 ≈ batch。
2. **SGLang 有四个进程**：HTTP+Tokenizer、Scheduler（GPU）、Detokenizer，
   拆分的理由都是"让 CPU 工作与 GPU 计算并行"。
3. **RadixAttention 用一棵 radix tree 索引 KV 槽位**，让前缀复用变成默认行为，
   零 API 成本。
4. **`lock_ref` 决定 KV 的可驱逐性**；`available + evictable` 才是能接新请求的余量。
5. **Continuous batching = 每轮 `filter_batch` + `merge_batch`**，
   加上 chunked prefill 解决长 prefill 打断 decode 的问题。
6. **Retraction + 自适应 `new_token_ratio`** 是对"用户声明的 max_new_tokens 远大于实际"
   这一现实的工程妥协。
7. **Overlap 调度的核心是 `FutureMap`**——用设备内存做跨迭代通信，避免任何 sync。
8. **CUDA Graph 对 decode 是必需品**：600+ 次 kernel 启动变成 1 次；
   代价是固定形状，所以有 bucket 和 padding。
9. **融合是 kernel 层的主题**：all-reduce+RMSNorm、silu_and_mul、
   top-k/top-p+采样，每个融合都省掉一次 HBM 往返。
10. **调参顺序**：先拉满 KV 池 → 再调调度保守度 → 再按负载选优化 →
    最后才是并行策略。

---

## 10.5 回到目录

- 入门教程：[00 路线图](00_roadmap.md)
- 进阶教程：[../advanced/00_roadmap.md](../advanced/00_roadmap.md)

**建议的下一步**：挑一个你现在最关心的性能问题，去读对应的进阶章节，
然后用 `bench_serving` 验证你的假设。

祝读码愉快。

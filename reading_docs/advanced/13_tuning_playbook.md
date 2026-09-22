# 进阶 13 · 调参方法论与故障诊断

> 对应入门 [09 性能测量与调参](../beginner/09_benchmark_and_tuning.md)。
> 这一章把前面 12 章的所有权衡收拢成一套可执行的方法。

---

## 1. 方法论：先算天花板，再找差距

**不要上来就调参数。** 先做三件事：

### 步骤 1：算显存天花板

```python
# 每 token 每层的 KV 字节
if MHA:
    bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
elif MLA:
    bytes_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) * dtype_bytes

# 全部层
cell_size = bytes_per_token_per_layer * num_layers

# 可用 KV 显存
available = total_vram * mem_fraction_static - model_weights - activation_reserve

# 最大 token 数
max_total_num_tokens = available // cell_size
```

**验证方法**：和启动日志里的 `max_total_num_tokens` 对比。
如果差很多，说明 `activation_reserve` 估错或权重加载有临时占用（第 08 章）。

### 步骤 2：算并发天花板

```python
每请求 KV = (prompt_len + expected_output_len) * cell_size
max_concurrency = max_total_num_tokens / 每请求 KV
```

### 步骤 3：算吞吐天花板

```python
decode 吞吐上限 ≈ (显存带宽 / 权重字节) × max_concurrency
```

**实测通常能到 50–70%。低于这个比例就有优化空间。**

### 步骤 4：对比

```bash
python -m sglang.bench_serving --backend sglang --model <model> \
    --dataset-name random --random-input-len 2048 --random-output-len 512 \
    --num-prompts 500 --request-rate 20
```

看输出里的 `gen throughput` 和上面算的天花板比值。

**如果实测 = 天花板的 60%+，你的配置基本没问题，剩下的是 workload 特性。**
**如果只有 30%，按第 2 节的诊断流程走。**

---

## 2. 诊断流程：从症状到根因

### 症状 A：吞吐低，GPU 利用率低

```
第一步：看 token_usage
├── token_usage < 0.9 且 #queue-req > 0
│   → 调度太保守
│   → --schedule-conservativeness 0.3
│
├── token_usage > 0.95 且频繁 retract
│   → KV 真的不够
│   → 第 4 节的扩容决策树
│
└── token_usage 正常但 GPU 空转
    → 看 trace，区分是 CPU 还是通信
    ├── CPU：有 gap，GPU 空闲，CPU 在跑
    │   → 检查 overlap 是否开启（默认开）
    │   → 检查是否触发了 is_disable_overlap_for_batch
    │   → 检查 process_batch_result 里的 radix tree 操作是否太重
    └── 通信：NCCL kernel 占比 > 30%
        → 减 TP，或开 TBO
```

### 症状 B：TTFT 高，TPOT 正常

```
TTFT = 排队 + 分词 + prefill + 采样 + 反分词

├── 排队占大头
│   → 并发不够（扩容）或调度策略不好
│   → --schedule-policy shortest-prefill-first
│
├── prefill 占大头
│   ├── cache_hit_rate 低 → 前缀没复用
│   │   → 检查 workload 是否真的有共享前缀
│   │   → --schedule-policy lpm
│   │   → 开 HiCache
│   └── cache_hit_rate 高但 prefill 还是慢
│       → --chunked-prefill-size 太大？试试调小
│       → attention backend 是否选对（fa3 用于 prefill）
│
└── 分词/反分词占大头
    → 请求特别短（< 50 token）时容易发生
    → --tokenizer-worker-num 提高分词并行度
```

### 症状 C：TPOT 抖动大（P99 / P50 比值高）

```
抖动来源：prefill 打断了 decode

├── 默认行为：prefill 批和 decode 批不混
│   → --enable-mixed-chunk（让它们混在一个 forward 里）
│
├── 调度策略让长 prefill 频繁进来
│   → --prefill-decode-interval N（prefill 后强制跑 N 轮 decode）
│
├── chunked prefill 的 chunk 太大
│   → 调小 --chunked-prefill-size
│
└── DP attention 下的碎片化 prefill
    → --enable-prefill-delayer
```

### 症状 D：GPU 显存 OOM

```
区分 OOM 发生在 prefill 还是 decode
（看 traceback 的位置）

├── prefill OOM（激活太大）
│   → --chunked-prefill-size 4096 或 2048
│   → 降 --mem-fraction-static 到 0.8 / 0.7
│
└── decode OOM（KV 不够）
    → 降 --max-running-requests
    → KV 量化 --kv-cache-dtype fp8_e4m3
    → 降 --cuda-graph-max-bs-decode（CUDA Graph 的填充也吃显存）
```

### 症状 E：频繁 retract

```
KV cache pool is full. Retract requests. #retracted_reqs: 1,
  #new_token_ratio: 0.9998 -> 1.0000
```

```
├── 每分钟 1 次左右 → 正常，不用管
├── 频繁出现
│   → --schedule-conservativeness 1.3
│   → 检查 max_new_tokens 声明是否过大
│   → 扩容（KV 量化 / HiCache / 加卡）
└── 大量同时 retract（#retracted_reqs 一次几十个）
    → 说明并发远超容量，是配置问题不是调参问题
```

### 症状 F：投机解码没有收益

```
看 spec_accept_length
├── < 2 → 负收益
│   ├── draft 模型不匹配 → 换 draft
│   ├── batch 太大（接近 compute-bound）→ 大 batch 时关掉投机
│   └── 输出太发散 → NGRAM 或关掉
└── > 3 但吞吐没涨
    → 看 draft 前向的成本占比（trace 里 draft_extend 的时间）
    → --speculative-num-steps 调小
    → --speculative-num-draft-tokens 调小
```

---

## 3. 用 trace 定位问题

```bash
python -m sglang.launch_server --model-path <model> --profile-dir /tmp/prof
# 另开终端
curl http://localhost:30000/start_profile
# 跑请求
curl http://localhost:30000/stop_profile
```

### 看 trace 的三步法

**第一步：找大块空白**

```
如果有连续 >1ms 的 GPU 空闲：
  → CPU 侧太慢，或通信等待
  → 看空白期间 CPU 在跑什么
```

**第二步：分类 GPU 时间**

| kernel 类别 | 占比健康值 | 超标怎么办 |
|---|---|---|
| GEMM（cutlass/deep_gemm/triton） | 40–60% | 已经是瓶颈，考虑量化 |
| attention（flashinfer/FA3） | 15–30% | 检查后端选择 |
| NCCL / all-reduce | < 20% | 减 TP / 开 TBO |
| elementwise（norm/activation） | < 10% | 检查融合是否生效 |
| 空白 | < 10% | overlap 问题 |

**第三步：看 stream 的并行度**

```
有 tbo / sbo 时，应该能看到 compute 和 NCCL 在两条 stream 上重叠
如果没有重叠 → TBO 没生效，或者 op 序列解析失败
```

### 相关的辅助工具

```bash
# 单 batch 延迟分解（不涉及调度）
python -m sglang.bench_one_batch --model-path <model> --batch-size 32

# 离线吞吐
python -m sglang.bench_offline_throughput --model-path <model> --num-prompts 1000
```

SGLang 还带了 `llm-torch-profiler-analysis` 的能力（skill），
能直接对 trace.json(.gz) 生成 kernel / overlap / fuse 三张表。

---

## 4. 扩容决策树（完整版）

```
KV 不够用
│
├─ 1. 显存还有余量？（available_gpu_mem > 8 GB）
│     → 提高 --mem-fraction-static（0.01 一档，直到 available 到 5–8 GB）
│     → 成本：0
│
├─ 2. CUDA Graph 占了太多？
│     → 降 --cuda-graph-max-bs-decode
│     → 成本：大 batch 时启动开销回升
│
├─ 3. 能接受精度损失？
│     → --kv-cache-dtype fp8_e4m3        KV 减半，成本最低
│     → 成本：少量精度损失（通常 <1% 的指标下降）
│
├─ 4. 模型量化还有空间？
│     → --quantization fp8              权重减半，顺带 decode 加速
│     → 成本：需要 FP8 权重或在线量化 + 精度损失
│
├─ 5. 有 CPU 内存？
│     → --enable-hierarchical-cache
│     → 成本：H2D 传输 + host 管理开销；只优化 TTFT
│
├─ 6. 有 SSD / 远端存储？
│     → --hicache-storage-backend file / nixl / mooncake
│     → 成本：传输延迟；适合跨会话复用
│
├─ 7. 混合 SWA 模型？
│     → --swa-full-tokens-ratio 调低
│     → 成本：SWA 层的窗口外信息丢失（不影响正确性）
│
└─ 8. 都不行
      → 降 --max-running-requests（牺牲并发换稳定）
      → 或加卡（DP / TP）
```

---

## 5. 一张完整的能力-成本表

| 优化 | 提什么 | 成本 | 什么时候用 |
|---|---|---|---|
| `--mem-fraction-static` ↑ | 并发 | 0（但过高会 OOM） | 永远先做 |
| `--schedule-conservativeness` ↓ | 并发 | retract 风险 | token_usage 低 + 队列有货 |
| `--kv-cache-dtype fp8` | 并发 ×2 | 精度 | 显存紧张 |
| `--quantization fp8` | 显存 + decode 速度 | 精度 | 显存紧张 + decode 主导 |
| `--enable-mixed-chunk` | TPOT 稳定性 | prefill 效率略降 | TTFT/TPOT 抖动大 |
| `--chunked-prefill-size` ↓ | TTFT、省激活显存 | prefill 效率 | 长 prompt + 低延迟要求 |
| `--schedule-policy lpm` | 命中率 → 吞吐 | CPU 排序开销 | 大量共享前缀 + 队列 <128 |
| `--schedule-policy shortest-prefill-first` | 平均 TTFT | 长请求饿死风险（有反饥饿机制） | 延迟敏感 |
| `--cuda-graph-max-bs-decode` ↑ | 大 batch 吞吐 | 显存 | 大 TP + 大 batch |
| `--enable-hierarchical-cache` | TTFT | host 内存 + H2D | 跨请求前缀复用 |
| `--enable-dp-attention` | MoE 模型的吞吐 | 每步全局同步 | DeepSeek 类 |
| `--ep-size` | MoE 显存 + 吞吐 | all-to-all | MoE + 多卡 |
| `--enable-two-batch-overlap` | 通信-计算重叠 | batch 对齐要求变严 | MoE + EP，NCCL 占比高 |
| `--speculative-algorithm` | decode 吞吐 | draft 显存 + 算力 | 小 batch + 输出可预测 |
| `--enable-torch-compile` | 小模型小 batch | 编译时间 | 小模型 + 小 batch |
| `--enable-prefill-delayer` | DP 下的 GPU 利用率 | 最坏 TTFT（有上界） | DP attention + 碎片化 prefill |
| `--disable-radix-cache` | **减少** CPU 开销 | 失去前缀复用 | 前缀完全不共享 |

---

## 6. 三个"反直觉"的结论

### ① 命中率高不等于吞吐高

`cache_hit_rate` 只衡量"多少 token 不需要重算"。
但决定吞吐的是**可驱逐容量**（`kv_evictable_tokens`）。

**极端情况**：一个巨型 system prompt 被 100% 命中，但占满 KV 池，
并发只有 5。命中率 99%，吞吐极低。

**正确的看板**：`cache_hit_rate` + `kv_evictable_tokens` + `token_usage` 三者一起看。

### ② token_usage 不是越高越好

长期贴着 1.0 会持续触发 retraction，增加抖动。
**0.85–0.95 是甜蜜区。**

### ③ 大 batch 下投机解码、torch.compile 都可能变成负收益

- 投机：target 已经 compute-bound，验证 k+1 个 token 成本线性增长
- torch.compile：手写 kernel 在大形状下更优

**所有优化都有它的适用边界。** 看到一个优化在某个场景有效，
先问"它的边界在哪"。

---

## 7. 性能文档的阅读顺序

如果你要深入某个方向，按这个顺序读：

| 方向 | 顺序 |
|---|---|
| **总体调参** | `docs/docs/advanced_features/hyperparameter_tuning.mdx` → 本教程第 09 章 → 本文件 |
| **KV / 缓存** | `mem_cache/README.md` → 进阶 03 → 进阶 08 → 进阶 09 |
| **调度** | 进阶 04 → 进阶 05 → 源码 |
| **Kernel** | 进阶 07 → `layers/attention/attention_registry.py` → 具体后端源码 |
| **并行** | 进阶 10 → `distributed/parallel_state.py` → `layers/communicator.py` |
| **MoE** | 进阶 11 → `layers/moe/moe_runner/runner.py` → 具体 runner core |
| **投机** | 进阶 12 → `speculative/eagle_worker_v2.py` |
| **PD 分离** | 进阶 10 的 §5 → `disaggregation/prefill.py` / `decode.py` |

---

## 8. 一个完整的实战案例

**场景**：DeepSeek-V3 类 MoE 模型，8×H100，长 prompt（4K）+ 中等输出（1K），
要求高吞吐。

### 基线

```bash
python -m sglang.launch_server --model-path <dsv3> \
    --tp-size 8 --port 30000
```

观察：`token usage 0.55`，`#queue-req 800`，`gen throughput 2800 tok/s`

### 第 1 轮：放开调度

```bash
... --schedule-conservativeness 0.3
```

`token usage 0.78`，throughput 3600

### 第 2 轮：MoE 用 EP 而不是 TP

```bash
... --ep-size 8 --tp-size 8
```

**为什么**：MoE 权重巨大，TP 会频繁 all-reduce；EP 只发 token。
throughput 5200，NCCL 占比从 34% 降到 18%

### 第 3 轮：开 DP attention

```bash
... --enable-dp-attention --dp-size 8
```

**为什么**：attention 部分完全无通信，MoE 部分走 EP。
throughput 6400

### 第 4 轮：开 TBO

```bash
... --enable-two-batch-overlap
```

**为什么**：MoE 的 all-to-all 与 expert GEMM 重叠。
throughput 7100，NCCL 占比 12%

### 第 5 轮：加大 KV 池

```bash
... --mem-fraction-static 0.88 --cuda-graph-max-bs-decode 512
```

`available_gpu_mem 6.1 GB`，throughput 7600

### 第 6 轮：前缀缓存 + 调度策略

```bash
... --schedule-policy lpm
```

（假设 workload 有共享前缀）
`cache_hit_rate 0.42`，throughput 8900

### 最终验证

```bash
python -m sglang.bench_serving --backend sglang --model <dsv3> \
    --dataset-name random --random-input-len 4096 --random-output-len 1024 \
    --num-prompts 500 --request-rate 20
# 检查 TTFT / TPOT 的 P99 是否可接受
```

**每一步只改一个参数，都记录 token_usage / throughput / NCCL 占比。**
任何一步如果指标劣化，立刻回退。

---

## 9. 最后：如何继续深入

1. **挑一个你真正遇到的问题**，用 §2 的流程定位到具体章节
2. **读该章节引用的源码**，特别是带注释的分支
3. **用 trace 验证你的假设**——trace 不会骗人
4. **一次只改一个参数**，记录前后对比

SGLang 源码里最有价值的东西不是算法，而是**每一处优化旁边的注释**——
它们通常会说清"为什么这样做"和"什么时候这样做不对"。

---

## 10. 自检

1. 算天花板的三个公式分别是什么？
2. `token_usage` 低但 `#queue-req` 高，改什么？
3. `token_usage` 高且频繁 retract，按成本从低到高的扩容手段？
4. TPOT P99 抖动大，有哪几种可能的原因？
5. trace 里 NCCL 占比 34%，你会做什么？
6. 为什么命中率高不等于吞吐高？
7. 大 batch 下哪两个常见优化会变成负收益？

---

回到目录 → [进阶教程路线图](00_roadmap.md) | [入门教程路线图](../beginner/00_roadmap.md)

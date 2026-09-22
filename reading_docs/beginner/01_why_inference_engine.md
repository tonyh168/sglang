# 01 · 为什么需要推理引擎

> 本章不读源码。目的是把"为什么 SGLang 长这样"这件事讲清楚。
> 后面每一章的优化手段，都能在这一章找到它的动机。

---

## 1.1 两种完全不同的计算：Prefill 与 Decode

一个大模型生成一段文本，分成两个阶段：

```
Prompt: "请介绍一下北京" (5 tokens)
   ↓
┌─────────────────────────────────────────┐
│ PREFILL（预填充）                        │
│ 一次性把 5 个 token 全部送进模型          │
│ 5 次矩阵乘（每次是 [5, hidden] × W）      │
│ 产出：5 组 KV，以及第 1 个生成 token      │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│ DECODE（解码）              ← 重复 N 次  │
│ 每次只送 1 个 token                      │
│ 1 次矩阵乘（[1, hidden] × W）            │
│ 产出：1 组新 KV，1 个新 token            │
└─────────────────────────────────────────┘
```

这两个阶段的硬件特性**完全相反**：

| | Prefill | Decode |
|---|---|---|
| 每次处理的 token 数 | 几百 ~ 几万 | 1（每请求） |
| 矩阵乘形状 | `[T, K] × [K, N]`，T 很大 | `[1, K] × [K, N]`，M 维=1 |
| 瓶颈 | **算力**（FLOPs bound） | **显存带宽**（memory bound） |
| GPU 利用率 | 高，容易打满 tensor core | 极低，绝大部分时间在等权重搬运 |
| 决定指标 | **TTFT**（首 token 延迟） | **TPOT / ITL**（逐 token 延迟） |

### 关键的数字直觉：算术强度

看一次 decode 的矩阵乘：权重是 `[K=4096, N=4096]`，FP16 占 `4096×4096×2 = 32 MB`。
输入只有 1 行，需要 `2×4096×4096 = 33.5 MFLOP`。

```
算术强度 = 33.5e6 FLOP / 32e6 Byte ≈ 1 FLOP/Byte
```

而 H100 的比率大约是 `989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/Byte`。

**差了 300 倍。** 意味着 decode 时 tensor core 有 99.7% 的时间在空转，GPU 只是在
一节一节地把 32MB 权重从 HBM 搬进 SM，然后算上微不足道的一点点。

> 这就是为什么 decode 阶段**加 batch 几乎免费**：batch=1 和 batch=32 搬的权重
> 是一样多的（32MB），但后者多算了 32 倍的活。这就是"批处理"的全部理由。

---

## 1.2 朴素实现错在哪：`model.generate()` 的三种浪费

假设你用 HuggingFace 的 `model.generate()` 服务 100 个并发用户，最朴素的写法是
**每个请求一个线程/进程**：

```python
# 伪代码：最朴素的服务
def handle(request):
    for token in model.generate(**request):   # 内部是 while 循环，逐 token 生成
        yield token
```

三种浪费：

### 浪费 ①：batch = 1 的 decode

每个请求都在单独做 `[1, K] × [K, N]` 的矩阵乘。100 个请求 = 把同一份权重从 HBM
读了 100 遍。显存带宽被打满，但有效算力只有 0.3%。

**解法：Continuous Batching（连续批处理）**

把 100 个请求的当前 token 拼成一个 batch，一次前向处理：

```
朴素：  前向 × 100 次，每次 [1, 4096]
连续：  前向 × 1 次，形状 [100, 4096]
```

权重只读一遍，算力利用率提升 ~100 倍。

### 浪费 ②：等最慢的那个

如果按固定 batch 收集（"凑满 32 个请求才发车"），那么：

- 短请求早就生成完了，却要等长请求结束才能让出位置 → GPU 空转
- 新来的请求要等下一批 → 排队延迟

**解法：Continuous Batching 的"连续"二字**

批不是静态的。**每个 decode step 结束后，就重新组装一次 batch**：完成的请求
立刻退出、释放 KV，排队的请求立刻补进来。

```
step 1: [A B C D]        → D 生成 EOS，结束
step 2: [A B C E]        → E 从队列补进来
step 3: [A B E F]        → C 结束，F 补进来
...
```

这就是 SGLang 里 `running_batch` 与 `waiting_queue` 的关系（见第 04 章）。

### 浪费 ③：重复计算同一段前缀

这是**最容易被忽略、但收益最夸张**的一条。

多个请求常常共享前缀：
- System prompt（"你是一个专业的助手……" 500 tokens）
- Few-shot 示例（1000 tokens）
- 多轮对话的历史（每一轮都包含前面所有轮）
- 同一份长文档被问多个问题（RAG）

朴素实现里，每个请求都要**从头算一遍**这些前缀的 KV。500 tokens 的 system prompt
在 1000 个请求上就是 50 万 token 的重复 prefill。

**解法：前缀缓存（Prefix Caching）——SGLang 的 RadixAttention**

把已算好的 KV 按 token 序列存起来，新请求先查表：

```
请求 1: [S1 S2 S3 S4 S5 | A1 A2 A3]      → 全算，存下 [S1..S5] 和 [A1..A3]
请求 2: [S1 S2 S3 S4 S5 | B1 B2]         → 命中 S1..S5，只算 B1 B2
请求 3: [S1 S2 S3 S4 S5 | A1 A2 A3 C1]   → 命中 S..S5 和 A1..A3，只算 C1
```

**这就是为什么 SGLang 叫 "Structured Generation Language" 却以 RadixAttention 闻名**
—— 它把前缀复用做成了引擎的默认行为，而不是可选项。

而且注意：**前缀缓存还要求调度器配合**。如果调度器把请求 3 排到请求 1 前面，
那请求 3 就命中不了。所以有了 cache-aware scheduling（LPM / DFS-weight，见第 04 章）。

---

## 1.3 Chunked Prefill：解决 prefill 打断 decode 的问题

现在有了连续批处理，但还有一个矛盾：

**Prefill 是长任务，decode 是短任务。**

一个 32K 长 prompt 的 prefill 可能要 800ms，这期间：

- 所有正在 decode 的请求**停住不动**，卡了 800ms
- 用户体验：本来每 20ms 出一个字的对话，突然卡了 0.8 秒

```
时间轴（无 chunked prefill）：
decode: ███ ███ ███ [—————— 800ms 的 prefill ——————] ███ ███
                    ↑ 所有 decode 请求 TPOT 抖动 800ms

时间轴（有 chunked prefill）：
        chunk1  chunk2  chunk3  chunk4
decode: ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███
        每个 step 里：少量 prefill token + 正常 decode token 混在一起跑
```

**Chunked Prefill**：把一个长 prefill 切成 8192（默认 `--chunked-prefill-size`）大小
的块，每块和当前的 decode batch **合并成一次前向**。

代价：
- 切块后 prefill 本身的效率略降（短矩阵乘）
- 但 TPOT 抖动从 800ms 变成接近 0

所以 SGLang 默认在开启 RadixAttention 时**自动启用** chunked prefill。
这是延迟/吞吐权衡的经典案例。

---

## 1.4 那显存呢？KV Cache 才是真正的稀缺资源

模型权重是固定的：8B 模型 FP16 = 16GB。
但 **KV Cache 随并发数和上下文长度线性增长**：

```
KV bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes

Qwen3-8B 为例：2 × 36 层 × 8 KV头 × 128 dim × 2 字节 = 每 token 每序列 147 KB
并发 100 个请求、平均 4096 上下文 → 100 × 4096 × 147KB ≈ 60 GB
```

**比权重还大 4 倍。** 这就是为什么：

- 第 06 章要讲内存池、分配器、分页
- 第 03 章要讲前缀缓存怎么省 KV
- 第 07 章要讲 TP 把 KV 切到多张卡上
- HiCache 要把 KV 放到 CPU 内存甚至 SSD 上（三级存储）

**你能跑多少并发，几乎完全由 KV cache 容量决定。**

---

## 1.5 把上面的东西合起来：一个推理引擎的必答题

| 问题 | SGLang 的答案 | 章节 |
|---|---|---|
| 怎么提高 GPU 利用率？ | Continuous Batching（每个 step 重组 batch） | [04](04_scheduler.md) |
| 怎么省重复 prefill？ | RadixAttention 前缀缓存（radix tree） | [03](03_radix_attention.md) |
| 怎么让长 prompt 不卡住别人？ | Chunked Prefill（prefill 与 decode 混批） | [04](04_scheduler.md) |
| 怎么让 CPU 不拖 GPU 后腿？ | Overlap Scheduling（CPU 调度与 GPU 计算重叠） | [04](04_scheduler.md) |
| 怎么减少 kernel 启动开销？ | CUDA Graph（把整个 decode 前向录成一个图） | [05](05_model_runner.md) |
| 怎么在有限显存下提高并发？ | KV 池 + 分页分配器 + KV 量化 | [06](06_memory_and_kv_pool.md) |
| KV 不够放怎么办？ | HiCache（GPU → Host → SSD 三级） | [06](06_memory_and_kv_pool.md) |
| 单卡放不下模型怎么办？ | TP / PP / EP | [07](07_parallelism.md) |
| 怎么让 decode 一次出多个 token？ | 投机解码（EAGLE / MTP / NGram） | [08](08_speculative_and_advanced.md) |

---

## 1.6 本章自检

读完应该能回答：

1. 为什么 decode 阶段 batch 从 1 加到 32，延迟几乎不变但吞吐涨了 30 倍？
2. 前缀缓存节省的是什么？为什么它需要调度器配合？
3. Chunked prefill 牺牲了什么换来了什么？
4. 一个 8B 模型在 4096 上下文、100 并发下，权重和 KV 谁占的显存多？

答不上来就回看 1.1（算术强度）和 1.2（三种浪费）——这几条是理解后续所有优化的地基。

---

下一页 → [02 架构总览与请求生命周期](02_architecture_overview.md)

# 07 · 并行策略：TP / DP / EP / PP / PD

> 对应进阶 [10 并行策略内部实现](../advanced/10_parallelism_internals.md)。
> 源码：`python/sglang/srt/distributed/parallel_state.py`、`layers/dp_attention.py`、
> `python/sglang/srt/disaggregation/`

---

## 7.1 五种并行，各解决什么问题

一张表先说清：

| 并行 | 切什么 | 解决什么 | 代价 | 参数 |
|---|---|---|---|---|
| **TP** 张量并行 | 每层的权重矩阵 | 单卡放不下模型 / 提升单层算力 | 每层 2 次 all-reduce | `--tp-size` |
| **PP** 流水线并行 | 层（按深度切） | 超大模型、跨节点 | pipeline bubble | `--pp-size` |
| **DP** 数据并行 | 请求 | 吞吐 | 每卡一份完整模型 | `--dp-size` |
| **EP** 专家并行 | MoE 的专家 | MoE 模型（专家数 >> 卡数） | all-to-all 通信 | `--ep-size` |
| **PD** 分离 | prefill / decode 角色 | TTFT 与 TPOT 解耦 | KV 要跨机传输 | `--disaggregation-mode` |

**新手最容易搞混的是 DP 和 TP**：

```
TP=4：4 张卡合起来装一个模型，一起算同一个请求（请求被"切"了）
DP=4：4 张卡各装一个完整模型，各算各的请求（请求被"分"了）
```

---

## 7.2 TP：张量并行

### 怎么切

```
ColumnParallelLinear / RowParallelLinear / QKVParallelLinear
（layers/linear.py:145 LinearBase 的子类）
```

一个 MLP `Y = X·W`（`W: [K, N]`）：

```
Column 切（切输出维）：
    卡0: W[:, :N/2]   卡1: W[:, N/2:]
    → 每张卡算出 Y 的一半，拼接即可，无需通信

Row 切（切输入维）：
    卡0: W[:K/2, :]   卡1: W[K/2:, :]
    → 每张卡算出部分和，必须 all-reduce 才能得到完整 Y
```

典型的 transformer 层：`qkv_proj`（column）→ attention → `o_proj`（row，需要 all-reduce）
→ `gate_up_proj`（column）→ `down_proj`（row，需要 all-reduce）。

### 代价：all-reduce

**每层 2 次 all-reduce**（attention 后一次、MLP 后一次）。60 层就是 120 次。

这就是为什么 TP 通常限制在单机内（NVLink 带宽远高于跨机 IB）。

### 关键优化：融合 all-reduce + RMSNorm

```python
# distributed/parallel_state.py:786 GroupCoordinator.fused_allreduce_rmsnorm
if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
    use_1stage_ar = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
else:
    total_bytes = input_.numel() * input_.element_size()
    use_1stage_ar = total_bytes <= 128 * 1024
fused_outputs = ca_comm.custom_fused_ar_rms(_, residual_inp_, weight_, eps, use_1stage_ar)
```

**为什么这个融合重要**：

```
朴素：all-reduce 写 [tokens, hidden] 到 HBM
      → RMSNorm 再从 HBM 读回来
      = 一次完整的 HBM 往返

融合：reduce 的结果直接落在 norm kernel 的寄存器里
      = 省掉一次 [tokens, hidden] 的读 + 写
```

60 层 × 2 个 norm × 8K tokens × 7168 hidden × 2 Byte ≈ **每次前向省掉几百 MB 的 HBM 流量**。

1-stage（每 token 一个 block，上限 80 token）vs 2-stage 用 128 KiB 阈值切换，
这样大 prefill 批会自动走 2-stage 而不是报错。

### 层间通信的编排

`layers/communicator.py:223 ScatterMode` 定义了 TP/DP rank 间的数据布局：

```
假设 TP=4, DP=2, 开启 dp-attention，系统处理 seq a,b,c,d
Model input/output: [ab, ab, cd, cd]  （四个 rank）
SCATTERED:     [a, b, c, d]
TP_ATTN_FULL:  [ab, ab, cd, cd]
FULL:          [abcd, abcd, abcd, abcd]
MOE_FULL:      MoE 组内 full
```

`LayerCommunicator` 的职责就是**用最少的 collective 在这些模式间搬运 hidden states**，
并把 collective 与其后的 layernorm 融合。

---

## 7.3 DP attention：DeepSeek 系模型的必修课

### 核心洞察

```
Attention 层：权重小（KV 头少），激活大 → 通信量大
MoE 层：      权重大（几十 GB）        → 通信量小但计算量大
```

如果全用 TP：MoE 权重被切碎，每个 MoE 层都要 all-reduce，通信成为瓶颈。
如果全用 DP：每张卡都要装完整模型，显存放不下。

**折中：attention 用 DP，MoE 用 TP/EP。**

```python
# arg_groups/fields/parallel.py:170
enable_dp_attention: A[bool,
    "Enabling data parallelism for attention and tensor parallelism for FFN. "
    "The dp size should be equal to the tp size."]
```

### 效果

- **attention 部分完全无通信**（每个 DP rank 有自己的 KV cache 和自己的批）
- **MoE 部分走 EP**（专家并行：只发 token，不发权重）
- 代价：attention（DP）和 MoE（EP/TP）的并行度不同，中间要做 gather/scatter

```python
# layers/dp_attention.py:585
get_parallel().attn_tp_group.reduce_scatter_tensor(...)
...
torch.distributed.all_gather_into_tensor(...)
```

### MLP 与 attention 的并行度解耦

```python
# utils/common.py:3958
def require_mlp_tp_gather():
    """Check if the input of MLP is obtained by all-gather rather than all-reduce."""

def require_attn_tp_gather():
    """Check if the input of attention is scattered."""

def require_mlp_sync():
    return get_parallel().enable_dp_attention or require_gathered_buffer()
```

**`require_mlp_sync()` 是关键**：DP attention 下每步必须做一次全局同步，
保证所有 rank 的 batch 形状一致（否则 collective 会 hang）。

这就引出了第 04 章的 `PrefillDelayer`——为了减少这种同步带来的空等。

### 一个顺带的控制面优化

```python
# arg_groups/fields/parallel.py:177
enable_dp_attention_local_control_broadcast: A[bool,
    "With DP-attention, send control messages to every DP group leader and "
    "broadcast within attn_tp_group instead of the full tp_group. "
    "Eliminates a costly all-ranks gloo sync on every scheduler iteration."]
```

**每个 scheduler 迭代省一次全 rank 的 gloo 同步**——纯控制面的胜利。

### 相关参数

```bash
--enable-dp-attention --dp-size 8      # DeepSeek 类模型的常见配置
--moe-dense-tp-size                    # MoE 里 dense 层的 TP size
--load-balance-method                  # auto/round_robin/follow_bootstrap_room/
                                       # total_requests/total_tokens
```

---

## 7.4 EP：专家并行

MoE 模型有 E 个专家（DeepSeek-V3 是 256 个），但只有 top-k（8 个）被激活。

```
TP 方案：每张卡装 E/tp_size 个专家 → 权重切开了，但每个 token 都要广播
EP 方案：每张卡装 E/ep_size 个专家 → 把 token 路由到专家所在的卡
```

EP 的通信是 **all-to-all**（把 token 发到对应的 expert rank，算完再发回来）。

`layers/moe/token_dispatcher/` 下有多种实现：

| Dispatcher | 说明 |
|---|---|
| `standard` | TP-only MoE，不需要 dispatch |
| `deepep` / `deepep_v2` | DeepEP 库，高吞吐 all-to-all |
| `flashinfer` | FlashInfer 的 MoE A2A |
| `pplx` / `nixl` / `mooncake` / `mori` | 其他传输后端 |
| `ascend_tp` | NPU |

`--moe-a2a-backend` 选择，`--ep-size` 指定大小。

### 一个必须知道的分组 GEMM 陷阱

```python
# moe_runner/runner.py 启动校验
# hpc_ops 必须被每个 MoE 层选中，否则
# "would consume global ids as local ones and misroute tokens under EP>1,
#  so fail loudly at startup"
```

MoE kernel 用的是**局部专家 id**，如果某些层用全局 id 就会路由错。
SGLang 的选择是**启动时大声报错**而不是运行时算错——这是好设计。

---

## 7.5 PP：流水线并行

按层切：卡 0 拿 1-15 层，卡 1 拿 16-30 层……

```
卡0: [层1-15] [层1-15] [层1-15] ...
卡1:          [层16-30] [层16-30] ...
                ↑ bubble
```

`--pp-size` 指定。`--pp-max-micro-batch-size` 控制微批大小。

PP 下的两个额外机制：
1. **动态 chunk size**（第 04 章）：让每个 micro-batch 的执行时间恒定，减少 bubble
2. **`scheduler_pp_mixin.py`**：PP 专属的调度逻辑

---

## 7.6 PD 分离：把 prefill 和 decode 放到不同机器上

### 动机

```
Prefill：compute-bound，对 batch 大小不敏感，要快（TTFT）
Decode：  memory-bound，需要大 batch 和长生命周期 KV（TPOT）
```

**混在一起时，每次 prefill 都会打断所有 decode。** PD 分离让两者各自跑到最优形状，
并且可以独立扩缩容。

```bash
# Prefill 节点
python -m sglang.launch_server --disaggregation-mode prefill ...
# Decode 节点
python -m sglang.launch_server --disaggregation-mode decode ...
```

### Decode 侧的关键技巧

`disaggregation/decode.py` 的生命周期：

```
1. PreallocQueue:  握手 + 预分配 KV
2. TransferQueue:  轮询传输状态
3. WaitingQueue:   用请求构造 PrebuiltExtendBatch
4. RunningBatch:   合并进 running batch 开始 decode
```

**第 3 步是精髓**：decode 节点**从不跑 prefill forward**，只构造一个
`PrebuiltExtendBatch` 来填充 `req_to_token` / `out_cache_loc` 元数据。

```
"prefill" 的代价 = 一次元数据填充，而不是一次完整前向
```

### 传输后端

`--disaggregation-transfer-backend` 的 choices：`mooncake`（**默认**）、`nixl`、
`ascend`、`fake`、`mori`、`mooncake_tcp`。

还有 KV 校验（`disaggregation_enable_kv_checksum`）：对每个请求的 KV 页算 Adler-32，
decode 侧验证。

### 为什么需要 `--num-reserved-decode-tokens`

默认 512。decode 节点必须在传输完成**之前**就为请求预留 KV 槽位，
所以要留一部分预算。

---

## 7.7 怎么选：一张决策表

| 你的情况 | 建议 |
|---|---|
| 模型单卡放得下，要吞吐 | **DP**（多实例 + 网关路由） |
| 模型单卡放不下，单机内 | **TP**（`--tp-size` = 卡数） |
| MoE 模型（DeepSeek/Qwen-MoE） | **TP + EP**，或 **DP attention** |
| 超大模型跨节点 | **TP + PP** |
| TTFT 和 TPOT 都要好 | **PD 分离** |
| 长上下文 + 高并发 | TP + KV 量化 + HiCache |

官方调参文档的原话：

> Data parallelism is better for throughput. When there is enough GPU memory,
> always favor data parallelism for throughput.

**优先 DP，除非显存不够。**

---

## 7.8 本章自检

1. TP 和 DP 的区别是什么？为什么 TP 通常限制在单机内？
2. `fused_allreduce_rmsnorm` 省掉了什么？
3. DP attention 为什么对 DeepSeek 类模型有效？代价是什么？
4. PD 分离下 decode 节点为什么不用跑 prefill？
5. `require_mlp_sync()` 为真时会发生什么？带来了什么问题？
6. 你有 8 张卡、一个 70B 模型、要最大吞吐，你选什么？

---

下一页 → [08 投机解码与高级特性](08_speculative_and_advanced.md)

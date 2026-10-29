# Qwen Direct 第 26 次请求 OOM：缓存来源链与恢复建议

> 发布说明（2026-10-30）：本报告随组会 PPT 和总结同步。保留原实验事实、历史状态与预算；机器路径改用别名，未包含的运行材料明确标为本地资料。文中原“不推送”描述属于编写时记录，本次推送依据后续用户授权。

## Material Passport

- 日期：2026-10-28；工作流：academic-research-suite / experiment-agent。
- 范围：只读日志和本地源码审计、CPU 缓存契约模拟；新增模型请求、GPU 分配和重试均为 0。
- 状态：源码与轨迹支持的诊断；**未进行 GPU 修复对照，不声称已完成因果验证**。
- 运行：qwen_direct_pilot30_v1_1_2026-10-28（本地资料：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_1_2026-10-28`）。旧失败、预测和账本均保留。

## 结论与实际完成范围

高置信度判断：vLLM 跨请求保留视觉编码结果，逐步占用 GPU 显存，是此次 OOM 的主要原因。来源链和输入规模能够解释故障发生在第 26 条；但没有逐条显存堆快照，也未做清理缓存后的 GPU 对照，其他运行缓冲区的贡献尚不能完全排除。

本轮已尝试 26 次：QVHighlights 20/20 完整，TVSum 前 5 条完整，第 6 条 `Se3oxnaPsz0` 在视觉编码阶段发生 CUDA OOM，后 4 条未请求。因此是 **25 条有效结果、1 条技术失败、4 条未执行**。完整 QV20 可单独评价；TVSum5 只能标为执行前缀的部分诊断，不能冒充完整 val10 结果。

第 26 次请求仅运行约 0.39 秒即失败，尚未生成评分。失败原因不是 JSON 格式、评分向量长度或标签评价异常。原始日志第 124 行（本地资料：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_1_2026-10-28/worker.log:124`）记录：视觉 GELU 申请 138 MiB 时仅剩 80.06 MiB 可用，进程占用 23.42 GiB，其中 PyTorch 已分配 22.68 GiB。

## 可审计的来源链

| 环节 | 实际行为与影响 | 本地来源 |
|---|---|---|
| 初始化额度 | encoder 缓存额度为 **151,250 tokens**，权重占用 **16.6168 GiB**，KV 固定为 **2 GiB**。缓存额度不是一次性预分配全部显存。 | worker.log:20（本地资料：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_1_2026-10-28/worker.log:20`） |
| 缓存额度计算 | 额度取调度配置与模型最大多模态项估计的较大值；Qwen3-VL 的最大视频 token 估计还包含时间戳系数。它不是根据这批真实输入的显存余额逐条收缩。 | encoder_cache_manager.py:296（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/core/encoder_cache_manager.py:296`）、qwen3_vl.py:693（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/model_executor/models/qwen3_vl.py:693`） |
| 完成请求 | `free()` 只撤销请求引用，将条目标为可回收；编码结果仍保留，直到下一次分配时额度不足才逐出。 | encoder_cache_manager.py:225（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/core/encoder_cache_manager.py:225`） |
| GPU 中的实际引用 | 每个视频的编码结果写入 `encoder_cache[mm_hash]`；仅收到调度器 `free_encoder_mm_hashes` 时才删除相应张量。 | gpu_model_runner.py:2163（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py:2163`）、gpu_model_runner.py:768（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py:768`） |
| Qwen3 视觉输出宽度 | 主视觉特征和三层 DeepStack 特征沿特征维拼接，缓存宽度是 `4096 × (1+3) = 16384`，不能只按 4096 估算。 | qwen3_vl.py:574（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/model_executor/models/qwen3_vl.py:574`）、模型配置（本地资料：`$SHARED_DATA_ROOT/MemDreamer/models/Qwen3-VL-8B-Instruct/config.json`） |
| 显存预算的另一限制 | 显式指定 `kv_cache_memory_bytes` 后，当前版本不按 `gpu_memory_utilization=0.90` 自动决定 KV 大小。模型仍执行 profile run，但不能把 0.90 当作可靠的总显存保护线。 | gpu_worker.py:282（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/worker/gpu_worker.py:282`） |

日志中的“64 帧 override 超过模型最大 2 帧”来自 dummy profiling 输入构建，不代表实际样本只传入了 2 帧。冻结输入均为 64 帧；CPU 与 vLLM 的实际多模态 token 展开已经逐条严格匹配。该提示说明 profiling 的输入构造与真实视频不同，不能用这次初始化成功推断后续持续运行一定安全。

## CPU 模拟与显存估算

CPU 脚本从当前安装的 `encoder_cache_manager.py` 提取**原 `EncoderCacheManager` 类**，在独立命名空间执行；不导入 vLLM、PyTorch、CUDA 或模型。输入是冻结的样本顺序、帧哈希和 prompt token 计数。

模拟用完整 prompt token 数作为视频 encoder token 数的**上界**，并非声称二者完全相等。视频占位范围是完整 prompt 的子集，因此连这个上界都未耗尽额度，就可以排除“额度已经触发正常逐出”的解释。

| 检查 | 结果 | 能说明什么 |
|---|---:|---|
| 前 25 条完整 prompt token 合计 | 125,992 | 仍低于 151,250 |
| 加入失败的第 26 条后的上界 | 130,609 | 正常分配逻辑仍无需逐出 |
| 前 25 条不同帧哈希 | 25 | 每条都可能新增独立编码缓存 |
| 原额度下 CPU 模拟 26 次分配 | 0 次逐出，管理器保留 26 项 | 与旧请求缓存持续留存一致 |
| 仅将容量设为 8,192 的 CPU 反事实，模拟全部 30 条 | 29 次逐出，最多保留 1 项 | 原逐出通路能在较小容量下持续回收；不是 GPU 修复验证 |

注意：模拟中的第 26 个**管理器条目**是执行编码前的分配状态；实际第 26 条编码失败，不能因此声称它已经生成第 26 份 GPU 编码张量。

前 25 条成功输入的视觉 token 共 **114,240**。按照源码拼接宽度和 BF16：

```text
114240 × 16384 × 2 bytes = 3,743,416,320 bytes = 3.486328125 GiB
```

这是保留编码张量的**估算下界**，不是显存实测；尚未计入时间戳占位填充、分配器开销和当前视觉编码激活。

支持性案例：第 **24** 次 `JKpqYvAdIsw` 与第 **26** 次 `Se3oxnaPsz0` 均为 `video_grid_thw=[[32,20,26]]`，前者成功、后者失败。同样大小的输入在更晚位置失败，支持累积显存压力解释；视频内容不同，不能单凭这一对案例宣布严格因果成立。

## 已核实无效或不宜混入的修复

- **`LLM.reset_mm_cache()` 不能清掉本次问题中的 GPU encoder 输出字典。** 本机调用链只清多模态输入/处理缓存及 `mm_budget.cache`：llm.py:363（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/entrypoints/llm.py:363`） → core.py:471（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/engine/core.py:471`） → gpu_model_runner.py:601（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py:601`） → utils.py:126（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/v1/worker/utils.py:126`）。
- `reset_prefix_cache()` 处理 KV 前缀缓存；本轮本来就关闭了 prefix caching。`mm_processor_cache_gb` 也已经为 0，两者不能代替 encoder 输出缓存回收。
- 单独 `torch.cuda.empty_cache()` 不会释放仍被字典引用的活跃张量。不要把它当作来源链已确认的修复。
- 当前版本的 `encoder_cache_size` 为不可直接配置的 `init=False` 字段。`scheduler_cls` 扩展入口真实存在，但源码明确提示接口兼容性不保证；若后续研究限容方案，应固定版本并另行验证，不在本次恢复中同时更换调度实现。scheduler.py:88（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/config/scheduler.py:88`）、scheduler.py:120（本地资料：`$SHARED_DATA_ROOT/longvt-repro/env/inference/lib/python3.10/site-packages/vllm/config/scheduler.py:120`）

## 最保守的恢复方案：待授权，不执行

建议启动**一个新的独立 worker，仅补齐以下 5 条**，完成后退出进程释放缓存。保持原模型、64 帧像素、prompt、评分格式、解码和评价规则；不重算已经成功的 25 条，不改变推理算法，不修改旧失败记录。

| 样本 | 本轮真实状态 | 恢复动作 |
|---|---|---|
| `Se3oxnaPsz0` | 已请求 1 次，OOM | 技术失败重试 1 次 |
| `Yi4Ij2NM7U4` | 未请求 | 首次推理 |
| `cjibtmSLxQ4` | 未请求 | 首次推理 |
| `fWutDQy1nnY` | 未请求 | 首次推理 |
| `sTEELN-vY30` | 未请求 | 首次推理 |

这批 5 条小于保守分批上限 10。以后扩样可固定每 worker 不超过 10 条后退出重载；这是降低缓存累积风险的工程措施，尚不是已验证的 GPU 保证。新增 worker 应记录每条前后的显存与退出收据，首个错误仍停止，不自动追加重试。

原协议是 **30 次总尝试、0 次重试**。当前已用 26 次，恢复 5 条会使累计达到 **31 次**：新增调用数是 **5**，其中超出原总额度的额外尝试是 **1**，重试额度需要从 0 调整为 1。两者不能混淆。还需记录新增一个 worker，即从 v1 初始化失败、v1.1 正式运行，到恢复 worker 的累计第三次工程启动。

| 预算项目 | 当前 | 建议恢复后的约束 |
|---|---:|---|
| 模型请求尝试 | 26 | 上限 31，需额外 1 次尝试授权 |
| 失败样本重试 | 0 | 上限 1，需授权 |
| 已计费 token，含失败全预留 | 140,457 | 原绝对上限 245,760 不变 |
| 5 条最大新增 token 预留 | — | 34,287；累计保守上界 174,744 |
| GPU 生命周期累计，含 v1 失败 | 280.481364 秒 | 原绝对上限 3,600 秒不变，剩余 3,319.518636 秒 |

本轮报告和 CPU 模拟**不构成新增尝试授权**，恢复尚未执行。额度归当前 Direct pilot；此前 QV 诊断和 TVSum transfer 的独立账本不转入、不重置。授权后也必须先冻结恢复清单、旧记录哈希、累计预算和新退出收据规则。

## 复现与新产物

脚本：diagnose_encoder_cache.py（本地资料：`$ZEROSHOT_ROOT/experiments/qwen_direct/diagnose_encoder_cache.py`）

已保存的机器结果：encoder_cache_diagnosis.json（本地资料：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_1_2026-10-28/encoder_cache_diagnosis.json`）。结果包含源码 SHA256 和函数真实行号、原始证据哈希、逐次 CPU 缓存事件、预算计算及待授权恢复清单。

```bash
CUDA_VISIBLE_DEVICES='' $SHARED_DATA_ROOT/longvt-repro/env/inference/bin/python \
  $ZEROSHOT_ROOT/experiments/qwen_direct/diagnose_encoder_cache.py \
  --run $ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_1_2026-10-28 \
  --output /tmp/qwen_direct_encoder_cache_recheck.json
```

脚本拒绝覆盖已存在的输出。此次 CPU 模拟全部断言通过，Ruff 通过；新增实验模型调用 0、GPU 分配 0、恢复重试 0。

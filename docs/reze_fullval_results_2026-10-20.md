# QVHighlights 最新实验结果：REZE 基线与高光重排

整理日期：2026-10-20（北京时间）。本轮全量评估完成于 **2026-10-20 04:14:44**，覆盖 validation 全部 **1,550 查询、1,519 视频片段、474 源视频**。本次整理没有新增推理。

**当前最好点估计为 P_pair：VeryGood mAP 43.76、Hit@1 76.84，相对 REZE B0 提高 0.77 / 5.16 个百分点。** 两项配对区间均高于零，但 mAP 点估计增幅仍不足原方案的 1 pp 目标。P_pair 相对 P_abs 的 mAP 区间跨零，目前更明确的收益体现在首个高光选择。

本轮只验证前五候选的复评与双向比较；尚未加入 memory、event graph、回源查证或训练。此前 100 条训练探索未通过 A 阶段门槛，本轮是用户另行要求的全量扩展，保留原停止结论。validation 此前已查看，不是未触碰的测试集。

## 1. 完整 validation 结果

均为官方 **HD** 指标，百分数越高越好；没有本轮 MR 或 test 成绩。新实验与对照使用同一标签和官方评估器，所有查询均保留在分母中。

| 方法 | Fair mAP / Hit@1 | Good mAP / Hit@1 | VeryGood mAP / Hit@1 | 查询回退 |
|---|---:|---:|---:|---:|
| MERIT 原适配，历史主实验 | 46.69 / 41.68 | 37.85 / 39.42 | 21.67 / 31.23 | 9 |
| CLIP ViT-B/32，2s 中点帧 | 69.81 / 77.29 | 59.57 / 74.65 | 36.50 / 61.68 | 0 |
| VTG-GPT，作者中间文本评分 | 67.02 / 77.03 | 57.54 / 74.19 | 35.10 / 61.23 | 0 |
| REZE B0，Qwen3-VL-8B | 80.07 / 86.39 | 69.51 / 84.71 | 42.99 / 71.68 | 0 |
| B0 + P_abs，候选绝对复评 | 80.37 / 88.77 | 70.03 / 86.77 | 43.57 / 74.06 | 0 |
| B0 + P_pair，双向比较重排 | 80.39 / 89.61 | 70.11 / 88.00 | **43.76 / 76.84** | 0 |

来源：[本轮完整指标](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/results/evaluation.json)、[CLIP](../comparisons/qvhighlights/reze_fullval_2026-10-20/references/clip_val/evaluation_report.json)、[VTG-GPT](../comparisons/qvhighlights/reze_fullval_2026-10-20/references/vtg_gpt_val/evaluation_report.json)、[MERIT 历史报告](../comparisons/qvhighlights/merit_main_2026-10-15/val_report.json)。

CLIP 用原视频代表帧与原始 query 计算相似度；VTG-GPT 复用作者发布的 caption 和 query 改写，没有重新生成这些文本。REZE B0 是按论文附录完成的独立实现；本地模型、输入与预算并未在这些方法间统一，不能把跨方法分差单独归因于记忆机制。LongVT、VideoARM 和 MERIT 开发集实验仍见[历史总汇](reproduction_results_summary_2026-10-19.md)；VideoARM 的成功子集不加入本表排名。

## 2. 本轮实际改进

1. **冻结基线与候选。** 复用已有 B0 全量预测；B0 对 3s 窗口以 2 FPS 观察，取 Yes/No 的 top-20 概率，插值聚合到官方 2s 网格。按 B0 分数取前五格，同分按 clip ID 排序。
2. **固定输入证据。** 每个候选最多四帧，均来自 B0 已观察的采样帧；重新解码以校验像素一致性。P_abs 与 P_pair 的 A/B 帧完全匹配，不获取新时间位置的画面。
3. **分别检验绝对复评与相对比较。** 每条 query 有 10 个无序候选对、两个输入顺序，每种方法各 20 次调用。P_abs 问 A 是否充分展示 query，B 仍作为上下文；P_pair 问 A 是否有证据支持的相对优势。两者使用同一冻结 Qwen3-VL-8B-Instruct，BF16，输出一个 Yes/No token。
4. **保守重排。** P_abs 对每个候选的四次概率取平均；P_pair 仅在一个方向概率 ≥0.6、反方向 ≤0.4 时采纳概率差，并累加比较优势。平局依次按 B0 分数和 clip ID 打破。最终只将原前五格的分数重新分配，其余分数不变，不新增分值。

模型 revision：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。提示、候选、阈值和排序逻辑沿用 A 阶段冻结设置，详见[提示与接口](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/run/source_snapshot/experiments/reze_evidence/backend.py)、[排序实现](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/run/source_snapshot/experiments/reze_evidence/ranking.py)和[全量计划](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/run/plan.json)。

## 3. 配对差异与不确定性

差值由未舍入的逐查询指标计算，单位为百分点。按 **474 个源视频聚类**进行 10,000 次配对 bootstrap，seed=42；重采样时保留查询权重。区间为探索性区间，未做多重比较校正。

| 比较 | VeryGood mAP Δ [95% CI] | Hit@1 Δ [95% CI] |
|---|---:|---:|
| P_abs − B0 | +0.582 [0.289, 0.879] | +2.387 [0.581, 4.177] |
| P_pair − B0 | +0.771 [0.479, 1.064] | +5.161 [3.488, 6.863] |
| P_pair − P_abs | +0.189 [−0.059, 0.442] | +2.774 [1.307, 4.285] |

两种复评对 B0 的提升在本次验证集上有配对统计支持。P_pair 比 P_abs 更明确地改善 Hit@1，尚不能确认其 mAP 优势。区间不涵盖模型预训练、实现选择、重复开发或 BF16 调度造成的系统差异。

逐查询 mAP 胜 / 平 / 负：P_abs 对 B0 为 **477 / 712 / 361**，P_pair 对 B0 为 **470 / 782 / 298**，P_pair 对 P_abs 为 **361 / 876 / 313**。P_abs / P_pair 实际改变了 1,498 / 1,292 条分数向量；三组最高分并列均为 146 条。分数多重集不变，因此本方法仍受候选覆盖和原始并列分数限制。

## 4. 调用、token 与时间开销

下表 B0 为历史运行；P_abs / P_pair 为各自的**新增**开销。模型帧呈现次数不是唯一帧数或编码器实际执行次数。

| 方法 | 调用数 | 输入 tokens | 输出 tokens | 模型帧呈现次数 | 含 B0 总 token 比 |
|---|---:|---:|---:|---:|---:|
| B0，历史 | 77,308 | 45,071,825 | 77,308 | 463,764 | 1.000× |
| P_abs，新增 | 31,000 | 24,648,800 | 31,000 | 247,984 | 1.547× |
| P_pair，新增 | 31,000 | 24,927,800 | 31,000 | 247,984 | 1.553× |

本轮实际新增 **62,000 次调用**，与返回账本一致；预算上限 68,200 次，未使用重试余量。两组共同使用一张 RTX 4090，墙钟约 **3 小时 53 分钟**，实际分配 **3.868 GPU 小时**，工作进程重启 0 次、查询回退 0 条。B0 缓存复用，不计为本轮新增推理。

单独采用 P_pair 时，含 B0 的冷启动 token 总量为 **70,107,933**；P_abs 为 **69,828,933**。两组共享推理批次，账本中的各约 3,322.89 秒是批次时间对半分摊，**不是各方法独立运行的实测延迟**。

历史 B0 使用四张 RTX 4090，墙钟 32.50 分钟；CLIP 单卡 16.79 分钟；VTG-GPT 单卡 96.95 秒仅覆盖已有文本的读取与评分，不含 caption/query 生成。这些时间不能直接解释为同预算端到端速度排名。

## 5. 与此前 A 阶段的关系

| 固定训练 explore100 | VeryGood mAP | Hit@1 |
|---|---:|---:|
| B0 | 44.41 | 71.00 |
| P_abs | 44.71 | 76.00 |
| P_pair | 44.58 | 78.00 |

A 阶段两种方法的 mAP 提升均不足 1 pp，配对区间均跨零，因此决定为 `stop_v1_no_qualifying_stage_A_arm`。全量结果不追溯修改该决定，也不将训练探索集和 validation 的绝对分数直接比较。原结果、门槛与成本见 [A 阶段归档](../comparisons/qvhighlights/reze_fullval_2026-10-20/stage_a/REPORT.md)。

后续若继续，应先判断前五候选覆盖、并列分数和未改变排序的查询是否限制 mAP，再决定是否引入新证据。当前结果支持继续研究候选比较，但还不能证明事件记忆或回看有收益。本轮未自动启动下一阶段。

## 6. 结果与复算入口

- [结果包与 CPU 复算说明](../comparisons/qvhighlights/reze_fullval_2026-10-20/README.md)：完整性、官方指标、逐查询结果、配对区间、分数置换和调用账本核验。
- 全量预测：[B0](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/results/B0_predictions.jsonl)、[P_abs](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/results/P_abs_predictions.jsonl)、[P_pair](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/results/P_pair_predictions.jsonl)。
- [逐查询指标](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/results/per_query.jsonl)、[完成凭证](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/run/completion.json)、[实际调用账本](../comparisons/qvhighlights/reze_fullval_2026-10-20/fullval/run/attempts_0.jsonl)。
- [来源与公开文件双哈希清单](../comparisons/qvhighlights/reze_fullval_2026-10-20/publication.json)：预测和指标保持原始字节；机器路径使用来源别名，历史凭证中的原始哈希不改写为公开文件哈希。

完整原始逐请求 logits、像素证据、视频和权重保留在原实验存储，本次上传精选结果与冻结代码。仓库可独立复算已发布预测的 HD 指标，不代表可以直接接续原机器上的推理任务。

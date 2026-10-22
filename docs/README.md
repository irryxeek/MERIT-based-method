# 文档索引

更新日期：2026-10-22。以下结果、方案和详细报告均已纳入本仓库；原始运行产物的发布范围以各报告说明为准。

- [实验实现与复现入口](reproduction_index.md)：代码、配置、数据条件、测试与结果对应表。
- [仓库代码补齐报告](repository_completion_2026-10-22.md)：新增实现、部署修复、验证结果和发布边界。

- [实验结果汇总报告](experimental_results_report_2026-10-22.md)：完整实验、开发对照、跨数据集结果、成本和适配分析。
- [会议纪要后的推进计划](next_steps_after_meeting_2026-10-22.md)：候选覆盖、同预算定向补证、长视频粗到细及模块消融，新实验尚未启动。
- [报告同步发布约定](report_publication_workflow.md)：以后每次推送包含新增或修改的报告、方案和索引；[来源及公开文件哈希](report_publication_manifest.json)记录本次覆盖范围。

## REZE 完整结果与原执行方案

| 文档 | 用途 |
|---|---|
| [完整 validation 结果](reze_fullval_results_2026-10-20.md) | 1,550 查询，B0 / P_abs / P_pair 的 VeryGood mAP 为 42.99 / 43.57 / 43.76，Hit@1 为 71.68 / 74.06 / 76.84。 |
| [结果包与 CPU 复算](../comparisons/qvhighlights/reze_fullval_2026-10-20/README.md) | 预测、逐查询比较、成本、冻结代码与来源哈希。 |
| [全量执行记录](reze_fullval_execution_2026-10-20.md) | 完成状态、范围、冻结配置和成本记录。 |
| [A 阶段结果](reze_stage_a_results_2026-10-19.md) | 固定 explore100 未通过原门槛的历史结论。 |
| [A 阶段执行状态](reze_execution_status_2026-10-19.md) | 工程修复、检查和当时的停止决定。 |
| [原执行方案 v1](reze_execution_plan_2026-10-19.md) | A/B/C 阶段设计、确认集、预算和门槛。 |
| [原研究方向](reze_research_directions_2026-10-19.md) | 比较、事件证据、记忆与回看的假设。 |
| [冻结参数](../comparisons/qvhighlights/reze_fullval_2026-10-20/stage_a/plan.json)、[原方案发布清单](reze_plan_publication_2026-10-20.json) | 原始方案参数与发布凭证，保持原样。 |

原 v1 制定于 2026-10-19，公开版本附截至 2026-10-20 的执行状态。A 阶段及单独授权的全量评分扩展已完成；A 阶段未通过原门槛，B/C 与 confirm100 尚未执行。新的会议推进计划单独记录，不改写原停止决定。

## 跨数据集结果与数据准备

| 文档 | 范围与结论 |
|---|---|
| [跨数据集结果与适配分析](reze_cross_dataset_results_analysis_2026-10-22.md) | 任务目标、摘要选择诊断及适配优先级。 |
| [TVSum 完整结果](reze_tvsum_execution_2026-10-22.md) | 50 条完成，固定 val10 / all50 Top-5 mAP 为 61.32 / 54.89。 |
| [MR.HiSum 可用全量结果](reze_mrhisum_full_execution_2026-10-21.md) | 1,888/2,000，mAP50 / mAP15 / F1 为 57.00 / 23.90 / 38.27。 |
| [MR.HiSum pilot100](reze_mrhisum_execution_2026-10-21.md) | 先前固定 100 条，保留独立运行记录。 |
| [QFVS 结果分析](reze_qfvs_analysis_2026-10-20.md) | 四查询 F1 38.63，低于同预算均匀/随机选片。 |
| [QFVS 推理记录](reze_qfvs_execution_2026-10-20.md) | 四条完整视频、固定查询与原生评价。 |
| [跨数据集与 SVHighlights 计划](reze_cross_dataset_2026-10-20.md) | 含 10 月 22 日状态更正：七场启动失败，无有效 REZE 指标。 |
| [QFVS 原始视频记录](qfvs_raw_videos_2026-10-20.md) | 下载、检查及标注网格边界。 |
| [TVSum / MR.HiSum 数据记录](tvsum_mrhisum_download_2026-10-20.md) | 下载范围、失败补试及可用性。 |
| [TVHighlights 数据核查](tvhighlights_download_2026-10-21.md) | 尚无本地可用视频和标注，与 TVSum 分开记录。 |

## MERIT 适配与开发实验

| 文档 | 内容 |
|---|---|
| [原 QVHighlights 适配](merit_qvhighlights_adaptation.md) | 三键检索、时间邻域与完整 2 秒评分。 |
| [密集评分适配](merit_hd_adapter_implementation_2026-10-16.md) | M0 / D0 / H1 与固定 train100。 |
| [方法新增内容](merit_method_additions_2026-10-16.md) | 区分设计目标、原型与待验证贡献。 |
| [高光证据研究方案](highlight_evidence_research_2026-10-16.md) | 四类 cue、事件关系和定向查证设计。 |
| [证据 pilot](highlight_evidence_pilot_2026-10-16.md) | 固定 train20 的 E1/E2/E3 及回退。 |
| [记忆粒度](highlight_memory_granularity_2026-10-16.md) | 固定候选与输入帧的 C10/F2 对照。 |
| [可靠性修复](highlight_memory_repair_2026-10-16.md) | 结构化输出、失败诊断与同样本复测。 |
| [评分输入消融](highlight_memory_ablation_2026-10-16.md) | 帧、描述、cue 和图边对照。 |
| [完整 2 秒纯记忆评分](highlight_memory_only_2026-10-16.md) | 完整时间轴描述与 cue 读取。 |
| [纯记忆评分分析](highlight_memory_only_analysis_2026-10-17.md) | 同分、证据利用和读取问题。 |
| [R1/R2 方案](highlight_memory_reader_plan_2026-10-17.md) | 冻结的读取修复与停止规则。 |
| [R1 结果](highlight_memory_reader_run_2026-10-17.md) | R1 已完成，R2 按规则未运行。 |

## 相邻零样本项目报告

| 文档 | 内容 |
|---|---|
| [CLIP / VTG-GPT 结果](zeroshot_baselines/RESULTS_2026-10-19.md) | 两种对照的结果及文本复用边界。 |
| [CLIP / VTG-GPT 协议](zeroshot_baselines/PROTOCOL.md) | 数据、模型和评分流程。 |
| [REZE 基线结果](zeroshot_baselines/REZE_RESULTS_2026-10-19.md) | 原 B0 独立实现的结果、修复和成本。 |
| [REZE 基线协议](zeroshot_baselines/REZE_PROTOCOL.md) | 采样、概率评分与时间网格。 |

## 历史完整结果与记录

- [此前实验总汇](reproduction_results_summary_2026-10-19.md)：MERIT、LongVT、VideoARM 的完整集、成功子集与开发结果。
- [统一分析](../comparisons/qvhighlights/analysis_2026-10-15/README.md)、[对比数据索引](../comparisons/qvhighlights/README.md)：精选归档及初始化审计。
- [后续方向摘要](research_directions_summary_2026-10-15.md)、[方案详版](research_directions_2026-10-15.md)：历史研究记录。
- [原执行快照](execution_status.md)、[原自动恢复说明](automation.md)：历史运行与配置，不表示任务当前仍在运行。

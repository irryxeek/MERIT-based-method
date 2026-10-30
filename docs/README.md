# 文档索引

## 最新：组会后 Direct 与 Grounding 进展（2026-10-28–30）

- [工作总结](progress_after_meeting_2026-10-27_to_2026-10-30.md)、[12 页可编辑 PPT](presentations/ReVALUE_Grounding实验进展_2026-10-28至10-30.pptx)、[PDF 预览](presentations/ReVALUE_Grounding实验进展_2026-10-28至10-30.pdf)、[讲稿与可编辑源页面](presentations/revalue_grounding_progress_20261030/README.md)。
- [Grounding 反例单页 PPT：qid448](presentations/Grounding反例_qid448_单页介绍_2026-10-30.pptx)、[讲稿与原帧证据](presentations/qv_grounding_counterexample_qid448_20261030/README.md)：标注 92–112 秒为鸡舍，预测 130–145 秒落到道路与花景，IoU=0。属于助手初筛案例；条件核验与重新定位的修正收益尚未验证。
- [Grounding 主线确认](temporal_grounding_mainline_2026-10-29.md)：以匹配 query 的事件起止区间为主要任务；定位 R1 与高光排序 mAP 分开。
- [Direct pilot 结果](qwen_direct_pilot_results_2026-10-28.md)：QV20 与 TVSum5 的有效整表输出全部视频内同分；[原协议](qwen_direct_pilot_protocol_v1_2026-10-28.md)、[启动兼容修订](qwen_direct_pilot_runtime_amendment_v1_1_2026-10-28.md)、[OOM 来源链](qwen_direct_oom_diagnosis_2026-10-28.md)。TVSum Direct 仍为 5/10，未补跑。
- [显式时间键结果](qwen_direct_temporal_results_2026-10-28.md)：20/20 完成，仍在视频内同分；[冻结协议](qwen_direct_temporal_output_protocol_v1_2026-10-28.md)、[准备收据](qwen_direct_temporal_preparation_receipt_v1_2026-10-28.md)。
- [单目标粒度结果](qwen_direct_granularity_results_2026-10-29.md)：40 次完成，20 对固定目标中 11 对分值不同；该比例不是准确率，未计算新全网格指标。见[冻结协议](qwen_direct_granularity_protocol_v1_2026-10-28.md)。
- [Grounding QV20 结果](qwen_direct_grounding_results_2026-10-29.md)：20/20 合法，R1@0.5 / R1@0.7 为 70% / 50%；[冻结协议](qwen_direct_grounding_protocol_v1_2026-10-29.md)。
- [新增 100 条结果](qwen_direct_grounding_expand100_results_2026-10-29.md)：96 次尝试、94 条合法，1 OOM、1 非法区间、4 未尝试；完整分母 R1 为 42% / 28%，52 条合法反例。见[原协议](qwen_direct_grounding_expand100_protocol_v1_2026-10-29.md)、[冷启动分批修订](qwen_direct_grounding_expand100_runtime_amendment_v1_1_2026-10-29.md)及[初始结果审阅](qwen_direct_grounding_expand100_review_2026-10-29.md)。两批均为 train 开发诊断，不是官方全量验证。
- [52 条反例的语义核查](qwen_direct_grounding_52_counterexample_analysis_2026-10-29.md)：助手初筛 22 条较明确错误、15 条标注／范围疑点、10 条混合、5 条证据不足；[逐例数据与五组代表证据](../comparisons/qwen_direct/grounding_expand100_case_audit_2026-10-29/README.md)。发布 GT 与原指标未更改，仍待独立人工复核，尚无 Agent 修复收益。

本次同步 18 份本阶段报告与方案，以及 PPT、PDF、讲稿和页面源文件；仅更新文档和精选证据，未新增实验。原始运行缓存、模型、完整视频和新运行器代码不属于这次汇报交付；相关引用明确标为本地材料。历史协议中的待授权状态按原日期保留，后续执行事实以对应结果报告为准。

## 上次组会及历史阶段记录

- [本周实验进度报告（2026-10-21–27）](weekly_experiment_progress_2026-10-21_to_2026-10-27.md)、[更新版 15 页 PPT](presentations/ReVALUE_实验进度汇报_2026-10-21至10-27_v2_TaskSpec补充.pptx)及[原 14 页 PPT](presentations/ReVALUE_实验进度汇报_2026-10-21至10-27.pptx)：更新版增加 TaskSpec 介绍，讲稿见[交付目录](presentations/revalue_weekly_2026-10-21_to_2026-10-27/README.md)。

- [四态比较器诊断记录（2026-10-27）](revalue_four_state_comparator_diagnosis_2026-10-27.md)：QV / TVSum 的状态分布、双向胜负、回看触发和待验证原因；基于既有记录，未新增实验模型调用。

- [TVSum 离线诊断 v1（2026-10-27）](revalue_tvsum_offline_diagnosis_v1_2026-10-27.md)：零新增调用；400 份旧比较、448 条事实来源核验；分解草稿遗漏与容量驱逐，19 项 CPU 测试通过。[TaskSpec 单变量协议](revalue_tvsum_taskspec_protocol_v1_2026-10-27.md)仅准备，400 次/2,457,600 token/1 GPUh 为待授权独立建议。

- [ReVALUE / TVSum val10 跨数据集结果（2026-10-27）](revalue_tvsum_transfer_results_2026-10-27.md)：四组、670 次完成；B0 61.32、P_pair 58.38，C_direct4/E_memory 与 B0 逐项相同。含逐视频结果、同分敏感性、成本与证据问题；[冻结协议](revalue_tvsum_transfer_protocol_2026-10-27.md)。

- [阶段实验成果与问题总结（2026-10-27）](experimental_progress_and_issues_2026-10-27.md)：全量候选复评收益、ReVALUE 记忆与回看结果、跨数据集成绩、可靠性瓶颈和下一步顺序。

- [SVHighlights 七场结果](reze_svhighlights_pilot7_results_2026-10-27.md)：补齐已于 10 月 23 日完成的原方法测试，HL-mAP 16.27；输入与预算限制单列。

- [ReVALUE 定向物体核查结果（2026-10-27）](revalue_object_presence_results_2026-10-27.md)：60/60 次完成，两臂状态完全相同；14 个明确正例全部保留，但六个模糊例均未弃答，内容关未通过。见[冻结协议](revalue_object_presence_protocol_2026-10-26.md)。

- [ReVALUE 来源压缩与观察输出探针结果（2026-10-26）](revalue_memory_compaction_results_2026-10-26.md)：14 视图节省 25.30% token，累计补回 104 个旧事实 ID；37/37 输出对照完成，135 项 CPU 测试通过。句子完整性改善但观察内容未通过，暂停 A/B/C 比较。见[CPU 协议](revalue_memory_compaction_protocol_2026-10-26.md)及[输出探针协议](revalue_observer_output_probe_protocol_2026-10-26.md)。

- [ReVALUE 下一轮实验计划：来源压缩与旧证据保留（2026-10-26）](revalue_next_experiment_plan_2026-10-26.md)：保留原设计快照；本轮已完成 CPU 容量与 37 次输出探针，后续审查将内容关前置于 A/B/C 比较，实际决定见最新结果。

- [ReVALUE 来源绑定修复与单帧验证（2026-10-26）](revalue_frame_grounding_repair_results_2026-10-26.md)：672/672 次完成，86 项测试通过；来源接口和空事实过滤已修复，单帧观察仍为诊断用途，尚无新 HD 指标。见[冻结协议](revalue_frame_grounding_repair_protocol_2026-10-26.md)。

- [ReVALUE 离线诊断与 28 次观察对照结果（2026-10-26）](revalue_r2_observation_probe_results_2026-10-26.md)：已执行下一步，技术检查通过，查询盲观察仍有来源/时间错配，未晋级评分重跑；[逐案例诊断](revalue_r2_offline_diagnosis_2026-10-26.md)、[冻结协议](revalue_r2_observation_probe_protocol_2026-10-26.md)。

- [ReVALUE 现阶段实验方案（2026-10-26）](revalue_current_experiment_plan_2026-10-26.md)：上午整理快照；完整框架、R2 结果、R3 功能验收与扩大条件。后续已完成诊断，下一轮细化安排见本页顶部。

- [ReVALUE R2 修复启动记录（2026-10-26）](revalue_r2_repair_execution_2026-10-26.md)：保留启动快照；现已完成 100/100、零技术回退、7 条配对观察，精度未过门槛，结果见上方方案。[诊断](revalue_r2_diagnosis_2026-10-26.md)、[修复协议](revalue_r2_repair_protocol_v3_2026-10-26.md)。

- [ReVALUE R2 旧轮启动记录（2026-10-25）](revalue_r2_execution_2026-10-25.md)：v2 启动快照；该轮现已完成，问题与修复见上方新记录。

- [ReVALUE R1 结果（2026-10-25）](revalue_r1_results_2026-10-25.md)：20/20 完成；决策记忆与直接比较持平，尚未证明记忆收益。

- [ReVALUE R2 执行协议（2026-10-25）](revalue_r2_protocol_2026-10-25.md)：100 条、五条件、配对观察、940 次缓存核验与共享多卡预算。

- [GPU 使用约定（2026-10-25）](gpu_resource_policy_2026-10-25.md)：后续新运行使用任意空闲卡，可拆分任务优先并行；不修改现有冻结运行。

- [ReVALUE R0 故障修复与验证（2026-10-25）](revalue_r0_repair_results_2026-10-25.md)：v1 故障根因、紧凑输出、引用约束与显存修复；当时 37 项测试与原 20 条检查通过，v5 修正全 unknown 退化；R1 启动段落为历史快照。

- [工程修复协议 v2](revalue_r0_repair_protocol_v2_2026-10-25.md)：160 次严格复用、140 次新验证及不重置的累计预算。

- [ReVALUE R0/R1 启动记录（2026-10-25）](revalue_r01_execution_2026-10-25.md)：保留 v1 准备快照，当前状态见上方修复报告。

- [QFVS 全量 180 查询结果（2026-10-25）](reze_qfvs_full_results_2026-10-25.md)：两组全量完成；180/176 查询五组对照、成本和结论边界。

- [ReVALUE 执行方案 v3（2026-10-23）](revalue_execution_plan_v3_2026-10-23.md)：总体设计基线；对齐原版图解的五类价值、两层记忆、四态比较、四类补看、事件选择和完整边界。原预算、门槛与冻结内容保留，当前顺序见上方现阶段方案。

- [QFVS 全量验证执行记录（2026-10-23）](reze_qfvs_full_validation_2026-10-23.md)：全部 180 查询的冻结配置、双卡分工、五组对照、176 条单独报告与自动恢复。

- [下一阶段执行记录（2026-10-23）](reze_next_stage_execution_status_2026-10-23.md)：前一轮启动快照；QFVS 四查询、SVHighlights 七场及 QV V1 已完成，Agent 四组尚未运行。

- [历史执行方案 v2（2026-10-23）](reze_next_stage_execution_plan_v2_2026-10-23.md)：保留原实验引用；未启动的 Agent 部分由 v3 替代，文中的进度和五类价值待定描述属于历史快照。

- [数据集特点、输入输出与评价指标](datasets_inputs_outputs_metrics_2026-10-22.md)：六个数据集的任务区别、标注含义、当前适配协议与实验范围。

- [QFVS 概念提示与时间选择优化](reze_qfvs_adaptation_2026-10-22.md)：固定四查询的配对消融；原分数时间选择 F1 为 40.41，新提示运行与 GPU 迁移见最新执行记录。

更新日期：2026-10-27。R2 观察组仍为 mAP / Hit@1 44.41 / 71.00。最新完成 60 次定向物体核查：两臂均与助手参考一致 24/30，未通过内容关。此前压缩容量通过；尚无新 HD 指标，未启动 A/B/C、100 条复核或 R3/R4。历史计划与启动快照按原日期保留。

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

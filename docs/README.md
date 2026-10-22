# 文档索引

更新日期：2026-10-22。本页列出已发布到仓库的结果与文档。

[实验结果汇总报告（2026-10-22）](experimental_results_report_2026-10-22.md)：汇总 QVHighlights 完整结果、MERIT 开发实验、VideoARM 部分结果，以及 REZE 在 TVSum、MR.HiSum、QFVS 上的迁移测试，含耗时、适配判断和证据入口。尚未发布的支持资料在报告中标注为本地资料。

## REZE 实验方案

- [完整执行方案 v1](reze_execution_plan_2026-10-19.md)：A 阶段候选比较、B 阶段证据记忆、C 阶段定向回看，以及确认集、预算和推进门槛。
- [研究方向与设计依据](reze_research_directions_2026-10-19.md)：已有结果诊断、候选方向、简单对照与贡献边界。
- [冻结参数清单](../comparisons/qvhighlights/reze_fullval_2026-10-20/stage_a/plan.json)、[文档发布清单](reze_plan_publication_2026-10-20.json)：原始方案参数及来源校验和。

方案制定于 2026-10-19；公开版本文首另列截至 2026-10-20 的执行状态。A 阶段及单独授权的全量评分扩展已完成；A 阶段未通过原门槛，B/C 与 confirm100 尚未执行。

## 最新结果

- [REZE 基线与高光重排：完整 validation](reze_fullval_results_2026-10-20.md)：1,550 查询；B0 / P_abs / P_pair 的 VeryGood mAP 为 **42.99 / 43.57 / 43.76**，Hit@1 为 **71.68 / 74.06 / 76.84**。同时整理 CLIP、VTG-GPT 对照和此前 A 阶段停止结论。
- [结果包与复算说明](../comparisons/qvhighlights/reze_fullval_2026-10-20/README.md)：预测、逐查询比较、成本、冻结代码与来源哈希；可用 CPU 复算。

## MERIT、LongVT 与 VideoARM 历史资料

- [复现实验结果总汇（2026-10-19）](reproduction_results_summary_2026-10-19.md)：此前 MERIT、LongVT、VideoARM 的完整验证、成功子集与开发实验；不同样本范围分开讨论。
- [MERIT 在 QVHighlights 上的适配](merit_qvhighlights_adaptation.md)：原始三键记忆检索、完整时间轴评分和边界。
- [统一分析](../comparisons/qvhighlights/analysis_2026-10-15/README.md)、[历史实验归档](../comparisons/qvhighlights/README.md)：预测、配对诊断与初始化审计。
- [后续方向摘要](research_directions_summary_2026-10-15.md)、[研究方案详版](research_directions_2026-10-15.md)：2026-10-16 时的方案记录，不能代替后续完成报告。
- [原 MERIT 执行快照](execution_status.md)、[原自动恢复说明](automation.md)：历史运行及配置说明，不表示当前任务仍在运行。

本地研究工作区还保留未随本次上传的开发记录。以各结果包声明的文件范围和复算能力为准。

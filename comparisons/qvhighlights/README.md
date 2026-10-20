# QVHighlights 实验结果归档

最新结果：[REZE 全量高光重排（2026-10-20）](reze_fullval_2026-10-20/README.md)。包含三组完整 val 预测、配对区间、调用成本、冻结代码、CLIP / VTG-GPT 对照以及 A 阶段原停止结论；附 CPU 复算入口。

历史分析：[MERIT、LongVT 与 VideoARM 的统一评测与配对比较](analysis_2026-10-15/README.md)。包含 MERIT 完整结果、VideoARM 同样本子集评分，以及源视频聚类 bootstrap；原快照保持不变。

- [MERIT 最终预测与报告](merit_main_2026-10-15/README.md)：完整 val 1,550 条，预测和报告保持原始字节。
- [VideoARM 初始化审计](analysis_2026-10-15/videoarm_initial_audit.json)：419 条成功记录的初始化与最终输出比较，含双方成功的 413 条子集。
- [后续研究报告](../../docs/research_directions_summary_2026-10-15.md)、[详版方案](../../docs/research_directions_2026-10-15.md)：更新至 2026-10-16。
- [本次公开导出说明](analysis_2026-10-15/PUBLICATION.md)：来源校验和、公开文件校验及可复算范围。

当前快照：[ 2026-10-15_194620 ](2026-10-15_194620/README.md)。

包含 LONGVT 两套完整 QVHighlights HD 验证协议（Global/Tool 共四组），以及 VideoARM 的 28 个历史运行。完整结果、部分结果和开发诊断分开登记，附预测、报告、配置、账本、逐 query 对齐表及 SHA-256 来源清单。

优先查看快照中的 `comparison_summary.csv` 与 `README.md`。VideoARM HD 验证集只有 419 条成功预测，尚不能作为全量成绩与 MERIT/LONGVT 直接比较。

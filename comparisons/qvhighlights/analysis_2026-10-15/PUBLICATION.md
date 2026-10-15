# 2026-10-16 报告与分析公开导出

本次导出两份研究报告、统一比较与时间分析、VideoARM 初始化审计，以及 MERIT 完整验证预测和最终报告。全部来自既有结果；没有新增模型推理、训练或恢复暂停实验。此前的 `2026-10-15_194620` 历史归档保持不变。

## 文件身份

- [publication.json](publication.json) 同时记录来源文件 SHA-256 与本次公开文件 SHA-256。机器目录使用仓库相对位置或 `source_runs/merit_qvh_main_v1-15f5f6e4ee00` 来源别名。
- `analysis.json` 的 `input_sha256`、`timing_comparison.json` 的 `inputs_sha256`、`videoarm_initial_audit.json` 的 `input_sha256` 仍表示**原始分析使用的输入哈希**。部分输入在上次公开导出时替换过路径，所以这些历史哈希不保证等于公开文件哈希。
- `merit_query_sha256.json` 指向保留在原实验中的 1,550 份查询记录；本次未导出完整查询日志，不应将哈希清单误认为日志本身。
- MERIT 的 `val_predictions.jsonl`、`val_report.json` 和三个逐查询 CSV 保持原始字节。统计量、预测、标签、状态和预算没有因公开导出而改变。
- `analyze.py` 和 `videoarm_initial_audit.py` 保持原始字节。`timing_analyze.py` 仅将固定机器路径改为必填 `--merit-run` 参数；计时计算逻辑不变。
- 文档添加公开入口、最终结果链接及可复算范围说明；两份研究报告的实验方案保持原意。

## 可复算范围

完整 MERIT HD 指标可由本次发布的 1,550 条预测与归档的官方标签、评估器复算。VideoARM 初始化审计只需要仓库内的压缩 SQLite、预测、标签和分析文件，运行方式见 [README](README.md)。

完整配对诊断脚本还读取原 MERIT 查询状态和协议；原始计时脚本还读取记忆、请求和运行日志。这些缓存未公开，需要自行提供兼容的原始运行目录。运行审计脚本会生成当前输入对应的新哈希，不应把它与历史来源哈希混淆。

公开导出检查记录在 [publication_verification.json](publication_verification.json)。检查包括预测与统计内容保真、相对链接、初始化审计复算，以及 MERIT 官方指标复算；这些检查不代表重新完成 VideoARM 全量实验或新方案实验。

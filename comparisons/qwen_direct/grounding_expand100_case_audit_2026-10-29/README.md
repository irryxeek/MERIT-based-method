# QV Direct Grounding：52 条反例审核的精选发布材料

对应[完整分析报告](../../../docs/qwen_direct_grounding_52_counterexample_analysis_2026-10-29.md)与[阶段总结](../../../docs/progress_after_meeting_2026-10-27_to_2026-10-30.md)。这是基于原始输入帧的助手初筛，不是独立人工标注，不改变发布 GT、原预测或指标。

## 已发布

- [52 条逐例 CSV](all52_case_reviews.csv)与[详细 JSON](final_case_reviews.json)：22 条较明确定位错误、15 条标注／范围疑点、10 条两者并存、5 条证据不足。
- [原 CPU 验证收据的路径别名副本](validation_receipt.json)、[单独裁决记录](adjudications.json)。内嵌历史哈希仍指原始材料；发布文件的实际哈希另见发布清单。
- 五组代表性原帧：[鸡舍错位 qid448](evidence_cards/qv_448.jpg)、[近景／远景错位 qid4703](evidence_cards/qv_4703.jpg)、[圣诞树重复实例 qid3676](evidence_cards/qv_3676.jpg)、[早餐范围 qid2603](evidence_cards/qv_2603.jpg)、[软件边界混合问题 qid5466](evidence_cards/qv_5466.jpg)。

完整 52 条的 64 帧拼图、原视频、原运行输入及其他归档保留在研究工作区。公开副本使用路径别名标明来源，不代表仓库包含相应机器目录；没有新增采样或实验模型调用。

来源与公开副本 SHA-256 见[报告发布清单](../../../docs/report_publication_manifest.json)。

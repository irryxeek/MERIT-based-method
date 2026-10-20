# REZE 固定评分方案：全量验证结果

完成时间：2026-10-19T20:14:44.690953+00:00

QVHighlights 验证集全部 1,550 条查询、474 个源视频。复用冻结 B0，并列报告 P_abs / P_pair。
此前 explore100 未通过 A 阶段门槛；本轮是用户要求的扩大验证，不改变该历史结论。验证集此前已查看。

| 方法 | VeryGood mAP | Hit@1 | 含 B0 总 token 比 | 回退查询 |
|---|---:|---:|---:|---:|
| B0 | 42.99 | 71.68 | 1.000× | 0 |
| P_abs | 43.57 | 74.06 | 1.547× | 0 |
| P_pair | 43.76 | 76.84 | 1.553× | 0 |

## 配对差异

- P_abs_minus_B0：mAP +0.582 pp，95% CI [0.289, 0.879]；Hit@1 +2.387 pp，95% CI [0.581, 4.177]。
- P_pair_minus_B0：mAP +0.771 pp，95% CI [0.479, 1.064]；Hit@1 +5.161 pp，95% CI [3.488, 6.863]。
- P_pair_minus_P_abs：mAP +0.189 pp，95% CI [-0.059, 0.442]；Hit@1 +2.774 pp，95% CI [1.307, 4.285]。

源视频聚类配对 bootstrap：10,000 次，seed=42。所有查询保留在分母中。

## 开销与边界

新增调用（含重试预留）：62,000；实际分配 GPU 小时：3.868。
完整运行墙钟时间：3.884 小时。复用 B0 的历史运行时间单独记录。
两组共享批次，分摊时间不代表独立延迟。只置换原前五候选分数；并列分数与候选覆盖仍是限制。
未训练、未引入 memory 或回看；本轮结束后不自动扩展实验。

## 结果文件

- 完整指标、开销与哈希：`source_runs/qvhighlights-zeroshot/reze_evidence_fullval_v1/results/evaluation.json`
- 逐查询比较：`source_runs/qvhighlights-zeroshot/reze_evidence_fullval_v1/results/per_query.jsonl`
- 原始输出、帧证据与日志：`source_runs/qvhighlights-zeroshot/reze_evidence_fullval_v1/runs/val1550_pairs`

# LongVT-RFT 在 QVHighlights 上的 GroundVTS 启发式迁移推理报告

## 结论与状态

冻结配置下的 Global 和 Tool 验证推理均已完成，各覆盖 1,550 个唯一 query。两种模式的预测均通过原生 schema 和向量检查，fallback 为 0，query 与帧覆盖完全配对。官方评分、10,000 次 source-video cluster bootstrap、配对审计和工具证据覆盖分析均已完成。

结果不支持“加入工具后整体指标确定提升”的结论：Tool 的 mAP 点估计略高，但三个阈值的配对置信区间都跨过 0；Hit@1 点估计在三个阈值都下降，其中 Good 的区间完全低于 0。Tool 还使用了更多视觉帧和 token，因此这不是等预算比较。

这是冻结 LongVT-RFT 的训练免费推理适配，借鉴了 GroundVTS 的分级时间片表达；没有实现 GroundVTS 的可训练 VTS 模块或渐进式 SFT，不能称为 GroundVTS 复现。v2b 用合法的两秒 clip-grid span 代替连续秒数，是在开发阶段 schema 问题后冻结的协议修订，结果只支持对该协议的结论。

## 冻结协议与数据完整性

- 基础模型：`longvideotool/LongVT-RFT`，revision `16ed9314e4ea600dc4682b172a614252f2f383ad`，BF16、TP=2、65,536 context；未训练或微调模型。
- Global 使用 64 个均匀采样帧，不调用工具。Tool 先看相同的全局帧，再至多两次调用 `crop_video`，每次最多 32 帧。clip span 按 QVHighlights 官方两秒 clip 索引编码，级别映射为 4/3/2。
- manifest、Global traces、Tool traces 和两份 predictions 均覆盖 1,550 个唯一 qid，集合完全一致。配对评分确认 1,550 个 query 的帧身份一致；35 个 query 的两种预测向量完全相同。
- Global：1,550/1,550 原生有效，0 fallback。Tool：1,550/1,550 原生有效，0 fallback；状态为 1,443 `success` 和 107 `success_with_tool_failure`。
- 工具调用共 1,541 次，涉及 1,534 个 query；1,434 次成功，107 次由保护逻辑拒绝。其中 106 次请求超出当前官方局部视频时长，1 次（qid 6658）提交了与当前 query 不匹配的视频路径，被路径校验拦截。拒绝只影响该次额外 crop；最终预测仍有效。它们不是媒体读取或 API 基础设施故障。

## 官方验证集指标

数值为官方 evaluator 输出的百分比；Δ 为 Tool − Global，单位是百分点。

| 相关性阈值 | Global mAP / Hit@1 | Tool mAP / Hit@1 | Δ mAP / Δ Hit@1 |
|---|---:|---:|---:|
| Fair | 32.93 / 37.87 | 33.38 / 35.81 | +0.45 / −2.06 |
| Good | 26.18 / 36.39 | 26.52 / 33.81 | +0.34 / −2.58 |
| VeryGood | 14.94 / 28.45 | 15.12 / 27.23 | +0.18 / −1.22 |

## 配对 source-video bootstrap

对每个 source video 做有放回 cluster 抽样，重复 10,000 次，seed 42；按 query 加权，474 个 source video。表中点估计是 paired query-level mean delta，区间为 95% bootstrap CI。

| 阈值 | 指标 | 配对 Δ | 95% cluster CI |
|---|---|---:|---:|
| Fair | mAP | +0.44 | [−0.44, +1.32] |
| Fair | Hit@1 | −2.06 | [−4.56, +0.45] |
| Good | mAP | +0.34 | [−0.40, +1.10] |
| Good | Hit@1 | −2.58 | [−5.11, −0.07] |
| VeryGood | mAP | +0.19 | [−0.31, +0.68] |
| VeryGood | Hit@1 | −1.23 | [−3.50, +1.03] |

因此 mAP 的小幅正向点差异不确定；Good Hit@1 则显示 Tool 低于 Global 的区间证据。配对 audit 另按官方 tie 规则复算 Hit@1 和 mAP；其 tie-break diagnostics 不是官方指标，单独保存在 audit 文件中。

## 工具证据和额外成本

- 1,534 个 query 请求过工具；按“官方 clip 中点落入成功执行 crop”的审计规则，902/1,550 个 query 至少覆盖一个 released relevant clip。按阈值统计，至少覆盖一个正例 clip 的比例为 Fair 901/1,534（58.7%）、Good 887/1,533（57.9%）、VeryGood 834/1,500（55.6%）；平均正例 clip recall 分别为 31.1%、31.2%、31.8%。这些只是 crop evidence coverage，不代表下游预测质量。
- 两种模式均读取 99,200 个全局帧；Tool 另读取 25,894 个局部帧，总读取帧数 125,094，比 Global 多 26.1%。
- Tool prompt tokens 为 26.46M，Global 为 14.67M（+80.4%）；completion tokens 为 1.11M 对 0.76M（+45.8%）。每 query 平均耗时为 14.43 秒对 9.26 秒（+55.9%），p95 为 21.71 秒对 18.92 秒。
- Tool 将 all-zero score vectors 从 Global 的 47 个降到 13 个，constant vectors 从 130 个降到 55 个；这些诊断没有改变官方评测定义，也不能单独说明模式优劣。

## 解释边界

QVHighlights validation 的标签在旧实验中被查看过，因此不是完全未触碰的测试集。v2b 的离散 clip span 量化了连续时间定位，且 Tool 获得额外视觉预算。结果适合检验 GroundVTS 启发式的 frozen-base 工程迁移，不等价于同预算比较，也不足以声称它复现了 GroundVTS 或普遍提升长视频理解。

## 产物

- 推理、协议、恢复方法及历史状态：[README.md](README.md)、[PROGRESS.md](PROGRESS.md)、[PLAN.md](PLAN.md)
- Global 指标：`results/full_v2b/longvt_global/metrics.json`
- Tool 指标：`results/full_v2b/longvt_tool/metrics.json`
- 配对指标与 bootstrap：`analysis/full_v2b/paired_summary.json`
- 完整性、tie 与 tool evidence audit：`analysis/full_v2b/paired_audit.json`
- 逐 query 诊断：`analysis/full_v2b/paired_diagnostics.csv`
- 两种模式的预测和 trace：`results/full_v2b/longvt_global/`、`results/full_v2b/longvt_tool/`

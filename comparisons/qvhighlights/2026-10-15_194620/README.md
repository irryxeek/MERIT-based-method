# QVHighlights 历史实验对比归档

归档时间：2026-10-15T19:46:20+08:00。本目录保存用于 MERIT 对比的历史结果，约 133.6 MiB。本目录为该本地快照的公开副本，没有重新推理或恢复暂停任务。

## 快速入口

- [机器可读汇总](comparison_summary.json)、[汇总 CSV](comparison_summary.csv)：完整 LONGVT HD 结果与 VideoARM 正式运行状态。
- [逐查询对齐表](alignment/query_alignment.csv)：1,550 个验证查询、视频身份、四组 LONGVT 的状态/回退标记，以及 VideoARM 的 HD/MR 状态。
- [VideoARM 全部运行索引](videoarm_run_index.json)：28 个历史运行的任务、协议、模型、覆盖率及已有评分。
- [来源清单](manifest.json)：480 份来源产物的示例路径、原始与公开版本 SHA-256、归档位置和压缩方式。
- [核验记录](verification.json)、[文件校验和](SHA256SUMS)：文件完整性、预测与账本一致性、官方指标复算。

## 完整验证集高光检测结果

下表使用相同的 QVHighlights validation 1,550 查询、1,519 视频和官方评估器提交 `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`。单位为百分数，主指标为 VeryGood。

| 方法与协议 | 预测覆盖 | 原生有效 / 零分回退 | mAP | Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| LONGVT compact v1 / Global 64 | 1,550/1,550 | 1,534 / 16 | 13.90 | 20.65 |
| LONGVT compact v1 / Tool | 1,550/1,550 | 1,522 / 28 | 14.60 | 22.71 |
| LONGVT GroundVTS 启发式 v2b / Global 64 | 1,550/1,550 | 1,550 / 0 | 14.94 | 28.45 |
| LONGVT GroundVTS 启发式 v2b / Tool | 1,550/1,550 | 1,550 / 0 | 15.12 | 27.23 |

四组指标已由归档内的官方评估代码重新计算，与历史文件一致。四份预测共 6,200 条，并与 trace 中的分数逐条匹配。GroundVTS 启发式实验使用冻结的 LongVT-RFT，没有实现 GroundVTS 的训练模块，不能称为 GroundVTS 复现。

## VideoARM 的覆盖范围

- **HD 验证集 M1**：计划 1,550 条，成功 419、失败 34、暂停 4、待处理 1,093。状态为用户暂停，归档时没有对应 `scores.json`，因此不填全量 mAP/Hit@1，也没有为未完成样本补造预测。
- **MR 验证集 M1**：1,550 条均进入终态，成功 1,113、失败 437。已有官方评分仅覆盖成功子集；原计划分母的“失败指标记零”是另列的自定义口径，不能混作完整官方提交。
- 另外保留 26 个 smoke/dev 历史运行。v1 含已登记的 stop 接口问题；v1/v2、开发集/验证集、MR/HD 必须分别解释。

HD 的 419 条已有预测全部可与四组 LONGVT 预测对齐：[共同可用 qid](alignment/common_available_hd_qids.json)。其中 405 条在所有方法中均为原生有效：[共同成功 qid](alignment/common_native_success_hd_qids.json)。两者均为选择后的诊断子集，不能替代完整验证集对比。

## 目录与保存内容

- `longvt/compact_v1/`：原报告、冻结配置、清单、Global/Tool 预测、官方指标、逐题差异、配对审计和原始响应轨迹。
- `longvt/groundvts_v2b/`：第二套完整协议的配置、预测、指标、bootstrap 与工具证据分析。
- `videoarm/reproduction/qvhighlights/`：原报告、协议修订、数据与模型审计、清单、环境记录和相关源码快照。
- `videoarm/runs/<run_id>/`：预测、已有评分、运行身份、进度、逐 query 状态，以及压缩后的 SQLite 请求/响应/预算账本。
- `common/`：官方 validation 标签、选定训练开发查询的标签和官方 evaluator 源码。两套 LONGVT 标签文件虽字节格式不同，但解析后与官方标签逐项一致。

`traces.jsonl.gz`、`attempts.jsonl.gz` 在路径脱敏后无损压缩；`state.sqlite.gz` 在一致性备份上替换路径，保留完整 items、requests 和 metadata 表，并重新检查数据库完整性。SQLite 每阶段记录代替重复保存 VideoARM trace 文件。模型权重、源视频、采样图片、虚拟环境和凭据未复制；原始来源全部保留。没有纳入其他数据集实验、未完成的 LONGVT LoRA 训练或 LONGVT 开发试跑预测。

公开副本保留当时的状态描述，并将本机路径、GPU UUID 和主机名替换为示例值。报告中指向未压缩 trace 的路径，在本归档内对应 `.gz` 文件；原始文件身份由 `manifest.json` 的 `source_sha256` 保留，公开文件使用单独的 `archive_sha256` 与 `public_uncompressed_sha256`。它们是历史证据，不能作为当前任务的启动指令。

## 比较时的限制

LONGVT 使用 `longvideotool/LongVT-RFT`；VideoARM 使用 `Qwen3-VL-8B-Instruct`，后者与当前 MERIT 的模型及 revision 一致。但三者的抽帧、像素大小、提示词、工具预算、记忆和评分表达均不同，因此分数差异不能归因于单独的检索模块。LONGVT 两套协议也应分别报告，不能事后挑较高者代表统一基线。

MERIT/LONGVT 的失败采用零分**预测向量**再走官方排序指标；VideoARM 的自定义口径是缺失项**指标记零**。这两种处理不等价。MR 的区间检索指标也不能与 HD 的 VeryGood mAP/Hit@1 混列。本归档未纳入仍在运行的 MERIT 中间预测，后续应使用完整冻结结果。

## 校验与读取

在本快照目录执行：

```bash
sha256sum -c SHA256SUMS
# 使用已安装 numpy、scikit-learn 的 Python；默认只读核验并复算四组 LONGVT 指标：
python tools/verify_archive.py
```

`--write-reports` 会明确重新生成汇总、对齐表和核验记录，此后需相应更新校验和清单；日常核验不需要该参数。读取压缩文件可使用 Python `gzip.open`；如需查询 SQLite，请先解压到临时目录。

## 公开导出记录

[publication.json](publication.json) 记录公开导出范围及账本核验。原始本地归档保持不变。所有逐查询预测文件与本地归档逐字节一致；已评分运行的指标与覆盖率保持一致。公开账本只替换路径和机器标识，行数、预测与预算数值保持不变。原始请求身份和冻结源码 SHA 指向脱敏前的记录，不能用它们直接恢复公开示例路径下的推理缓存。

`verification.json` 保留本地归档阶段的历史核验；`publication_verification.json` 记录本次公开导出核验。公开副本不支持 `--check-sources`，因为机器路径已替换。历史源码和报告用于解释结果，其中指向原机器的命令需要重新配置后才能使用。

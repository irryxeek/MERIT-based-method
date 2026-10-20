# REZE 全量验证结果公开归档

发布日期：2026-10-20。正式解读见[最新实验报告](../../../docs/reze_fullval_results_2026-10-20.md)。

完整 validation：1,550 查询、1,519 视频片段、474 源视频。VeryGood mAP / Hit@1：B0 **42.99 / 71.68**，P_abs **43.57 / 74.06**，P_pair **43.76 / 76.84**。各组均无查询回退。本次仅整理已完成结果，没有新增推理。

## 文件布局

| 目录或文件 | 内容 |
|---|---|
| [fullval/results/](fullval/results/) | 三组完整预测、官方指标、逐查询比较、原自动报告 |
| [fullval/run/](fullval/run/) | 冻结计划、输入绑定、完成凭证、调用账本、代码快照 |
| [fullval/audit/](fullval/audit/) | 模型与视频输入哈希、历史工程检查记录 |
| [stage_a/](stage_a/) | 固定训练 explore100 的原结果、预测、成本与原始计划；未通过继续门槛 |
| [references/](references/) | CLIP / VTG-GPT 全量预测与指标，以及 REZE B0 完成、环境、资源记录 |
| [publication.json](publication.json) | 每个导出文件的原始 SHA-256、公开 SHA-256 与转换说明 |
| [publication_verification.json](publication_verification.json) | 本次发布后的 CPU 核验记录 |

`fullval/results/REPORT.md` 与 `stage_a/REPORT*.md` 保留当时自动生成或复核后的文字，仅转换机器路径。它们的“尚未运行”“停止”等表述属于相应时间点；以最新实验报告解释后续全量扩展。自动全量报告将调用记作“含重试预留”，实际账本为 62,000 次预留且全部返回，上限 68,200 次，未使用重试余量。

## 无 GPU 复算

从仓库根目录执行，Python 3.10 或更高：

```bash
python3 -m venv .venv-eval
.venv-eval/bin/python -m pip install \
  -r comparisons/qvhighlights/reze_fullval_2026-10-20/requirements-eval.txt
.venv-eval/bin/python \
  comparisons/qvhighlights/reze_fullval_2026-10-20/verify_results.py
```

默认只读取文件，JSON 检查结果输出到标准输出。使用 `--output /path/to/new_receipt.json` 可另存新凭证；不会覆盖已有文件。脚本核验：

- 公开清单中全部文件的哈希；官方 validation 标签、评估器与原实验哈希一致。
- B0、P_abs、P_pair、CLIP、VTG-GPT 各 1,550 条唯一查询，身份、分数长度与有限值合法。
- 五组 Fair / Good / VeryGood 官方 HD 指标；三组 REZE 逐查询指标及三项源视频聚类配对区间。
- 前五候选外分数不变、分数多重集不变、并列与实际变化数量。
- 完成凭证、调用预留和返回账本、token 汇总、GPU 小时，以及 A 阶段原停止结论。

脚本不重跑视觉模型，不重新生成 caption 或记忆，也不声称重新审计未上传的逐请求 logits、输入像素或所有历史工程测试。A 阶段预测已保留，但本脚本只校验其文件完整性和原停止决定，不重算训练集指标。

## 来源、路径与边界

导出前逐一核验原全量归档收据中的 40 个文件。67 个来源文件的导出映射见 `publication.json`；其中 47 个保持原始字节，包括全部预测、机器可读指标和逐查询分数。机器路径、设备标识和指向原机器的文档链接作公开转换，统计数值与实验逻辑不变；原始文件未修改。

`source_runs/`、`source_workspaces/`、`source_data/`、`source_models/`、`source_dependencies/`、`source_environment/` 是原实验来源别名，不是仓库提供的目录。代码快照与 systemd 单元用于追溯实现；其中路径尚未参数化，不应作为可直接恢复运行的入口。

`source_sha256` 标识原文件，`public_sha256` 校验本包文件。`binding.json`、`plan.json`、`completion.json` 及原 `archive_receipt.json` 内嵌哈希仍指向原始字节；发生路径转换的文件应通过公开清单核验。`record_receipt.json` 是保留在实验存储中的原始查询记录哈希，不表示记录正文已上传。

官方 validation 标签与评估器复用[此前归档](../2026-10-15_194620/common/)；本次不重复复制。全量预测的 B0 SHA-256 为 `af8efe26f9cd3966e5d3c6b306d87913af8f38f096035bf891df8fa864063452`，与历史基线一致。不同方法的模型、预计算和硬件预算不统一，不能由本包推断同预算速度排名或记忆机制的因果收益。

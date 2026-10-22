# REZE 在 MR.HiSum 上的迁移测试

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

日期：2026-10-21。固定 100 条官方 test 视频，使用与 QVHighlights B0 相同的 Qwen3-VL-8B-Instruct 和 REZE 评分器。完成结果以试跑入口（本地资料：`reze_mrhisum_pilot100.local.json`）指向的 `status.json`、`evaluation.json` 为准。后续已启动[两卡全量可用集实验](reze_mrhisum_full_execution_2026-10-21.md)，两轮记录独立保存。

**100/100 条推理与评价已完成。** GPU 7 推理于 10:48–11:14 CST 运行，单次 worker 耗时 26.04 分钟，无推理重试。自动评价的模块导入错误已修复；12:33 CST 使用原冻结评价代码补算完成，未重跑模型或改变预测。

## 实验结果

以下均为固定 100 条样本上的百分数。

| 方法或对照 | mAP50 | mAP15 | 摘要 F1 |
|---|---:|---:|---:|
| REZE + Qwen3-VL-8B-Instruct | **58.33** | **23.97** | 39.21 |
| 常数评分 | 49.40 | 13.76 | 56.83 |
| 均匀随机排序的精确期望 | 53.67 | 21.04 | — |

REZE 相对随机排序期望的 mAP50 / mAP15 分别高 **4.66 / 2.92 个百分点**，视频级配对 bootstrap 95% 区间分别为 [2.33, 6.99] / [0.58, 5.38]。这批样本上存在高光排序信号；摘要 F1 比常数评分低 17.62 个百分点，尚不能说明摘要选择有效。常数对照仍经过相同的 KTS/背包和并列处理，不能将它视为随机摘要。

REZE 的 mAP50、mAP15、F1 的 95% bootstrap 区间分别为 [55.99, 60.67]、[21.61, 26.42]、[35.33, 42.99]。这些区间只描述当前可下载子集的抽样不确定性，不覆盖视频可用性或历史内容对齐偏差。

正式评分 **6,959 次**，加 9 次输入检查共 **6,968 次**；正式输入 6,129,511 tokens、41,521 帧。模型计算累计 13.84 分钟，解码与预处理累计 11.09 分钟，worker 总耗时 26.04 分钟。全部评分均取得 Yes/No 双侧概率，100 条视频均有非恒定评分。

## 样本与数据

MR.HiSum 补下载已完成：75 条补试中取回 19 条，测试集共取得 **1,891 / 2,000** 条视频。其中 1,888 条通过基础媒体及 ±2 秒时长检查，3 条需进一步对齐核查，109 条未取得。

本轮在上述 1,888 条中，按 `sha256("42:video_id")` 升序固定前 **100 条**。选择只使用官方划分、视频 ID 和媒体检查状态，不使用高光标签或模型结果。总标注时长 **20,779 秒，约 5.77 小时**，单条 121–300 秒。范围固定为这 100 条可用视频，结果单独标注为 pilot。

## 推理与评价

- 模型：`Qwen/Qwen3-VL-8B-Instruct`，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16。
- MR.HiSum 无查询文本，因此所有视频使用固定查询 **`Highlight of this video.`**。保留原 REZE 严格动作核验提示和 Yes/No 的 top-20 概率计算；该查询是本项目的接口适配选择。
- 3 秒窗口、2 fps，完整窗口 6 帧；保留原逐整数秒曲线的边界平均和最近邻填充。实际完成 **6,959 次正式评分请求**，另有 9 次 GPU 输入检查。
- 主指标：官方 **mAP50、mAP15**。先对每 5 秒片段取均值，以真实得分排名前 50% / 15% 的片段为正例，计算每视频 AP 后平均。末尾不足 5 秒的片段保留。
- 次指标：官方 KTS 分段与 15% 背包预算对应的摘要 F1。保留作者 `end-start` 段长约定；按原 F1 分支规则显式补零，再调用完整评价函数，避免其未使用的相关系数分支因长度不同报错。没有修改发布的标签和分段。
- 同样本参考：常数评分的 mAP/F1、均匀随机排序的精确期望 AP；视频级 bootstrap 10,000 次、seed 42。

评分核心、后端和 QV 解码参考文件逐字节保留；官方评价辅助代码固定于 `aeb667f71239887034c2974062a341d312762867`。标签只供独立 CPU 评价进程读取。高光检测与摘要指标分别报告。

这属于原 REZE 评分器的跨数据集迁移测试。固定泛化查询可能限制原动作核验提示的效果；100 条可下载样本的成绩不能代表完整 2,000 条测试集。基础媒体检查也尚未证明历史 YouTube 视频与标签逐秒语义完全对齐。

## 工程检查与恢复

首轮 `reze_mrhisum_pilot100_v1` 的媒体预检查发现：`video_9757` 为 VP9，Decord 能解码，但返回 4,301 个零时长帧区间，导致边界检查失败；模型调用数为 0。原代码、失败日志及时间戳诊断全部保留。

修复后的 `v2` 保留相同 100 条 ID、查询、标签和评分器。帧起始 PTS 不变，缺失的结束时间由下一帧 PTS 补齐，最后一帧由平均帧率推算；记录原始及恢复后的时间戳。没有删除该样本或修改原视频。13 项针对性测试通过，包含实际 VP9 解码、与 QV 输入像素一致性、5 秒读出、原生摘要边界、随机 AP 参考和检查点校验；Ruff 通过。

自动队列从 GPU 6/7 中选择一张无其他计算进程的卡，每批 16 窗口保存原始概率、帧索引与像素哈希。最多 3 次 worker 尝试、2 小时累计运行时间，累计请求上限为正式请求数 +128。GPU 推理退出后自动进行 CPU 评价；完成及停止状态不会自动重跑。

自动评价于 11:14 CST 因导入顺序失败：独立进程在添加 `vendor/` 路径前导入了官方 `model` 模块。此次仅在 CPU 评价进程的 `PYTHONPATH` 中补入冻结的 `source_snapshot/vendor/`，原评价公式、冻结代码、标签和 100 份预测均保持不变；原始失败状态及日志存入 `evaluation_attempts/001_import_failed/`，恢复命令、环境和前后哈希核对存入 `evaluation_attempts/002_pythonpath_recovery/receipt.json`。评价约用时 5 秒，新增模型请求为 0。开发版入口已修复，并增加独立子进程启动的回归测试。

## 文件与命令

实现：`$ZEROSHOT_ROOT/experiments/reze_mrhisum/`。

当前运行：`$ZEROSHOT_ARTIFACT_ROOT/reze_mrhisum_pilot100_v2/`。

关键产物：`binding.json`、`sampling_audit.json`、`inference.jsonl`、`source_snapshot/`、`media_preflight.json`、`gpu_smoke.json`、`predictions/`、`per_video_metrics.jsonl`、`evaluation.json`、`REPORT.md`、`evaluation_attempts/`。

```bash
cat $ZEROSHOT_ARTIFACT_ROOT/reze_mrhisum_pilot100_v2/status.json
systemctl --user status reze-mrhisum-pilot100.service
```

来源：[REZE 附录 A](https://arxiv.org/html/2608.04480v1)、[MR.HiSum 官方代码](https://github.com/MRHiSum/MR.HiSum/tree/aeb667f71239887034c2974062a341d312762867)、[MR.HiSum 论文第 4.3 节](https://proceedings.neurips.cc/paper_files/paper/2023/file/7f880e3a325b06e3601af1384a653038-Paper-Datasets_and_Benchmarks.pdf)。

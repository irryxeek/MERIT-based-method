# 现有数据集与 REZE 跨数据集测试

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

核对日期：2026-10-20。本轮目标是测试原 REZE 评分流程的迁移效果，沿用 QV 实验中的 Qwen3-VL-8B-Instruct。

**2026-10-22 状态补充：** TVSum 全 50 条已完成（固定 val10 / all50 Top-5 mAP 61.32 / 54.89）；MR.HiSum 可用 1,888 条及 QFVS 四查询已完成。SVHighlights 原 7 场任务在三次 GPU UUID 传参错误后停止，没有有效 REZE 指标。下方 17:26 描述是当时的准备阶段快照；最新分析见[跨数据集报告](reze_cross_dataset_results_analysis_2026-10-22.md)。

**17:26 CST 状态：准备与校验完成，自动队列已启用，正在等待空闲 GPU；模型推理尚未启动。** 首、中、尾共 21 个真实窗口通过原生模型输入检查，最大输入 2,280 tokens，低于冻结的 4,096 上限。计划评分 19,014 个窗口，输出 28,518 个评估片段分数。此段为状态快照，实时进度以运行目录的 `status.json` 为准。

## 已确认的数据

| 数据集 | 本地情况 | 本轮处理 |
|---|---|---|
| QVHighlights | 视频、标注和评估代码齐全；REZE 已完成 1,550 条验证查询 | 保留原结果 |
| SVHighlights | 320 场标注，270 场完整裁剪视频；201 场通过既有媒体/标注长度核验，69 场计数不一致，50 场缺源 | 固定 7 场任务启动失败后停止，无有效 REZE 指标 |
| QFVS | P01–P04 原视频已下载并校验，共 16.96 小时；预提取特征、180 个查询和原生摘要标注齐全 | 固定 4 查询已完成，Oracle F1 38.63；见[结果分析](reze_qfvs_analysis_2026-10-20.md) |
| VideoSIAH | 244 个源视频、652 个问题的本地文件齐全 | 问答任务，不纳入本轮高光检测 |
| TVHighlights | 2026-10-21 补充核查：仅论文及补充材料已取得，未找到官方视频/标注下载入口 | 用户原请求的数据集；此前误按 TVSum 下载，见[纠正记录](tvhighlights_download_2026-10-21.md) |
| TVSum | 全部 50 条视频、MAT/TSV 标注已下载；2 条媒体比标注多 1 帧，差异已记录 | 全 50 条已完成；UMT 固定 val10 / all50 Top-5 mAP 61.32 / 54.89；见[执行记录](reze_tvsum_execution_2026-10-22.md) |
| MR.HiSum | 31,892 条官方标注与划分已核对；补试后取得 test 视频 1,891/2,000，其中 1,888 条通过基础检查 | [1,888 条可用集](reze_mrhisum_full_execution_2026-10-21.md)已完成：mAP50 / mAP15 / 摘要 F1 为 57.00 / 23.90 / 38.27；[100 条试跑](reze_mrhisum_execution_2026-10-21.md)单独保留 |
| PatientPheX | 医学表型资料 | 与本轮视频高光检测无关 |

SVHighlights 媒体位于 `/data/linyifei/videoarm-repro/svhighlights/videoarm/full_available_v4_20261009/media/`。原 VideoARM 推理仍保持用户此前要求的暂停状态；本轮使用独立 REZE 任务。

## 固定 7 场测试

从此前 Uniform-Refine AV 与 VideoARM AV 均有完整输出的 20 场开发集视频中，每个可用运动类别取 `sha256("42:vid")` 最小的一场。仅使用预测 ID 和元数据确定名单，不按指标或高光标签挑选。篮球原视频缺失。

| 视频 | 时长/秒 | 2 秒评估片段数 |
|---|---:|---:|
| american_football_36 | 9194.034 | 4597 |
| baseball_13 | 10872.449 | 5436 |
| ice_hockey_8 | 6004.139 | 3002 |
| race_30 | 11357.034 | 5679 |
| rugby_32 | 6146.363 | 3073 |
| soccer_39 | 5832.084 | 2916 |
| volleyball_17 | 7629.080 | 3815 |

总内容时长 **15.84 小时**。本轮仅执行这 7 场完整视频，不自动扩展到全量。

## 方法与必要的接口处理

- 固定模型 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16，无训练。
- 与 QV 原 REZE 基线相同的 3 秒窗口、每完整窗口 6 帧、缩放、严格动作核验提示和 top-20 Yes/No 概率计算；冻结模块逐字节一致。
- 查询采用官方解析代码中的 `Highlight of this {sport} video.`；不增加运动专用 cue、记忆、候选重排或回看。
- 保留原 REZE 的逐秒分数曲线与 2 秒均值。SVHighlights 末尾只剩一个有效整秒时，保留该单秒片段；不填零，不静默截断向量。
- 每批读取 16 个窗口，控制长视频内存。实际 QV 视频对照已验证帧索引、PTS、像素和输入元数据一致。

这是原 REZE 评分流程的**跨数据集迁移测试**，不是 REZE 论文已经报告过的 SVHighlights 结果。

## 评估、自动执行与产物

使用冻结官方代码的 **HL-mAP、HL-Hit1、HL-Hitk、HL-IoU**；计算前严格检查视频 ID 与向量长度。对照只取相同 7 场。既有两个 AV 方法与 REZE 的视觉输入及预算不同，比较时须保留这一区别。[官方指标实现](https://github.com/leedongkyu2019/SVHighlights/blob/a463fafbca8136bc6839bddc91216e494d970833/eval.py)

代码：`$ZEROSHOT_ROOT/experiments/reze_svhighlights/`。

运行目录：`$ZEROSHOT_ARTIFACT_ROOT/reze_svhighlights_pilot7_v1/`。

准备阶段重新计算视频及模型文件指纹，并用真实视频的首、中、尾窗口检查解码和模型输入。9 项针对性测试通过，包含读帧等价性、边界、检查点和 GPU 占用检查。

自动队列仅在 GPU 4–7 中有一张无其他计算任务的空闲卡时启动。每 16 个窗口保存检查点，最多 3 次 worker 尝试、12 小时累计 worker 时间；完成后释放模型并在独立 CPU 进程评分。原始失败和已消耗请求保留。

用户级 `reze-svh-pilot7.timer` 已启用，每分钟检查一次；Linger 已启用，支持退出登录和重启后继续调度。首轮调度正常退出并写入 `waiting_for_idle_gpu`，未占用其他任务所在 GPU。

动态进度见 `status.json`，日志见 `worker_attempt_*.log`。完成后自动生成 `evaluation.json`、`predictions.json` 与 `REPORT.md`。尚未完成时不将部分输出描述为最终指标；7 场结果不能代表官方 320 场成绩。

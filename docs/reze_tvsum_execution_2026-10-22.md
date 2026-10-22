# REZE / TVSum 全量零样本高光检测执行记录

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

记录日期：2026-10-22（北京时间）。本轮继续 MR.HiSum 后的跨数据集测试，选择已完整下载的 **TVSum 50 条视频**。TVSum 与 TVHighlights 是两个数据集；后者仍只有论文及补充资料，见[核查记录](tvhighlights_download_2026-10-21.md)。

## 当前状态

**全部 50 条推理和评价已于 2026-10-22 01:23:35 CST 完成。** 正式评分 4,213 个窗口，输出 6,266 个 2 秒片段分数，无视频缺失。50 条视频、150 个真实窗口在运行前通过 CPU 原生输入检查，最大输入 777 tokens。

10 月 22 日中午已复核完成标记、标签、冻结源码、结果文件哈希和 50 条分片覆盖；详情见[跨数据集分析](reze_cross_dataset_results_analysis_2026-10-22.md)。

## 最终结果

| 范围 | REZE Top-5 mAP | 常数评分 | 随机排序精确期望 |
|---|---:|---:|---:|
| UMT 固定验证集 10 条（主报告） | **61.32** | 49.48 | 42.54 |
| 全 50 条零样本迁移（补充） | **54.89** | 46.56 | 41.32 |

相对随机期望的配对提升分别为 **+18.77 pp [7.47, 28.56]**、**+13.57 pp [9.69, 17.38]**；区间为视频级 bootstrap 95% 区间。REZE 本身的区间分别为 [49.32, 71.53]、[50.83, 58.86]。全 50 条包含 UMT 划为训练集的 40 条，不能与固定 10 条的监督方法结果混用。

全部正式评分都有 Yes/No 双侧概率，无 top-20 缺失回退；没有整条视频常数评分，8 条视频的 top-5 截断处有并列分数。统计及逐视频结果见运行目录 `evaluation.json`、`per_video_metrics.jsonl`。

第一分片 GPU 7 一次完成；第二分片 GPU 4 首次在 smoke 阶段发生 vLLM 多模态 masked-scatter 错误，自动重试后完成。共三次 worker 尝试，正式、smoke 及失败保留请求合计 **4,233** 次，原错误日志未删除。首个 worker 启动至最后一个退出约 **11.81 分钟**，累计 worker 时间 **14.64 分钟**；并非两张卡完全同步启动。正式输入 3,118,668 tokens、25,170 帧。

## 固定范围和方法

| 项目 | 本轮设置 |
|---|---|
| 数据 | 全部 50 条视频，10 类，总时长约 3.50 小时 |
| 分片 | 两份各 25 条，按 `sha256("42:video_id")` 排序后交错分配，不按标签或成绩选样本 |
| 模型 | Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、TP=1 |
| 查询 | 固定 `Highlight of this video.`；不输入标题、类别、caption 或标签 |
| 评分 | 原 QV REZE B0 提示和 Yes/No 概率计算；3 秒窗口、2 fps、完整窗口 6 帧、每批 16 个窗口 |
| 映射 | 保留原 REZE 含共享边界的逐秒曲线，再取相邻两秒均值；不足 2 秒的尾段不另建评价单元 |
| 规模 | 正式窗口请求 **4,213**（2,086＋2,127）；另有 18 次 GPU smoke；输出 **6,266** 个 2 秒片段分数 |
| 范围 | 只做高光检测主实验；无训练、提示词搜索、记忆、重排或回看，不计算摘要 F1 |

这是 REZE 评分流程的**跨数据集任务迁移**，不能表述为复现 REZE 原论文已有的 TVSum 成绩。通用高光 query 与严格动作核验提示之间可能存在适配局限，本轮固定这一接口，避免同时引入方法修改。

## 评价协议

使用 [UMT 作者实现](https://github.com/TencentARC/UMT/tree/bde05e3de39049bf90d8807f6b2a4ed8b42acbb0/datasets)及其发布的 [TVSum 标注包](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/tvsum-ec05ad4e.zip)，不沿用 MR.HiSum 的 mAP50／mAP15。

- **主报告：UMT 固定验证集 10 条的 Top-5 mAP**，每类一条。
- **补充报告：全部 50 条的 Top-5 mAP**，其中包括 UMT 划为训练集的 40 条；本实验不使用任何一条训练或调参。此分数不能混作监督方法的验证集成绩。
- 20 名标注者分别按严格大于其中位数二值化，在预测 top-5 内计算作者的梯形 AP，再平均。分母是 top-5 内的正例数，不是通常的完整排序 sklearn AP。
- 保留当前 CPU float32 的排序行为并记录并列分数。逐分片与未修改的 UMT `evaluate` 方法体交叉核对。
- 同时报告常数分数、均匀随机排序的精确期望、配对差值，以及按视频重采样 10,000 次的 95% bootstrap 区间（seed 42）。单类验证集只有一条，不能据此推断可靠的类内泛化区间。

全部 50 条的标注长度均与作者发布的视频／音频特征最短长度、原视频完整 2 秒区间数一致；特征只用于检查长度，不输入模型。原始 MAT/TSV 是逐帧展开标注，不能将数组长度当作 2 秒片段数。本轮直接使用作者发布的 2 秒 HD 标注。`WG0MBPpPC6I`、`Se3oxnaPsz0` 比原逐帧标注多一帧的既有差异保留，不修改视频或标签。

## 校验与自动恢复

23 项针对性测试通过，覆盖分数边界、作者指标一致性、随机对照的穷举核验、固定划分、分片合并、独立评估进程、占用 GPU 排除及并发 GPU 文件锁。新实验代码通过 Ruff；不表示整个工作区测试已通过。

每条视频先做首／中／尾共三个窗口的 CPU 原生模型输入检查；全部通过后才允许 GPU smoke 和正式评分。三个原 REZE 模块与 QV B0 逐字节校验一致。

用户级 systemd 定时器每分钟检查一次。仅使用无其他计算进程、显存占用低于 512 MiB、利用率低于 5% 的卡；加载前用 UUID 复核物理卡。两个分片共享 GPU 文件锁，避免同时抢占同一空闲卡，不中止其他任务。

每 16 窗口落盘，保留原始 logprob、帧索引／像素哈希和请求台账。每分片最多三次 worker 尝试、累计两小时 worker 时间、正式请求数加 128 次开销；900 秒无进展停止。心跳用于恢复后的时间核算。Linger 已启用，支持退出登录和机器重启后继续调度；需用户服务与 SSD 恢复可用。

推理完成后释放模型，在独立 CPU 进程评估；汇总器严格核对 50 条无遗漏、无重复、来源哈希一致，再分别生成 val10 和 all50 结果。失败记录保留，不能将部分输出当成最终成绩。

## 文件和查看命令

- 实现：`$ZEROSHOT_ROOT/experiments/reze_tvsum/`，[详见实现说明](../baselines/zeroshot/experiments/reze_tvsum/README.md)。
- 运行：`$ZEROSHOT_ARTIFACT_ROOT/reze_tvsum_full50_v1/`。
- 本机指针：reze_tvsum_current_run.local.json（本地资料：`reze_tvsum_current_run.local.json`）。
- 自动产物：`evaluation.json`、`per_video_metrics.jsonl`、`hd_predictions.jsonl`、`REPORT.md`、`complete.json`。
- 原始记录：`shard_*/chunks/`、`request_attempts/`、`worker_attempt_*.log`；冻结源码：`source_snapshot/`。

```bash
systemctl --user list-timers 'reze-tvsum-full50-v1-*'
cat $ZEROSHOT_ARTIFACT_ROOT/reze_tvsum_full50_v1/status.json
cat $ZEROSHOT_ARTIFACT_ROOT/reze_tvsum_full50_v1/shard_0/status.json
cat $ZEROSHOT_ARTIFACT_ROOT/reze_tvsum_full50_v1/shard_1/status.json
```

启动前粗估为双卡 10–20 分钟、单卡 20–30 分钟，不包含排队时间；最终实际耗时以上方完成记录为准。

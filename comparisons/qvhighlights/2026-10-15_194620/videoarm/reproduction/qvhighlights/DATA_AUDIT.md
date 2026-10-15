# P0 数据身份与映射审计

冻结来源、SHA-256 和获取时间：`checks/source_lock.json`。只读复用官方 README 所指 [UNC 原视频压缩包](https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz) 的已有本地缓存；压缩包身份 `checks/raw_archive_identity.json`，解压目录 `/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos`。该目录有 12,562 个 MP4，其中本任务官方 train/val/test 所需 10,148 个片段全部存在。无特征包下载，无字幕或 ASR 输入。

不加载发布的 `subs_train.jsonl` 或其他字幕文件。原始画面内固有印字/烧录文字保留；video-only 指模型输入模态，不承诺消除画面内所有语言线索。首个真实 smoke 视频的 24 帧、顺序和实际 PTS 已保存 `checks/initial_sampling_preview.{png,json}` 并人工视觉核对；这不是模型图像绑定能力或检索质量验收。

| split | queries | 数据集 vid | 源 YouTube ID | 可用媒体 query |
|---|---:|---:|---:|---:|
| train | 7,218 | 7,100 | 2,214 | 7,218 |
| val | 1,550 | 1,519 | 474 | 1,550 |
| test | 1,542 | 1,529 | 476 | 1,542 |

清单实际字段只有 `qid, query, vid, duration, media_path, timeline_id, num_clips, split`。推理入口要求字段集合精确相等，额外字段直接拒绝。准备程序投影 train/val 的非标签字段，test 使用无标签的 release；独立 scorer 在冻结后才读 test_with_gt。不同 query 即使共享 vid，也有独立 qid、预算、状态和语义缓存。

开发样本已在任何模型分数出现前落盘：`manifests/development_selection.json`、`dev50.jsonl`、`smoke8.jsonl`。算法为 seed 42 的确定性 SHA-256 排序：先按 `[42,"source",source_id]` 选源视频顺序，组内按 `[42,"query",qid,vid]` 选第一条 query，取前 50 个不同源视频；smoke 取前 8 条。JSON canonicalization 定义见 `common.digest`。不会依据媒体可用性、推理成功或得分替换样本。

`vid.rsplit("_",2)` 保留源 ID 内的下划线。媒体已裁剪，使用容器起点归一化的原始 PTS，源 start/end 只用于身份审计，不再次加偏移。官方 HD `mk_gt_scores` 定义 `num_clips=int(duration/2)`，已按固定版本确认。MR 不限制边界到 2 秒网格。

初始 ffprobe 元数据检查 10,148 个片段，0.25 秒严格一致性门槛下 9,926 通过、222 时长差异隔离，原始 `checks/media_metadata.jsonl` 保留。差异为标注时长减视频 stream 时长 −0.411078 至 1.742875 秒，没有超过 2 秒。这是在查看任何模型成绩前发现的数据时间粒度差异。登记的新 mapping 只允许恒等 PTS 坐标、时长差 −1 至小于 2 秒且每个官方 clip 存在真实帧；媒体结束之后没有像素、没有重复末帧填补、没有缩放时间轴或修改标签。

新元数据审计标记这些尾部差异为待完整 PTS 验证的映射，而非仅凭文件名认可。dev50 原门槛隔离 1 条，恒等映射、全 PTS、首尾实际解码及逐 clip 实帧检查 50/50 通过；原样本和原失败日志保留。部分末尾 clip 实际可见时长不足 2 秒，其代表帧在实际存在的时间范围内。

全量媒体阶段 `checks/media_all.jsonl` 保存每文件 SHA、PTS 序列哈希、首尾时间、VFR 标记和解码状态；动态数量 `checks/media_all_progress.json`。只有完成全量审计后才能冻结 val。检查失败的 vid 不删除，正式阶段按原分母单列失败。正式推理逐个核对冻结 media/PTS 身份。

全片解码由 `videoarm-qvh-media-audit.service` 运行，使用 12 个线程池任务、每个 ffprobe 单线程，写入新的 `QVH_DATA_ROOT/timelines/`；没有对共享媒体执行修改或重新编码。

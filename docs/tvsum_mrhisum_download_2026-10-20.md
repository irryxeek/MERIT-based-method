# TVSum 与 MR.HiSum 下载记录

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

日期：2026-10-20；名称纠正：2026-10-21。

**纠正：用户要求的 `tvhighlight` 指 TVHighlights。此前将其暂按 TVSum 处理是错误的名称映射。** 本文的 50 条视频属于 TVSum，不能计为 TVHighlights 下载完成。TVSum 已有文件保留；TVHighlights 的论文与补充材料已下载，视频和标注尚未取得，见[独立核查记录](tvhighlights_download_2026-10-21.md)。`mr.sum` 仍按此前的 MR.HiSum 范围执行。

**TVSum 全部视频与标注已下载并检查；MR.HiSum 全量官方标注已下载，原视频范围为官方测试集 2,000 条。** 首轮取得 1,872 条，75 条补试于 2026-10-21 10:12 CST 完成，新增取得 19 条。合计 **1,891 条**，其中 **1,888 条**通过基础媒体和时长检查、3 条需时长复核；109 条未取得。合并数量见本机路径入口（本地资料：`tvsum_mrhisum_download.local.json`）中的 `MR.HiSum.live_status`，指向数据根目录的 `availability.json`；原 `raw/status.json` 保留首轮结果。后续已启动[固定 100 条 REZE 测试](reze_mrhisum_execution_2026-10-21.md)。

## 来源与范围

| 数据集 | 已取得的官方资源 | 原视频范围 |
|---|---|---|
| TVSum | v1.1 压缩包、MAT/TSV 标注、元数据、缩略图及评估脚本 | 完整 50 条，约 3.50 小时 |
| MR.HiSum | `mr_hisum.h5`、`metadata.csv`、`mr_hisum_split.json`，共 31,892 个视频 ID | 首批官方 test 2,000 条，标注总时长约 112.54 小时；train/val 原视频未加入队列 |

TVSum 使用[作者官方下载页](https://people.csail.mit.edu/yalesong/tvsum/)的 `tvsum50_ver_1_1.tgz`，来源仓库版本为 `7bf3fb8ca032f5f4b42e22ba41f5e2490b819f96`。压缩包 671,779,858 字节，解压后视频共 681,259,038 字节。

MR.HiSum 使用[官方仓库](https://github.com/MRHiSum/MR.HiSum)版本 `aeb667f71239887034c2974062a341d312762867` 提供的下载链接及划分。train/val/test 分别为 **27,892 / 2,000 / 2,000**。官方 H5 含 `change_points`、`gt_summary`、`gtscore`，不含原视频或预提取特征。原视频根据官方元数据中的 YouTube ID 获取。

## 本地位置

数据根目录：`$SHARED_DATA_ROOT/benchmark_data/`。

| 内容 | 根目录下的相对路径 |
|---|---|
| TVSum 原视频 | `TVSum/official_release/ydata-tvsum50-v1_1/video/` |
| TVSum 原始 MAT / TSV | `TVSum/official_release/ydata-tvsum50-v1_1/matlab/`、`data/` |
| TVSum 校验结果 | `TVSum/video_audit.json`、`release_manifest.json` |
| MR.HiSum 全量标注 | `MR.HiSum/annotations/` |
| MR.HiSum 原视频 | `MR.HiSum/raw/videos/` |
| MR.HiSum 标注检查 | `MR.HiSum/annotation_audit.json` |
| MR.HiSum 下载清单与状态 | `MR.HiSum/raw/download_manifest.json`、`status.json` |
| MR.HiSum 单视频收据与尝试 | `MR.HiSum/raw/receipts/`、`attempts/`、`logs/` |
| MR.HiSum 合并可用数量 | `MR.HiSum/availability.json` |
| MR.HiSum 本次重试及有效收据索引 | `MR.HiSum/raw/retries/20261021_v1/`；其中 `effective_receipts.json` 按官方视频 ID 合并首轮及重试结果 |

官方说明、许可文件、原始压缩包、哈希和失败记录均保留。大文件保存在 SSD，本轮没有推送远程仓库。

## 校验与接入边界

TVSum 全部压缩成员通过 CRC 检查，50 个视频与 MAT/TSV 的 ID 一致。1,000 份用户评分在两种文件中逐值相同，均值与 `gt_score` 一致；每条视频首、中、尾各解码一帧，共 150 帧，通过检查。

实际发布的 MAT/TSV 数组已按帧展开，不能把数组长度直接当作 2 秒片段数。48 条视频的帧数与标注长度完全一致；以下 2 条各多 1 帧：

| 视频 ID | 媒体帧数 | 标注帧数 |
|---|---:|---:|
| `WG0MBPpPC6I` | 9,535 | 9,534 |
| `Se3oxnaPsz0` | 4,166 | 4,165 |

差异没有通过裁剪或补标签消除；接入评估时需核对末尾映射，保留官方标注长度。

MR.HiSum 的全部 H5 ID、元数据和划分一致；分数有限且在 `[0,1]` 内，摘要为二值，评分长度与元数据秒数一致。发布的相邻 `change_points` **共享端点**，原样保留。初始检查器误用了不重叠闭区间假设，修正记录保留为 `annotation_audit_inclusive_assumption.json`；没有改动标注。

原视频逐条检查源 ID、SHA-256、媒体信息和首/中/尾解码；与标注时长相差超过 2 秒时标记 `downloaded_timeline_review`。这是完整性与时长初筛，尚未证明逐秒语义对齐，不能直接当作评估准入。

## 2026-10-21：失败重试

用户授权补试失败项后，按失败原因而非高光标签冻结 75 条重试清单：

| 原失败原因 | 数量 | 本轮处理 |
|---|---:|---|
| 下载超时 | 2 | 优先断点续传 |
| 临时人机验证 | 18 | 降低请求频率后，按公开访问方式重试 |
| 代理限制 | 4 | 重新检查原来源；直接连接 YouTube 超时，因此仍使用既有代理 |
| 原因不明确的不可用提示 | 51 | 复查一次可用性 |
| 明确私有 / 已移除 / 年龄认证 | 49 / 3 / 1 | 保留原始缺失记录，未加入本次队列 |

首轮的 33 条 `source_or_download_error` 实际包含 27 条不可用提示、4 条代理限制和 2 条超时。新下载器补齐这两类不可用措辞并单列 `network_restricted`，避免将来源状态误计为连续网络故障；首轮代码、收据、日志与完成文件未改动。

本轮使用单线程，每条增加 **1 次程序尝试**，下载超时上限 **600 秒**，提取请求之间间隔 1 秒、条目之间间隔 3 秒；下载器内部网络重试仍为 1。继续使用原格式、媒体校验、120 GiB 累计媒体上限和 100 GiB 最低可用空间。原 `.part` 可续传，新增媒体保存在同一 `raw/videos/`，重试收据和日志单独保存。3 条已有时长异常不属于下载失败，未重下。

服务 `mrhisum-test-video-retry-20261021.service` 已启用，异常退出和重启后可恢复；程序同时锁定首轮与重试目录，防止并发修改同一媒体。每条完成后更新 `availability.json`，并写入按官方 ID 合并的 `effective_receipts.json`。`queue_finished` 表示重试队列处理完毕，不代表所有视频都已取得。

新脚本为 `scripts/retry_mrhisum_downloads.py`。冻结清单、原收据快照和哈希、实际运行代码均位于本次重试目录。12 项针对性测试通过，覆盖失败分类、重试范围、原结果保留、视频来源一致性和恢复时不重复下载；Ruff 与 systemd 配置校验通过。

```bash
# 当前总可用数量和补下载队列
cat $SHARED_DATA_ROOT/benchmark_data/MR.HiSum/availability.json
systemctl --user status mrhisum-test-video-retry-20261021.service

# 尚无本轮 complete.json / stopped.json 时，恢复冻结的重试队列
systemctl --user start mrhisum-test-video-retry-20261021.service
```

## 首轮后台下载与恢复（历史配置）

服务：`mrhisum-test-video-download.service`，已启用用户级 systemd 和 Linger。冻结清单按 `sha256("42:video_id")` 排序，不按标签分数筛选。

- 2 个下载线程，优先选择不高于 480p 的 H.264/MP4 视频与 M4A 音频；无音频格式时允许视频流。保留完整时间轴，不裁剪或转码。
- 每条最多 2 次程序尝试，每次最长 300 秒；下载器内部网络重试为 1。缺失、认证要求和解码错误单独记录，未提供账户 Cookie。
- 下载目录达到 120 GiB、磁盘剩余不足 100 GiB，或连续 8 次一般下载错误时停止并记录原因。预算按批次检查，最多存在一个并发批次的余量。
- 中断后保留 `.part`、尝试计数和成功收据；服务可在异常退出或机器重启后继续。结束时的 `complete.json` 表示队列处理完毕，仍须检查失败数，不等于所有视频可用。

试下载曾取得一条 AV1 视频，当前 Decord 无法读取。修复选择规则后补回该条，固定 3 条均通过检查；其中保留的 1 条 VP9 也已验证可解码，新下载限定 H.264。此前两次失败与 AV1 文件保存在 `raw/source_checks/av1_pilot_failure/`，该条另给两次修复后尝试预算。

```bash
# 首轮结果及原服务；当前进度请使用上面的 availability.json
cat $SHARED_DATA_ROOT/benchmark_data/MR.HiSum/raw/status.json
systemctl --user status mrhisum-test-video-download.service

# 在没有 complete.json / stopped.json 时恢复同一冻结队列
systemctl --user start mrhisum-test-video-download.service

# 重新核对 TVSum；不改媒体或标签
PYTHONPATH=$SHARED_DATA_ROOT/pdr_mem/tools/qfvs_inspection \
  $SHARED_DATA_ROOT/env/inference/bin/python \
  scripts/audit_tvsum_release.py \
  --root $SHARED_DATA_ROOT/benchmark_data/TVSum
```

运行依赖：Python 3.10、yt-dlp 2026.08.19、yt-dlp-ejs 0.8.0、Node 24.21.0、Decord 0.6.0、FFprobe 4.4.2；H5 检查使用已有独立目录的 h5py 3.15.1。EJS 单独安装在 SSD 工具目录，未升级推理环境。安装声明见 `pyproject.toml` 的 `download` 可选依赖；Node、FFmpeg/FFprobe 需另行提供。

新增脚本通过 Ruff；H.264/AV1 选择回归测试通过（`python -m pytest tests/test_dataset_downloads.py -q`），systemd 配置检查通过。下载阶段未启动模型推理；2026-10-21 用户另行要求的 MR.HiSum 推理见[独立执行记录](reze_mrhisum_execution_2026-10-21.md)。

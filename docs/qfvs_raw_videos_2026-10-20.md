# QFVS 原始视频下载与校验

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

日期：2026-10-20。**P01–P04 四条原始视频已下载、解压并完成完整性与抽样解码检查。** 视频共 16.9626 小时、1.535 GB；保留的 ZIP 为 1.492 GB。容量按十进制计算。

## 来源与位置

来源为 [UT Egocentric 官方公开版本](https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html)，通过其官方下载入口获取 `UTE_video.zip`。公开版本包含发布方的人脸模糊处理；本次保留原始文件字节，没有重新编码、裁剪或更改帧率。

- 视频目录：`$SHARED_DATA_ROOT/pdr_mem/qfvs/raw/official_release/`
- 原始压缩包：`$SHARED_DATA_ROOT/pdr_mem/qfvs/raw/archives/UTE_video.zip`
- 标注与特征：`$SHARED_DATA_ROOT/pdr_mem/qfvs/data/`
- 本机路径与校验记录入口（本地资料：`qfvs_raw_videos.local.json`）

| 视频 | 实际时长 | 大小 / MB | 原生标注片段数 |
|---|---:|---:|---:|
| P01.mp4 | 03:51:51.067 | 349.59 | 2,783 |
| P02.mp4 | 05:07:37.267 | 463.70 | 3,692 |
| P03.mp4 | 02:59:16.733 | 270.41 | 2,152 |
| P04.mp4 | 04:59:00.200 | 450.88 | 3,588 |
| 合计 | 16.9626 小时 | 1,534.58 | 12,215 |

四条视频均为 **480×320、15 FPS、MPEG-4 视频编码**，起始时间为 0。

## 已完成检查

1. ZIP 全部四个成员通过 CRC 校验，解压文件大小一致；计算并保存压缩包和每条视频的 SHA-256。官方入口未提供独立校验和。
2. `ffprobe` 元数据与 Decord 帧计数一致；全部帧的时间戳有限、递增，首尾时间与视频时长一致。
3. 每条视频首、中、尾各解码一帧，共 12 帧，尺寸与元数据一致，并保存 RGB 像素哈希。这是抽样解码，未逐帧解码整条视频。
4. P01–P04 与原生 `Tags.mat` 中的四个视频 ID、片段数量对应。检查没有修改媒体、标注或原特征。

详细证据位于数据根目录的 `raw/archive.json`、`raw/release_manifest.json`、`raw/complete.json` 和 `raw/video_audit.json`。其中 `release_manifest.json` 的时间轴待查字段保留下载完成时的快照，后续核查结果见 `video_audit.json`。

## 接入评估时的末尾边界

原标注说明采用 5 秒片段，摘要索引从 1 开始。按零起点 5 秒网格核对，末尾存在以下差异：

- P01、P02、P03 的最后一个标注片段分别仅有 **1.067、2.267、1.733 秒**有效视频。
- P04 的 3,588 个标注片段覆盖到 17,940 秒，原视频另有 **0.200 秒**尾部。

后续适配应保留原生片段数，在有效视频范围内采样最后一段；P04 额外尾部不增加标注项。不能统一用 `ceil(duration / 5)` 推导四条视频的评分长度。本次仅核对 ID、时间戳和网格数量，未逐片段复核语义标注。

## 复查命令与执行记录

```bash
cd $MERIT_ROOT
MERIT_PY=$SHARED_DATA_ROOT/env/inference/bin/python
QFVS_ROOT=$SHARED_DATA_ROOT/pdr_mem/qfvs

# 已有 complete.json 时直接保留已完成下载。
$MERIT_PY scripts/download_qfvs_videos.py --root "$QFVS_ROOT/raw"
$MERIT_PY scripts/audit_qfvs_videos.py --root "$QFVS_ROOT"
```

使用现有 Python 3.10 环境：Requests 2.34.2、Decord 0.6.0、SciPy 1.15.3，以及系统 FFprobe 4.4.2。下载相关依赖列入 `pyproject.toml` 的 `download` 可选依赖。

下载由独立用户服务完成；首轮直连缓慢，切换本机已有代理后约 6 分钟下载完成。服务器忽略 Range 请求，因此第二次从头下载，没有把完整响应追加到旧文件。两次尝试、服务配置和日志均保留在 `raw/`。下载脚本通过 Ruff 检查；另检查了路径越界拒绝、解压哈希/CRC，以及服务器忽略 Range 时重新下载的行为。

下载阶段未执行模型推理。随后已接入 REZE 的原生 5 秒片段接口、摘要选择与语义 F1 评估，并启动固定 4 查询测试，见[推理执行记录](reze_qfvs_execution_2026-10-20.md)。

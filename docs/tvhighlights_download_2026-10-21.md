# TVHighlights 下载与公开入口核查

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

核查日期：2026-10-21（Asia/Shanghai）。

**TVHighlights 的视频和标注尚未下载。已取得官方论文、补充材料及论文页面，未找到可用的官方数据下载入口。** 此前误将用户的 `tvhighlight` 请求映射为 TVSum；已下载的 TVSum 50 条视频是另一个数据集，不能替代本任务，原文件与校验记录保留。

## 官方依据

论文：[TVHighlights: LLM-Guided Human-Free Collaborative Training for Video Highlight Detection in Movies and TV Dramas](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_TVHighlights_LLM-Guided_Human-Free_Collaborative_Training_for_Video_Highlight_Detection_in_CVPR_2026_paper.html)，CVPR 2026。

- 正文表 1 将公开状态列为 **Forthcoming**，且明确区分 TVHighlights 与 TVSum。
- [补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Qiu_TVHighlights_LLM-Guided_Human-Free_CVPR_2026_supplemental.pdf)第 1 页的 Dataset Availability 说明：计划公开经过筛选的部分训练集和人工标注测试集，待匿名化及隐私、版权处理完成后发布。
- 论文中的完整数据规模为 1,721 条视频：训练 1,368 条，测试 353 条；这是论文统计，不代表已公开或已下载的数量。

本次检查了官方正文、补充材料中的文本与 PDF 链接，以及 GitHub 仓库、README 索引和 Hugging Face 数据集/模型/Space 搜索。没有找到可核实的官方发布入口；这不能证明不存在尚未被检索到的发布渠道。

## 本地文件与状态

根目录：`$SHARED_DATA_ROOT/benchmark_data/TVHighlights/`。

| 内容 | 相对路径 |
|---|---|
| 官方论文、补充材料、网页 | `sources/paper.pdf`、`sources/supplemental.pdf`、`sources/paper.html` |
| 提取文本 | `sources/paper.txt`、`sources/supplemental.txt` |
| 下载地址、字节数、SHA-256 | `sources/download_receipts.json` |
| GitHub / Hugging Face 检索记录 | `sources/release_checks.json` |
| 数据获取状态 | `status.json` |

当前视频数 **0**，标注文件数 **0**，无媒体下载队列，未启动模型推理。需要作者发布入口或提供数据访问链接后，才能下载并核验测试集视频、标签、划分及时间轴。论文中的作者联系方式可用于申请数据，本次未向作者发送消息。

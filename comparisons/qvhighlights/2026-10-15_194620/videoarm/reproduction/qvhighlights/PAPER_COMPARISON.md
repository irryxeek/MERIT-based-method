# M1 与 VideoARM 论文原始数据的对照

核对版本为 [arXiv:2512.12360v2](https://arxiv.org/html/2512.12360v2)，2026-03-30。原始 PDF 与文本在 checks/2512.12360v2.*，结构化抄录在 checks/paper_reference_m1.json。

论文没有 QVHighlights MR/HD 成绩，也没有 SVHighlights 成绩。其表 1 是视频问答准确率，不能与本任务的 mAP、R1、Hit@1 相减或据此声称接近原论文性能。

| 论文表 1 配置 | Video-MME overall | Video-MME long | LongVideoBench overall | EgoSchema 500 | MLVU | LVBench |
|---|---:|---:|---:|---:|---:|---:|
| o3 + GPT-4.1 + whisper-1 | 80.1 | 75.3 | 73.7 | 78.2 | 81.2 | 79.7 |
| o3 + GPT-4o + whisper-1 | 82.8 | 81.2 | 78.0 | 76.2 | 未报告 | 未报告 |

上述数字都是论文报告的 Accuracy (%)，并非本地新运行。表 2 的 Qwen3-VL Controller 消融报告 Video-MME-long 的 54.9%，但视觉理解工具仍为 GPT-4.1、音频仍为 whisper-1，且使用 200 QA 的消融子集；论文该表未给出 Qwen 模型大小和 revision，不能将它当作全 Qwen3-VL-8B、video-only 的原始基线。

| 对照项 | 论文原始方法 | 本次 QVHighlights M1 |
|---|---|---|
| 任务 | 视频问答 | 查询相关时间定位 MR；独立高光排序 HD |
| 模型 | o3 Controller，GPT-4.1/GPT-4o 视觉工具 | 同一本地 Qwen3-VL-8B-Instruct 承担各角色 |
| 模态 | 视觉与 whisper-1 音频；Video-MME 无发布字幕 | 只用视觉，无音频、ASR、发布字幕 |
| 时间范围 | 间隔定位与局部探索，帧索引 | 数据集当前片段起点的秒级时间范围，真实 PTS |
| 采样 | Interval Localizer 30–150 帧、3×2 拼图；短边 256/512 | 每请求最多 24 帧，不拼图；长边最多 672 |
| 停止 | 自主 Answer 或 10 步 | 自主 stop、工具及预算上限；记录具体停止原因 |
| 记忆 | 感知池、结果及工作记忆 | 感知证据、固定结果摘要和行动目的；存在 METHOD_DELTA.md 所列差异 |
| HD 覆盖 | 未报告该任务 | 每个官方 2 秒 clip 初始化；合法局部结果按时间顺序更新 |

论文表 7 的 1.89M token 来自 10 个视频、30 个 query、视频平均 41.3 分钟；其理论单条 30 分钟视频上界为 0.08M。此次约 150 秒片段、不同 tokenizer 与视觉编码，绝对 token 和墙钟耗时不直接可比，也不能套用论文的 1/34 或 1/50 节省比例。本文将记录本地真实 input/output token、图像、工具次数及停止频率作为描述性对照；论文没有给出相同本地 GPU 服务的吞吐基准。

本地 M1 的实际分数、覆盖率、失败与成本由 FINAL_REPORT.md 和各 runs/*/scores.json 更新。现有 smoke/dev 是训练开发子集，尚未运行的完整 val/test 不填结果。没有新增 B1/A1 时，本轮无法建立同协议的调度/记忆增益结论。

2026-10-14 用户将后续工作收缩到 M1；目前继续 QVHighlights MR/HD，不自行新增整套论文问答数据推理。若另外运行论文 benchmark，需要独立记录数据名单、问答协议与同指标可比范围。现有 SVHighlights 仍暂停。

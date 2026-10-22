# 高光检测与视频摘要数据集：特点、输入输出和评价指标

整理日期：2026-10-22。覆盖本项目涉及的 QVHighlights、TVSum、MR.HiSum、QFVS、SVHighlights 和 TVHighlights。

这些数据集分别考察查询相关性、片段重要性、观众重播偏好、摘要覆盖、体育事件和影视高潮。迁移同一评分模型时，应分别保留各数据集的输入条件、时间粒度、选择预算和评价协议。

本文的“输入”指推理时允许提供给模型的信息。真实高光标签、人工重要性分数、参考摘要和概念标签用于评价，不作为零样本推理输入。数据集的原任务与本项目的 REZE 适配配置分别说明。

## 1. 数据集特点与输入输出总览

| 数据集 | 内容与特点 | 推理输入 | 预测输出 |
|---|---|---|---|
| QVHighlights | 分钟级网络视频；根据具体自然语言查询检测高光，也支持时刻检索 | 视频＋自然语言查询 | HD：每个 2 秒 clip 的高光分数；MR：若干起止区间及置信度 |
| TVSum | 50 条视频、10 类主题，包含教程、新闻、生活记录等；每条视频有 20 人的重要性标注 | 视频；公开标题可作为额外主题信息，本项目当前未使用 | 片段重要性分数，或进一步选择摘要；当前适配输出 2 秒粒度分数 |
| MR.HiSum | 31,892 条 YouTube 视频；重要性来源于观众 Most Replayed 重播统计 | 视频，无需具体用户查询 | 高光分数曲线；也可输出所选摘要片段或二值摘要掩码 |
| QFVS | 4 条约 3–5 小时的第一视角生活视频；围绕概念查询生成有代表性的摘要 | 长视频＋概念查询，例如 `Book / Lady` | 一组被选中的 5 秒片段编号或时间区间；片段分数是选择的中间结果 |
| SVHighlights | 320 场、8 类体育比赛，平均约 2 小时；标签来自官方集锦与完整比赛的对齐 | 比赛视频＋运动类别；可使用音频、解说等模态 | 整条比赛每个 2 秒 clip 的高光分数 |
| TVHighlights | 影视剧短片段，包含打斗、破坏、特效、情绪高潮等；整理后共 1,721 条 | 影视视频，无需具体用户查询；视觉和音频可共同提供线索 | 每个 2 秒 clip 的高光分数 |

数据定义来源：[QVHighlights 官方说明](https://github.com/jayleicn/moment_detr/blob/main/data/README.md)、[TVSum 官方资料](https://github.com/yalesong/tvsum)、[MR.HiSum 官方资料](https://github.com/MRHiSum/MR.HiSum)、[QFVS 原论文](https://arxiv.org/html/1707.04960)、[SVHighlights 官方资料](https://github.com/leedongkyu2019/SVHighlights)、[TVHighlights 补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Qiu_TVHighlights_LLM-Guided_Human-Free_CVPR_2026_supplemental.pdf)。

## 2. 标注含义与评价指标

### QVHighlights：根据查询判断高光

例如输入一段生活视频和“一个人在户外打篮球”，模型需要找到与该查询相关的片段，再判断哪些片段最值得排在前面。官方数据同时提供相关时刻区间和三位标注者的片段显著性评分，评分范围为 0–4。[标注格式](https://github.com/jayleicn/moment_detr/blob/main/data/README.md)

- **HD mAP**：评价整条预测分数序列的高光排序；按官方实现汇总标注者和查询。
- **Hit@1**：检查最高分片段是否命中高光；官方实现对三位标注者采用至少一位认可的口径。
- **Fair / Good / VeryGood**：分别以人工分数至少 2 / 3 / 4 为正例标准。本项目主要报告 VeryGood mAP / Hit@1。
- **时刻检索指标**：使用预测区间与真值区间的时间 IoU 计算 MR mAP、Recall@1 等；与 HD 指标分别报告。当前 REZE 主结果针对 HD，不能由此声称同时完成了 MR 复现。

具体实现见[官方评价器](https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/eval.py)；本项目范围见[QVHighlights 完整结果](reze_fullval_results_2026-10-20.md)。

### TVSum：判断片段对全片的重要性

TVSum 原任务是视频摘要，标题可提供视频主题。例如教程的关键步骤可能很重要，即使画面没有剧烈动作。原始标注是人工重要性分数，并非针对每条视频给出一个自由文本检索问题。[作者资料](https://github.com/yalesong/tvsum)

本项目采用 **UMT 的高光检测扩展协议**：将预测映射到 2 秒网格，使用 **Top-5 mAP**。评价器对每位标注者的分数以严格高于其中位数划分正例，在预测前五个片段上计算梯形 AP，再汇总标注者与视频。这里是“前五个片段”，不是“前 5%”；也不等同于通常的全排序 AP。[固定版本 UMT 实现](https://github.com/TencentARC/UMT/blob/bde05e3de39049bf90d8807f6b2a4ed8b42acbb0/datasets/tvsum.py)

主结果使用 UMT 固定 val10；all50 是额外统计，包含其训练划分的 40 条。本次没有训练模型，也没有计算视频摘要 F1，不能把当前 Top-5 mAP 写成 TVSum 唯一的官方评价指标。[本地适配协议](../baselines/zeroshot/experiments/reze_tvsum/README.md)

### MR.HiSum：预测观众重播价值，并生成摘要

此前交流中的“mr.sum”在本项目指 **MR.HiSum**。其 `gtscore` 是归一化到 0–1 的 Most Replayed 统计，参考摘要则结合镜头边界和背包选择生成。因此标签反映观众重播行为，不能直接解释为人工对“精彩程度”的统一评分。[作者数据定义](https://github.com/MRHiSum/MR.HiSum#complete-mrhisum-dataset)

- **mAP50 / mAP15**：当前沿用作者实现，将逐秒曲线按连续 5 秒取均值，以真值排序前 `floor(50% × N)` / `floor(15% × N)` 个时间格作为正例，计算每视频 AP 后取平均。50 / 15 表示正例比例，不是时间 IoU 阈值，也不是预测分数阈值。
- **摘要 F1**：按 KTS 镜头边界聚合预测分数，在 **15% 视频时长预算**内用背包方法选片，再与参考摘要的二值时间掩码比较。
- 高光排序与摘要选择依赖不同性质：分数校准、镜头长度和选片覆盖会影响摘要 F1，排序改善不保证摘要指标同步改善。

实现和边界处理见[MR.HiSum 适配协议](../baselines/zeroshot/experiments/reze_mrhisum/README.md)，完整可用测试范围见[全量执行报告](reze_mrhisum_full_execution_2026-10-21.md)。

### QFVS：围绕查询选择有代表性的长视频摘要

输入例如 `Book / Lady`，输出是若干原生 5 秒片段。当前实现按“两个概念中至少一个出现”构造查询。摘要还需要保留代表性内容，因此只选大量相邻、内容相似的高分片段，未必能获得更好的摘要评价。[任务与标注定义](https://arxiv.org/html/1707.04960)

**语义 Precision / Recall / F1** 的计算先用片段概念标签的交并比衡量相似度，再对预测摘要与参考摘要进行最大权二分匹配。它衡量保留的语义信息，不是简单的时间区间重合率，也不等同于“检测到了多少百分比的高光”。[原论文评价方法](https://arxiv.org/html/1707.04960)

本项目使用固定 UniVTG 配置的 **2% 片段预算**，即选择 `floor(0.02 × 原生片段数)` 个片段；这是本次采用的协议配置。主报聚合参考 Oracle 的 P/R/F1，辅报三位用户参考的平均指标。Oracle 是参考摘要，不表示推理读取了标签。[适配协议与固定评价器来源](../baselines/zeroshot/experiments/reze_qfvs/README.md)

本地发布版本包含每视频 45 条、合计 180 条非空查询；现有完成结果是四查询开发测试。概念提示、Building 名词消歧与时间分散选择属于后续适配，须与原 B0 条件分开记录。[四查询诊断](reze_qfvs_analysis_2026-10-20.md)、[优化执行报告](reze_qfvs_adaptation_2026-10-22.md)

### SVHighlights：从完整比赛中找出体育高光

官方通过比赛集锦与完整比赛对齐产生 2 秒粒度二值高光标签，并提供运动类别、转录及音频相关数据。固定文本条件为 `Highlight of this {sport} video`，例如足球比赛高光；它不等同于 QVHighlights 的具体事件查询。[官方数据说明](https://github.com/leedongkyu2019/SVHighlights)

- **HL-mAP**：整场比赛的片段高光排序质量。
- **HL-Hit1**：最高分片段是否是真值高光。
- **HL-Hitk**：前 K 个预测中命中真值高光的比例，官方评价的 K 为该视频的真值高光片段数。
- **HL-IoU**：预测高光集合与真值集合的交并比。官方以第 K 高分为阈值，保留边界同分项，因此实际入选数量可能多于 K。

以上真值数量只参与评价，不能作为部署时模型已知的高光数量。实际产品选片应另外指定 Top-k 或时长预算。各运动类别和总体结果分别统计。[官方评价器](https://github.com/leedongkyu2019/SVHighlights/blob/a463fafbca8136bc6839bddc91216e494d970833/eval.py)

### TVHighlights：影视中的动作、情绪和叙事高潮

该数据集与 TVSum 不同。整理后的训练集有 1,368 条、测试集有 353 条，平均时长分别为 50.6 秒和 134.9 秒；数据单元是短片段，并非整部电影或整集电视剧。测试高光分为动作与打斗、冲击与破坏、CGI、情绪高潮及其他类型。[补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Qiu_TVHighlights_LLM-Guided_Human-Free_CVPR_2026_supplemental.pdf)

主要评价 **AP 和 AUC**，检查逐片段评分对高光与非高光的区分能力，并报告类别与整体表现。论文的“无需人工标注训练”不表示没有人工测试真值：测试片段有人工标注。[原论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Qiu_TVHighlights_LLM-Guided_Human-Free_Collaborative_Training_for_Video_Highlight_Detection_in_CVPR_2026_paper.pdf)

截至本文整理时，本地只有论文和补充材料，尚无可用原始视频与标注，因此没有本项目的推理或评价结果。[数据获取记录](tvhighlights_download_2026-10-21.md)

## 3. 本项目 REZE 的实际输入与读出

已运行的 REZE B0 使用 Qwen3-VL-8B-Instruct，从原视频读取 **3 秒窗口、2 fps、完整窗口 6 帧**，利用 Yes/No 概率形成连续分数，再按数据集时间网格读出。[评分协议](zeroshot_baselines/REZE_PROTOCOL.md)

| 数据集 | 当前文本条件 | 分数读出与后处理 |
|---|---|---|
| QVHighlights | 数据集自然语言 query | 窗口分数到逐秒曲线，再映射为 2 秒分数；P_abs / P_pair 是额外候选复评 |
| TVSum | 人为添加 `Highlight of this video.`，未输入标题 | 映射为 2 秒分数，按 UMT Top-5 mAP 评价 |
| MR.HiSum | 人为添加 `Highlight of this video.` | 逐秒曲线；HD 聚合到 5 秒；摘要另做镜头聚合与 15% 预算选择 |
| QFVS | 双概念 OR 查询；新条件改为视觉概念出现核验 | 映射到原生 5 秒片段，按固定 2% 预算选摘要 |
| SVHighlights | 带运动类别的通用高光提示 | 已接入 2 秒输出和原生评价器；尚无有效 REZE 基线结果 |
| TVHighlights | 尚未接入 | 等取得视频和标注后确定适配实现 |

这里的 **2 秒或 5 秒是输出／评价网格，不等于模型读取窗口**。当前上述 REZE 基线读取视觉帧，没有利用音频、ASR 或先通读全视频生成全局记忆；没有具体用户查询的数据集，其通用文本提示属于方法适配。各迁移实现见[统一复现索引](https://github.com/irryxeek/MERIT-based-method/blob/main/docs/reproduction_index.md)。

## 4. 输出示例与指标比较边界

以下数字仅示意输出结构，不是实际预测，也不是要求所有数据集使用同一 JSON 提交格式。

```text
高光检测：120 秒视频 → 60 个 2 秒片段分数
scores = [0.10, 0.20, 0.90, ...]

时刻检索：视频＋查询 → 若干起止区间及置信度
moments = [[30.0, 42.0, 0.88], [80.0, 94.0, 0.72]]

视频摘要：长视频＋可选查询 → 预算内的片段集合
selected_shot_ids = [17, 83, 216, ...]
```

- 高光分数用于排序，不必立即输出剪辑文件；摘要任务还需要实际确定选哪些片段。
- QFVS 的语义 F1 与 MR.HiSum 的摘要掩码 F1 含义不同，数值不能直接比较。
- TVSum Top-5 mAP、QVHighlights HD mAP、MR.HiSum mAP50/15 的正例与汇总方式不同，不能用绝对分数判断哪个数据集上的方法“更强”。
- 对比方法时，应固定样本范围、输入模态、模型、帧预算、输出网格和评价实现；新增记忆、比较或回看分别记录成本与消融结果。

## 5. 当前实验覆盖及文档入口

截至 2026-10-22 已汇总的完成范围如下；后续运行状态以对应执行报告为准。

| 数据集 | 已有范围 | 记录 |
|---|---|---|
| QVHighlights | 完整 validation 1,550 查询；REZE B0、P_abs、P_pair | [完整结果](reze_fullval_results_2026-10-20.md) |
| TVSum | 全 50 条已完成；分别报告固定 val10 与 all50 | [执行与结果](reze_tvsum_execution_2026-10-22.md) |
| MR.HiSum | 官方 test 2,000 条中的可用 1,888 条；不是无缺失完整测试集 | [全量可用集报告](reze_mrhisum_full_execution_2026-10-21.md) |
| QFVS | 4 条完整长视频、固定 4 查询已完成原基线；新概念提示结果尚未纳入汇总 | [原结果分析](reze_qfvs_analysis_2026-10-20.md)、[适配报告](reze_qfvs_adaptation_2026-10-22.md) |
| SVHighlights | 部分视频可用，已接入评价；此前七场 REZE 计划启动失败，无有效指标 | [跨数据集执行记录](reze_cross_dataset_2026-10-20.md) |
| TVHighlights | 论文资料已保存；尚无本地视频与标注，无实验结果 | [获取记录](tvhighlights_download_2026-10-21.md) |

已有实验结果、差异解释和后续适配判断见[跨数据集结果与分析](reze_cross_dataset_results_analysis_2026-10-22.md)。

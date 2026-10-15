# LongVT 在 QVHighlights 上的适配说明

## 1. 总体结论

本项目没有修改 LongVT-RFT 的模型结构，也没有为已完成的主实验重新进行 SFT、LoRA、RL 或 RFT。当前已经完成并验证的是一套外围的、training-free 的 QVHighlights 适配层：它冻结发布的 `longvideotool/LongVT-RFT` 检查点，将 LongVT 原本的长视频观察、推理和 `crop_video` 工具调用能力，转换为 QVHighlights 要求的逐 2 秒 clip 显著性评分。

因此，这项工作应理解为“将冻结的 LongVT-RFT 接入 QVHighlights highlight detection”，而不是复现 LongVT 论文中的原始指标或训练流程。

## 2. 任务与输出协议适配

### 2.1 将开放式视频理解转换为逐 clip 显著性评分

QVHighlights 要求模型针对 query，为视频中每个完整、互不重叠的 2 秒 clip 输出显著性分数。适配层据此：

- 使用 `int(duration / 2)` 计算官方 clip 数量；
- 向模型提供 query、视频本地时长、视频路径和所有全局帧的本地时间戳；
- 明确每个 clip 的时间范围；
- 使用整数 0–4 的五级显著性定义：
  - 0：无关；
  - 1：弱相关；
  - 2：相关但普通；
  - 3：强代表性 highlight；
  - 4：最显著、最具代表性的证据；
- 要求模型对完整时间轴评分，而不仅是已经裁剪或重点观察的区域。

### 2.2 两阶段推理

推理被拆成两个固定阶段：

1. **Evidence analysis**：模型先分析与 query 相关的时间区间、强证据和弱证据；Tool 模式可以在此阶段自主调用 `crop_video`。
2. **Score finalization**：模型再将已有分析转换为结构化评分结果，不再调用工具。

没有采用一次性生成完整长分数数组的方案。早期 smoke 实验表明，长数组、数字字符串或无界区间列表容易出现重复、截断和常数向量；同时，在 vLLM 0.12 中将 `tools=auto` 与 response schema 同时使用会抑制工具调用。因此，正常证据分析阶段保持非约束生成，只在格式重试时使用 JSON Schema。

### 2.3 “背景分数 + 局部覆盖区间”的压缩表示

模型最终输出一个紧凑 JSON：

```json
{
  "background_score": 0,
  "segments": [
    {"start_time": 20, "end_time": 30, "score": 4}
  ]
}
```

其转换规则如下：

- 一个 `background_score` 覆盖所有未被局部区间覆盖的 clip；
- 最多允许 5 个局部时间覆盖区间；
- 所有分数必须是 0–4 的整数；
- 区间使用视频本地秒数，而不是 clip index；
- 当某个官方 2 秒 clip 的 midpoint 落入半开区间 `[start_time, end_time)` 时，对该 clip 应用区间分数；
- 多个覆盖区间重叠时取最大分数；
- 未裁剪区域使用模型显式选择的背景分数，不会自动置零。

这种紧凑表示由适配器确定性展开成官方 evaluator 所需的完整 `pred_saliency_scores` 向量。

## 3. Global/Tool 配对实验适配

项目实现了两个共享相同输入条件的推理模式：

| 模式 | 全局输入 | 局部工具 |
|---|---|---|
| Global 64 | 最多 64 张均匀采样帧 | 不开放工具 |
| Tool | 与 Global 完全相同的全局帧 | 最多 2 次 `crop_video`，每次最多 32 帧 |

工具执行链路增加了以下约束和记录：

- 局部裁剪按约 1 FPS 解码；
- 检查工具请求中的视频路径必须与当前样本一致；
- 将请求区间限制在实际视频时长内，并拒绝空区间或非法区间；
- 把返回的局部帧、实际执行区间和帧时间戳重新注入下一轮模型消息；
- 裁剪只作为额外观察，裁剪区间本身不会直接转换成 highlight 预测；
- 保存局部帧、帧哈希、调用参数、延迟、失败原因和是否成功回注；
- 重复读取的局部帧计入 Tool 模式成本。

为了保证配对比较的有效性，完整 validation 中逐题检查了两组的全局帧 SHA-256；1,550/1,550 个样本的全局帧完全一致。

## 4. 输出校验与失败处理

适配层对模型输出实施严格客户端校验：

- 顶层只能包含 `background_score` 和 `segments`；
- 每个 segment 只能包含 `start_time`、`end_time` 和 `score`；
- 检查字段类型、分数范围、时间范围和最大区间数量；
- 拒绝布尔值、NaN、Infinity、越界区间、错误向量长度及额外字段；
- 只允许两种预声明并记录的确定性语法清理：
  - 移除最外层 Markdown code fence；
  - 仅在分隔符计数证明只缺一个外层对象右括号时补上该括号；
- 第一次解析失败后，允许一次 JSON-Schema 约束的格式重试；
- 重试后仍然无效时使用全零向量，并将该样本保留在官方指标分母中。

该策略避免通过删除失败样本、静默修复语义错误或选择性重跑来抬高结果。

## 5. 数据和媒体适配

### 5.1 Manifest 与标签隔离

项目从官方 QVHighlights annotation 生成无标签 inference manifest，其中只保留推理需要的字段，例如：

- `qid`、`vid`、`query`；
- 视频时长和 clip 数量；
- 本地视频路径；
- 原始 YouTube ID 和片段起止时间。

`relevant_windows`、`relevant_clip_ids` 和 `saliency_scores` 被单独写入 ground-truth 文件，不进入模型消息。

固定 pilot 使用 seed 42，每个视频只选一个 query；前 8 个有效视频用于 smoke，后 50 个用于 pilot，两者视频互斥。完整 validation 则覆盖全部 1,550 个 query-video pair。

### 5.2 官方媒体处理

数据处理链路包括：

- 校验官方约 133.7 GiB 压缩包的精确大小和 gzip 完整性；
- pilot 阶段只从官方压缩包中提取固定候选视频和预留候选；
- 完整 validation 显式解压并检查完整媒体集；
- 使用 `ffprobe` 检查视频时长；
- 使用 Decord 解码首帧、中间帧和末帧；
- 记录缺失文件、时长偏差、解码失败和替换情况。

完整 validation 中有 8 个官方视频比 annotation duration 短 1.157–1.471 秒，因此在推理前将媒体验收容差冻结为 1.5 秒。该变化只影响媒体验收，不改变模型输入帧预算、prompt 或评分规则。

## 6. 官方 evaluator 适配

项目固定使用 `jayleicn/moment_detr` 的官方 QVHighlights evaluator，并在调用前增加严格包装：

- qid 不得重复；
- qid、vid 和 query 必须与 ground truth 一致；
- 每个预测必须恰好包含 `int(duration / 2)` 个分数；
- 所有分数必须是有限数值。

官方发布的 1,550 行示例结果可以被本项目精确复现。官方示例中有 13 行少一个分数，原 evaluator 会静默补零；本地严格包装会记录该差异，并拒绝模型产生的错误长度输出。

## 7. 服务与运行配置

针对 LongVT-RFT 的 Qwen2.5-VL 架构，推理服务固定为：

- 模型：`longvideotool/LongVT-RFT`；
- Qwen2.5-VL tool-call chat template；
- Hermes tool-call parser；
- 自动工具选择；
- vLLM 0.12.0；
- TP=2，使用 GPU 6/7；
- 最大上下文长度 65,536 tokens；
- `temperature=0`、`top_p=1`、seed 42；
- 单帧最大像素数 50,176。

启动脚本会在启动前检查 GPU 6/7 和服务端口是否被占用；若存在冲突则拒绝启动，不会终止其他进程。

## 8. 可复现性和诊断

除官方 mAP/Hit1 外，项目还保存或计算：

- 冻结配置、manifest 和关键脚本的 SHA-256；
- 每轮原始模型响应、finish reason 和 stop reason；
- 全局帧和局部帧哈希；
- 工具请求、执行区间、返回帧和回注状态；
- 每题 Global/Tool 指标和预测差异；
- 有效完成率及全零 fallback 数量；
- 常数预测、不同分数数量和最高分并列数；
- 工具裁剪对 released relevant clip 的事后覆盖率；
- 帧数、token、延迟和 GPU 显存采样。

并列诊断尤其重要：官方 Hit1 使用 `numpy.argmax`，最高分并列时选择最早的 clip；官方 mAP 则按相同预测分数阈值成组处理。大量常数预测和最高分并列会限制评分分辨率，并影响 Hit1 绝对值的解释。

## 9. 完整 validation 结果

完整 validation 覆盖全部 1,550 个 query-video pair，并包含预先声明的全零 fallback。

| 正例阈值 | 指标 | Global 64 | Tool | Tool − Global |
|---|---|---:|---:|---:|
| Fair（标注分数 ≥2） | HL-mAP | 30.96 | 32.17 | +1.21 |
| Fair（标注分数 ≥2） | HL-Hit1 | 28.97 | 31.16 | +2.19 |
| Good（标注分数 ≥3） | HL-mAP | 24.47 | 25.57 | +1.10 |
| Good（标注分数 ≥3） | HL-Hit1 | 26.97 | 29.16 | +2.19 |
| VeryGood（标注分数 ≥4） | HL-mAP | 13.90 | 14.60 | +0.70 |
| VeryGood（标注分数 ≥4） | HL-Hit1 | 20.65 | 22.71 | +2.06 |

在本次固定配置下，Tool 在六项官方指标上均优于 Global 64。该结果是观测到的配对差异，没有进行统计显著性检验，因此不能解释为统计显著的普遍收益。

### 9.1 完成率

| 项目 | Global 64 | Tool |
|---|---:|---:|
| 有效预测 | 1,534/1,550 | 1,522/1,550 |
| 全零 fallback | 16 | 28 |
| 两组共同有效 | 1,506/1,550 | 1,506/1,550 |

共同成功子集上的指标方向与包含 fallback 的全量结果一致，因此总体正向差异并非仅由失败数量不同造成。

### 9.2 工具使用情况

- 228/1,550 个样本请求了工具；
- 共发起 246 次工具请求，其中 221 次成功；
- 工具返回局部帧总数为 2,852；
- 115/228 个有工具请求的样本至少覆盖一个 released relevant clip；
- 对 Fair、Good、VeryGood 正例 clip 的平均 recall 均约为 22%。

这说明工具调用带来了整体正向差异，但工具使用仍然稀疏，且局部证据覆盖不完整，定位策略仍有较大改进空间。

### 9.3 成本

| 成本项 | Global 64 | Tool | Tool 相对增量 |
|---|---:|---:|---:|
| 总读取帧 | 99,200 | 102,052 | +2.88% |
| Prompt tokens | 15,657,153 | 18,097,492 | +15.59% |
| Completion tokens | 742,367 | 856,225 | +15.34% |
| 平均耗时/样本 | 9.09 秒 | 10.98 秒 | +20.82% |

## 10. LoRA SFT 支线的状态

项目还准备了一套独立的 Global-only LoRA SFT 方案，主要包括：

- 使用官方 QVHighlights train labels；
- 直接生成完整 2 秒 clip 分数数组；
- 以三名 annotator 分数均值作为 released clip 训练目标；
- 使用 BF16 LoRA，`r=16`、`alpha=32`、`dropout=0.05`；
- 只训练语言模型 28 层中的 196 个 attention/MLP 模块；
- 冻结 vision、merger/projector、embedding 和 `lm_head`；
- 按源 YouTube 视频分组划分 train/dev，避免视频泄漏；
- 配置 assistant-only supervision mask、FSDP、断点恢复和配对 bootstrap。

截至当前报告，LoRA 支线只完成了 Stage A 的数据、脚本和一致性验证，尚未执行任何 optimizer step，也没有 adapter、A/B 指标或置信区间。因此，当前不能声称 LoRA SFT 已经改善 QVHighlights 表现；完整 validation 的已报告结果全部来自冻结模型的 training-free 适配。

## 11. 主要文件

- 适配协议与运行说明：[README.md](README.md)
- 冻结推理配置：[configs/pilot.json](configs/pilot.json)
- Prompt、视频编码和评分转换：[scripts/inference_common.py](scripts/inference_common.py)
- Global/Tool 推理主程序：[scripts/run_inference.py](scripts/run_inference.py)
- Manifest 生成：[scripts/prepare_manifests.py](scripts/prepare_manifests.py)
- 严格官方评测包装：[scripts/evaluate_highlight.py](scripts/evaluate_highlight.py)
- 完整 validation 报告：[FULL_VALIDATION_REPORT.md](FULL_VALIDATION_REPORT.md)
- LoRA SFT 阶段报告：[lora_sft/REPORT.md](lora_sft/REPORT.md)

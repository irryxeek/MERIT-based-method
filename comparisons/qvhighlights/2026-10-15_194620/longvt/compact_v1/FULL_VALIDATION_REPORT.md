# LongVT-RFT 在 QVHighlights Validation 上的完整实验报告

## 1. 摘要

本实验在完整 QVHighlights validation 集上比较了两个冻结的、无需训练的推理方案：

- **Global 64**：对每段视频均匀读取最多 64 帧，不使用工具。
- **Tool**：使用与 Global 64 完全相同的全局帧，并允许模型自主调用最多 2 次 `crop_video`，每次最多读取 32 个局部帧。

两组均完成全部 **1,550** 个 query-video pair。按官方 QVHighlights highlight evaluator 计算，Tool 在 Fair、Good 和 VeryGood 三个阈值上的 mAP 与 Hit1 均高于 Global 64：mAP 分别提高 **1.21、1.10、0.70** 点，Hit1 分别提高 **2.19、2.19、2.06** 点。

该结果说明，在本次固定配置和完整 validation 集上，允许工具调用带来了方向一致的正向差异。不过，工具仅在 228/1,550 道题上被请求，且只有 115/228 道题的裁剪区间覆盖至少一个 released relevant clip；同时预测存在大量最高分并列。因此，本报告将官方指标、工具证据覆盖与并列敏感性分开呈现，不把诊断指标解释为官方成绩。

> **范围说明：**这是将冻结的 `longvideotool/LongVT-RFT` 检查点适配到 QVHighlights highlight detection 的 training-free 实验，不是 LongVT 论文中某项原始指标的严格复现，也没有进行 SFT、LoRA、RL 或 RFT 训练。

## 2. 实验身份与冻结项

| 项目 | 固定值 |
|---|---|
| LongVT 源码 commit | `08d755b973e4ad990ac5cdd64fb992c804d840db` |
| 模型 | `longvideotool/LongVT-RFT` |
| 模型 revision | `16ed9314e4ea600dc4682b172a614252f2f383ad` |
| 官方 evaluator | `jayleicn/moment_detr` commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca` |
| 推理服务 | vLLM 0.12.0，TP=2，GPU 6/7 |
| 服务最大上下文 | 65,536 tokens |
| 推理随机性 | `temperature=0`、`top_p=1`、`seed=42` |
| Global 帧预算 | 最多 64 帧 |
| Tool 局部预算 | 最多 2 次调用，每次最多 32 帧 |
| 单帧最大像素 | 50,176 |
| 分数范围 | 整数 0–4 |
| 输出表达 | 1 个背景分数，加最多 5 个局部时间覆盖区间 |
| 区间映射 | 2 秒 clip 的 midpoint 落入半开区间；重叠取最大分数 |
| 失败策略 | 一次格式重试后仍无效则使用全零向量，并保留在官方分母中 |

完整 validation manifest 含 1,550 个 query 和 1,519 个唯一视频。所有媒体均通过唯一性、时长和首/中/末帧解码检查。官方包中 8 个视频比 annotation duration 短 1.157–1.471 秒，因此媒体验收容差在推理前冻结为 1.5 秒；该调整不改变模型输入帧预算、prompt 或评分规则。

配置、manifest、媒体检查、ground truth 与关键脚本的 SHA-256 保存在 [`frozen_sha256.txt`](results/full_validation_v1/frozen_sha256.txt)。

## 3. 官方完整集结果

以下结果覆盖全部 1,550 道题，并包含预先声明的全零 fallback。

| 正例阈值 | 指标 | Global 64 | Tool | Tool − Global |
|---|---|---:|---:|---:|
| Fair（标注分数 ≥2） | HL-mAP | 30.96 | 32.17 | **+1.21** |
| Fair（标注分数 ≥2） | HL-Hit1 | 28.97 | 31.16 | **+2.19** |
| Good（标注分数 ≥3） | HL-mAP | 24.47 | 25.57 | **+1.10** |
| Good（标注分数 ≥3） | HL-Hit1 | 26.97 | 29.16 | **+2.19** |
| VeryGood（标注分数 ≥4） | HL-mAP | 13.90 | 14.60 | **+0.70** |
| VeryGood（标注分数 ≥4） | HL-Hit1 | 20.65 | 22.71 | **+2.06** |

Tool 在六项官方指标上均为正向差异。该报告未进行置信区间或显著性检验，因此这些差异应描述为本次固定实验中的观测增益，而不是统计显著性结论。

## 4. 配对逐题分析

两组的 1,550 个 qid 完全匹配，且每道题输入的全局帧哈希均一致：

- 相同全局帧：1,550/1,550
- 不匹配：0
- 两组预测向量完全相同：72/1,550（4.65%）
- 两组均产生有效预测：1,506/1,550（97.16%）

### 4.1 逐题胜、平、负

| 阈值 | 指标 | Tool 更好 | 相同 | Tool 更差 |
|---|---|---:|---:|---:|
| Fair | HL-mAP | 429 | 798 | 323 |
| Fair | HL-Hit1 | 138 | 1,308 | 104 |
| Good | HL-mAP | 431 | 785 | 334 |
| Good | HL-Hit1 | 132 | 1,320 | 98 |
| VeryGood | HL-mAP | 408 | 834 | 308 |
| VeryGood | HL-Hit1 | 109 | 1,364 | 77 |

在三档阈值和两类指标中，Tool 更好的题数均多于 Tool 更差的题数。大量逐题相同主要与输出分数粒度低、背景分数覆盖范围大以及最高分并列有关。

### 4.2 共同成功子集

在两组均产生有效预测的 1,506 道题上，Tool 相对 Global 的差异为：

| 阈值 | mAP 差异 | Hit1 差异 |
|---|---:|---:|
| Fair | +1.40 | +2.46 |
| Good | +1.26 | +2.46 |
| VeryGood | +0.81 | +2.19 |

共同成功子集与固定 fallback 的全量结果方向一致，说明总体正向差异并非仅由两组失败数量不同造成。

## 5. 完成率与失败处理

| 项目 | Global 64 | Tool |
|---|---:|---:|
| 总样本数 | 1,550 | 1,550 |
| 有效预测 | 1,534 | 1,522 |
| 有效完成率 | 98.97% | 98.19% |
| 全零 fallback | 16 | 28 |
| 成功但存在工具失败 | 0 | 12 |

所有 fallback 均由冻结协议自动处理，没有删除失败样本或事后选择性重跑。主要最终解析失败来自无效区间边界、超过 5 个覆盖区间或格式重试后仍不满足严格 schema。运行中没有 OOM、服务中断或视频读取失败导致的批量中止。

格式解析允许且逐次记录两种预声明的确定性语法清理：移除最外层 Markdown fence，以及仅在分隔符计数能够证明只缺一个外层对象右括号时补上该括号。其他截断、越界或语义错误不会被静默修复。

## 6. 工具调用与证据覆盖

### 6.1 调用执行

| 项目 | 数值 |
|---|---:|
| 使用工具的样本 | 228/1,550（14.71%） |
| 工具请求数 | 246 |
| 成功工具调用 | 221 |
| 工具调用成功率 | 89.84% |
| 单样本最大执行次数 | 2 |
| 返回的局部帧总数 | 2,852 |

25 次失败工具调用包括：13 次无效裁剪区间、8 次过短区间导致无法满足最少帧数、2 次视频路径不匹配、1 次未知工具名以及 1 次缺少 `end_time`。失败调用被完整记录，模型仍可在可行时使用全局观察完成预测。

### 6.2 对 released evidence 的覆盖

工具证据覆盖定义为：某个官方 2 秒 clip 的 midpoint 落在一次成功执行的裁剪区间内。该诊断使用 ground truth 进行事后审计，ground truth 不进入模型消息。

| 覆盖口径 | 有工具请求的合格样本 | 覆盖至少一个正例的样本 | 平均正例 clip recall |
|---|---:|---:|---:|
| Released relevant clip / Fair | 228 | 115（50.44%） | 22.41% |
| Good（≥3） | 228 | 111（48.68%） | 22.32% |
| VeryGood（≥4） | 226 | 103（45.58%） | 22.25% |

与 50 条 pilot 中仅 2/10 个工具样本覆盖 relevant clip 相比，完整集覆盖比例更高；但约一半工具样本仍未裁剪到 released relevant clip，且平均正例覆盖率约为 22%。因此，工具带来的整体增益是在工具使用稀疏、局部证据覆盖不完整的条件下观察到的，仍存在明显的定位改进空间。

## 7. 评分并列诊断

官方 Hit1 使用 `numpy.argmax`，最高分并列时固定选择最早的 clip。官方 mAP 使用 `sklearn.metrics.precision_recall_curve`，相同预测分数按阈值成组处理，不依赖同分 clip 的内部顺序。

| 并列统计 | Global 64 | Tool |
|---|---:|---:|
| 全常数预测 | 547（35.29%） | 446（28.77%） |
| 唯一最高分预测 | 336（21.68%） | 274（17.68%） |
| 平均不同分数数量 | 1.79 | 1.83 |
| 平均最高分并列 clip 数 | 34.25 | 31.97 |
| 最大最高分并列 clip 数 | 75 | 75 |

### 7.1 Hit1 tie sensitivity

下表中的“均匀随机解并列期望值”仅用于敏感性诊断，不是官方指标。

| 模式 | 阈值 | 官方首个最大值 Hit1 | 均匀随机解并列期望值 | 任一最高分 clip 命中 |
|---|---|---:|---:|---:|
| Global | Fair | 28.97 | 32.36 | 65.87 |
| Global | Good | 26.97 | 30.52 | 65.16 |
| Global | VeryGood | 20.65 | 24.80 | 61.48 |
| Tool | Fair | 31.16 | 35.15 | 68.32 |
| Tool | Good | 29.16 | 33.10 | 67.42 |
| Tool | VeryGood | 22.71 | 27.06 | 63.23 |

两组均存在明显的同分退化。官方 Hit1 必须以首个最大值规则为准，但其绝对值对并列消解规则敏感。Tool 在官方规则和诊断性随机并列期望下均高于 Global，因而本次组间方向并未因该诊断而反转。

## 8. 成本与运行效率

| 成本项 | Global 64 | Tool | Tool 相对增量 |
|---|---:|---:|---:|
| 全局帧 | 99,200 | 99,200 | 0% |
| 局部帧 | 0 | 2,852 | — |
| 总读取帧 | 99,200 | 102,052 | +2.88% |
| 平均读取帧/样本 | 64.00 | 65.84 | +2.88% |
| Prompt tokens | 15,657,153 | 18,097,492 | +15.59% |
| Completion tokens | 742,367 | 856,225 | +15.34% |
| 平均耗时/样本 | 9.09 秒 | 10.98 秒 | +20.82% |
| 中位耗时/样本 | 7.87 秒 | 9.59 秒 | — |
| P95 耗时/样本 | 16.30 秒 | 19.42 秒 | — |
| 样本耗时合计 | 3.91 小时 | 4.73 小时 | +20.82% |

运行从 2026-09-21 16:17 左右开始，于 2026-09-22 01:13:46（Asia/Shanghai）写入完成标记。两组顺序执行，端到端墙钟时间约 8 小时 57 分钟。

GPU 6、7 上采样到的显存峰值均为 16,941 MiB。该值来自 `nvidia-smi` 的设备总显存采样，包含常驻 TP=2 vLLM 服务，不是单请求增量显存或框架 allocator 峰值。

## 9. 结论

1. 完整 validation 的结果与 50 条 pilot 的不稳定方向不同：Tool 在六项官方指标上均取得正向差异。
2. 增益幅度为 **+0.70 至 +1.21 mAP**、**+2.06 至 +2.19 Hit1**，共同成功子集结论一致。
3. Tool 的代价是总读取帧增加 2.88%、token 使用增加约 15%、平均延迟增加 20.82%。
4. 工具仅在 14.71% 的样本上被使用，成功调用中仍存在较弱的 released evidence 覆盖，说明工具定位策略尚有优化空间。
5. 大量常数预测和最高分并列限制了分数分辨率，并使官方 Hit1 的绝对值对“首个最大值”规则敏感；这不改变本次 Tool 相对 Global 的方向，但应作为解释结果的重要限制。
6. 本实验支持“在该冻结 training-free adaptation 下，Tool 优于 Global 64”的结论，但不应外推为 LongVT 原论文指标复现、训练收益或统计显著性结论。

## 10. 产物索引

- 冻结配置：[`config_frozen.json`](results/full_validation_v1/config_frozen.json)
- 冻结 manifest：[`manifest_frozen.json`](results/full_validation_v1/manifest_frozen.json)
- 文件哈希：[`frozen_sha256.txt`](results/full_validation_v1/frozen_sha256.txt)
- Global 官方指标：[`longvt_global/metrics.json`](results/full_validation_v1/longvt_global/metrics.json)
- Tool 官方指标：[`longvt_tool/metrics.json`](results/full_validation_v1/longvt_tool/metrics.json)
- 完整汇总与逐题指标：[`summary.json`](results/full_validation_v1/summary.json)
- Tie 与工具证据覆盖审计：[`paired_audit.json`](results/full_validation_v1/paired_audit.json)
- 逐题差异 CSV：[`per_question_differences.csv`](results/full_validation_v1/per_question_differences.csv)
- Global 完整轨迹：[`longvt_global/traces.jsonl`](results/full_validation_v1/longvt_global/traces.jsonl)
- Tool 完整轨迹：[`longvt_tool/traces.jsonl`](results/full_validation_v1/longvt_tool/traces.jsonl)
- 完整运行日志：[`run.log`](results/full_validation_v1/run.log)


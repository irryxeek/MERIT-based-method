# REZE / QFVS 概念提示与时间选择优化

日期：2026-10-22。依据：[四查询诊断](reze_qfvs_analysis_2026-10-20.md)。本轮是已用于诊断的四查询开发消融，不是全部 180 查询或独立确认集。原 B0 运行、预测和指标保留。

## 冻结的优化与对照

| 条件 | 窗口评分 | 摘要选择 | 新模型调用 |
|---|---|---|---:|
| B0_topk | 原严格动作提示，复用已核验原分数 | 原逐片段 Top-2% | 0 |
| B0_temporal | 与 B0_topk 相同 | 固定时间邻域抑制 | 0 |
| Concept_topk | 视觉概念出现提示 | 原逐片段 Top-2% | 20,356 |
| Concept_temporal | 复用 Concept_topk 分数 | 固定时间邻域抑制 | 0 |

另保留同预算 Uniform 对照。四条件同时报告，不仅挑选得分最高的组合。

概念提示检查两个概念中至少一个是否在任一输入帧可见，保留 OR 语义。Building 明确指建筑物，不指搭建或组装动作。该改动同时包含任务提示与名词消歧，不能将收益单独归因于其中一个。

时间选择仍按分数从高到低、同分按原片段 ID 排序；选中一个 5 秒片段后，抑制中心距离不超过 10 秒的邻居，直到选满 `floor(n_shots × 0.02)`。只固定这一半径，不根据四查询结果搜索参数。选择只读分数和时间索引；标签只用于后续 CPU 评价。时间分散是覆盖代理，不等同于语义去重。

## 固定项与配对检查

- 原 P01 Desk/Face、P02 Building/Boat、P03 Book/Lady、P04 Garden/Hall；完整原视频与原查询顺序。
- Qwen3-VL-8B-Instruct BF16，原固定权重 revision；3 秒窗口、2 fps、完整窗口六帧及原缩放。
- REZE Yes/No 概率计算、逐秒读出到原生 5 秒网格、2% 摘要预算和冻结官方语义匹配指标。
- 新窗口在模型调用前逐项核对 B0 的帧索引、请求时间、实际时间戳、原像素与模型输入像素哈希；不匹配即停止。
- 主报 Oracle P/R/F1，辅报三位用户均值；逐查询补充概念命中数、连续区间、最长连续片段及时间覆盖。

## 执行与资源

实现位于 [`baselines/zeroshot/experiments/reze_qfvs/`](../baselines/zeroshot/experiments/reze_qfvs/README.md)。新运行目录为 `$ZEROSHOT_ARTIFACT_ROOT/reze_qfvs_concept_pilot4_v1/`。

新增参数：`prepare.py --scope pilot4 --prompt-mode visual_concept_or_v1 --paired-baseline OLD_RUN --run NEW_RUN`。随后从新运行的 `source_snapshot/` 执行 `preflight_media.py` 和 `dispatch.py`。仅使用用户本轮指定的 6 号卡，并核对 UUID 与空闲状态。

先用 `compare_adaptation.py --run NEW_RUN --baseline-only` 计算原分数的选择消融；新推理完成后自动运行完整四组评价。准备时冻结源码、模型、输入和原运行文件哈希。逐批写检查点，最多三个 worker 尝试；正式 20,356 次、工程检查 12 次与补试共享 **20,416 次请求上限**，不自动扩大到 180 查询。

## 当前验证与结果位置

新增与原 QFVS CPU 测试合计 26 项通过、1 项因缺少本工作副本的 QV 媒体清单跳过。Ruff 检查通过。4 条真实视频的首、中、尾共 12 个预处理探针全部通过，与 B0 输入像素一致；最大输入为 614 token，低于 4,096 上限。新提示正式评分尚未开始。

- `comparison_baseline_only.json` / `.md`：原分数的 Top-k、时间抑制与 Uniform 对照。
- `comparison.json` / `.md`：完整四组比较及成本，须等新推理和配对核验完成。
- `status.json`、`worker_attempts.json`：进度、等待设备和有界恢复记录。
- `paired_baseline.json`、`binding.json`：原 B0 证据与本轮冻结协议。

这轮首先检验概念歧义与选择集中问题；达到简单对照水平后，再单独确定未参与诊断的查询确认，不把开发集改善当作完整数据集结论。

## 已完成的原分数选择消融

该表使用原分数，新增模型调用为零。完整指标与所选片段见[机器可读结果](artifacts/reze_qfvs_adaptation_2026-10-22/baseline_controls.json)。

| 条件 | Oracle P / % | Oracle R / % | Oracle F1 / % | 用户均值 F1 / % |
|---|---:|---:|---:|---:|
| B0_topk | 40.81 | 37.20 | 38.63 | 36.26 |
| B0_temporal | 42.78 | 38.84 | 40.41 | 37.81 |
| Uniform | 54.10 | 47.30 | 50.15 | 46.44 |

时间选择让四查询 Oracle F1 增加 **1.78 pp**，仍低于 Uniform **9.74 pp**。P02 的查询概念命中仍为 **0/73**；单靠时间分散没有修复其概念排序问题。P03 的 F1 从 49.36 升至 52.49，同时查询概念命中从 32/43 降到 17/43，说明概念命中与摘要覆盖指标并非同一目标。尚无新提示结果，不能据此确认 Building 歧义已解决。

## 自动执行状态

2026-10-22 21:20（北京时间）检查时，`reze-qfvs-concept-pilot4-v1.timer` 已启用，约每分钟检查一次，仅等待本轮指定的 6 号卡。该卡仍有其他账号的计算进程，当前状态为 `waiting_for_idle_gpu`，正式新评分 **0/20,356**。用户服务已开启 linger，SSH 断开后计时器仍可工作。

设备满足原空闲检查后自动加载模型，先完成 12 次 GPU 工程调用，再进行正式评分、Oracle/用户评价及四组比较。已有单卡原运行约 35 分钟，仅作新运行的量级参考；排队时长无法由此估计。

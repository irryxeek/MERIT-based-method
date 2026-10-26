# ReVALUE 观察输出字符限制探针：冻结协议 v1

日期：2026-10-26。CPU 来源压缩已经通过 14 视图与 16 条兼容性检查。本探针承接[本轮执行协议](revalue_memory_compaction_protocol_2026-10-26.md)的内容问题分支，由 `academic-research-suite / experiment-agent` run 模式执行。

## 问题与唯一改动

父单帧观察有 157 条 `visible`、584 条 `uncertainty` 达到 160/80 字符上限，但不能把达到上限直接视为语义错误或 token 截断。本轮检验：**在其余输入完全一致时，移除两个 JSON 字符上限能否减少句子片段，同时维持可解析、非空输出？**

新输出 schema 仅删除 `visible.maxLength=160` 和 `uncertainty.maxLength=80`。提示、两字段名称、必填约束、禁止额外字段、视觉输入、解码参数及 160 个输出 token 上限不变。使用同一 Qwen3-VL-8B-Instruct revision、BF16、temperature 0 和 compact EBNF 后端。两字段皆空仍为无效；不自动重写、裁剪、补句或语义重试。

这不是物体识别修复、动作理解实验或正式观察器替换。长描述保留原文，不能截回旧长度伪造兼容；新的事实入库策略尚未验证。

## 预先固定样本

从父目录 `revalue_frame_grounding_v1_2026-10-26/` 选择：

1. 固定 28 个观察窗口各取第 12 号位置（零起始），共 28 次；位置规则不读取输出或 GT。
2. 加入历史审阅中的十个 `(qid, frame_id)` 回归案例：七个已纠正场景案例、三个残余误判案例。
3. 按逻辑请求键去重并稳定排序，得到 **37 次请求**，覆盖七条 query、两种观察臂和全部 28 个窗口。

这是开发诊断抽样，含按历史错误选取的回归案例，不能估计总体准确率。原 672 次单帧输出作为配对参照，不重跑旧条件；保留其模型、提示、像素、帧时间和账本收据。新请求仅允许上述两处 schema 改动及由此必然改变的 grammar hash。

## 成本与执行

| 约束 | 绝对上限 |
|---|---:|
| 新逻辑请求 / 尝试 | 37 / 37，追加 0 |
| 每次输出 | 160 token |
| 每次真实多模态输入＋输出预留 | 2,048 token |
| 全局实际／保守预留 token | 75,776 = 37 × 2,048 |
| GPU | 一张当前空闲卡，服务生命周期最多 900 秒 |
| 模型展示 | 74 帧次，37 张原输入帧各复制两次 |

用一张卡避免小探针重复加载模型；任意空闲卡均获用户授权，不占用或停止他人任务。监督进程记录开始、PID、GPU UUID、输出增长与结束；逐调用账本先预留后发送。预算含模型加载、失败和退出清理，880 秒开始终止自己的进程组，最长 15 秒后强制清理，保留失败记录。完成后核查该进程组及 GPU 服务退出。中断不重置预算、不自动续跑失败请求。

运行器：`experiments/revalue/observer_output_probe.py`。在 `$ZEROSHOT_ROOT` 执行：

```bash
"$INFERENCE_PYTHON" experiments/revalue/observer_output_probe.py prepare \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_observer_output_v2_2026-10-26" \
  --reference "$ZEROSHOT_ARTIFACT_ROOT/revalue_frame_grounding_v1_2026-10-26" \
  --protocol "$MERIT_ROOT/docs/revalue_observer_output_probe_protocol_2026-10-26.md"
"$INFERENCE_PYTHON" experiments/revalue/observer_output_probe.py run \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_observer_output_v2_2026-10-26"
```

准备阶段冻结 37 条清单、原响应和父记录哈希、源码/提示/schema 快照、模型版本、预算及本协议。新目录与父产物隔离，不修改既有事实或 HD 预测。

## 评价与停止规则

报告全部 37 条的结构通过、token 截断、空/标点描述、两字段字符/token 长度和旧上限兼容性；计入全部失败。助手审阅配对句子完整性与具体视觉支持，采用“支持 / 矛盾 / 无法判断”，并区分完整句与语义正确。独立人工审阅保持未评，不以修好十个案例推算泛化。

若输出完整性改善但物体/动作误判仍在，仅保留为输出接口候选；不启动 A/B/C 或正式 100 条。若 token 截断增加、完整性无改善或产生新的空输出，记录负结果，停止该修复分支。不得在同一运行中加长 token 上限、改提示或再次选择帧使其通过。

本轮不生成新 HD 指标。只有自动观察器经过固定 28 窗口、全部拟入库事实的内容检查，才冻结匹配事实集开展 A/B/C；七条诊断仍不能证明普遍精度收益。

执行前收据修订：初始 v1 目录仅完成准备、模型请求为 0。代码审查补齐继承收据验证、异常退出清理和最终完成哈希后，在 v2 新目录冻结实际运行；样本、假设、schema 两处改动及预算不变。v1 准备快照保留，不重置任何已发生的模型成本。

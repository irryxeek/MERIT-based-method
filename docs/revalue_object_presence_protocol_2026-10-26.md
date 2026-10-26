# ReVALUE 定向物体证据核查：冻结执行协议

## Material Passport

- Origin Skill: experiment-agent / academic-research-suite 3.23.2
- Origin Mode: run
- Origin Date: 2026-10-26；精确冻结时间见 manifest
- Verification Status: UNVERIFIED
- Version Label: object_presence_v1

## 1. 问题与边界

[上一轮结果](revalue_memory_compaction_results_2026-10-26.md)显示，来源压缩通过容量检查，移除字符上限改善了描述完整性，但未解决视觉误判。本轮只检验：**对于指定区域中的指定物体，明确检查可见特征是否比直接判断更能拒绝无依据断言，同时保留真实可见物体？**

这是 ReVALUE 定向查证模块的开发探针。问题由助手在推理前固定，尚未实现自动发现证据缺口或自动选帧；使用已经读取的同一帧，不增加视频时刻。这里的“核查”不代表已接入 Agent 工具链，不改变正式观察器、活动记忆、候选排序或 HD 预测。

## 2. 固定样本与预先参考

从已完成的 37 张输出探针图中固定 15 张，覆盖原七条 query；每图两个物体问题，共 **30 个病例**。保留手机、狗、jet bridge 三个历史困难对象，加入同图清晰可见的常见对象、可明确排除的对象和模糊画面。清单见代码内 `fixtures/object_presence_cases_v1.json`；运行时原样复制到 `reference_labels.json`，在首次请求前哈希冻结。

协作助手在查看原图后预标 `supported / contradicted / cannot_determine` 及理由；它可以了解历史诊断，未见本轮输出。模型请求仅获取物体、区域和中性存在性断言，不读取旧 caption、参考标签、参考理由或高光 GT。父代理负责实现、运行、结果核对。两者属于同一模型家族的助手工作，**不是独立人工标注或跨模型验证**。

- `supported`：指定区域中有清晰可辨的该物体。
- `contradicted`：指定区域足够清晰，明确没有该物体或显示不兼容对象。
- `cannot_determine`：模糊、遮挡、画外或外观歧义不足以判断；不能把未看清当作不存在。

只处理静态物体/身体部位存在性，排除身份、性别、动作完成、房间关系及情绪强度。病例是刻意选择的开发诊断，30 个问题不独立，不能作为总体准确率或泛化证据。

## 3. 唯一对照因素

两臂共享 Qwen3-VL-8B-Instruct 的本地冻结权重、BF16、temperature 0、同一帧像素/PTS、后端和 JSON 字段 `evidence,status,uncertainty`。没有字符长度上限，均有 **160 output token** 硬上限。

| 条件 | 提示差异 | 请求数 |
|---|---|---:|
| direct | 定义三种状态，直接核查物体存在性并给可见证据 | 30 |
| evidence_first | 在同一基础提示中加入“检查轮廓/部件，区分手、阴影、水花、背景，不从姿势或场景猜测”的步骤 | 30 |

每题交替 A→B / B→A 顺序。每次均是独立上下文，无前一次答案。证据优先提示更长，输入 token 成本据实报告；不声称两组 token 完全相同。禁止看到结果后改问题、标签、阈值或重试答案。

## 4. 预定读数与推进条件

报告每臂与预先助手标签的一致数、明确正例保留数、非正例被肯定数、无法判断数、结构失败/截断数，以及完整混淆表和三例困难对象的逐例结果。另审阅解释是否引入额外不受支持的事实。标注分歧原样保留，追加审阅而不覆盖预标签；不计算显著性。

`evidence_first` 进入扩大内容审查的必要条件：

1. 60 次结构与 token 完整性全部通过；失败不从分母排除。
2. 三个已知非支持对象均不能被判为 `supported`。
3. 明确正例保留数至少 `ceil(0.90 × 正例数)`，排除一律弃答策略。
4. 非正例肯定数不高于 direct，参考一致数不低于 direct。
5. 解释内容审阅没有发现足以否定上述结论的额外无依据断言。

通过也仅允许规划固定 28 窗口、全部拟入库事实的扩大内容审查；**本次不会自动启动 A/B/C 比较、正式 100 条或全量推理**。原 +1 pp mAP 等门槛保留。若 direct 已达到相同读数，报告两者无明确收益差距及额外提示成本，不将提示变长解释为机制提升。

## 5. 预算、运行与冻结

- 最大 60 次新模型请求、**零重试**；单次真实多模态输入加输出预留 ≤2,048 token，总计 ≤122,880，每臂 ≤61,440。
- 一张启动时再次确认空闲的 GPU，最多 900 秒生命周期（0.25 GPU 小时），含加载/退出。最后 60 秒不接受新请求；880 秒开始清理本运行的进程组。
- 每 30 秒监控进程和输出，90 秒无变化仅提示；仅预算到期或所属运行收尾清理时终止该运行进程，不影响其他任务。
- 15 张不同源帧，每次为编码器复制两次；60 次共展示 120 帧次。没有新源时间点，不能推断动作或时间变化。
- 原运行、已完成预测、历史计划和 AGENTS.md 保留。新产物放在 `$ZEROSHOT_ARTIFACT_ROOT/revalue_object_presence_v1_2026-10-26/`。

在 `$ZEROSHOT_ROOT` 使用既有 inference Python 环境执行：

```bash
python experiments/revalue/object_presence_probe.py prepare \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_object_presence_v1_2026-10-26" \
  --reference "$ZEROSHOT_ARTIFACT_ROOT/revalue_observer_output_v2_2026-10-26" \
  --cases experiments/revalue/fixtures/object_presence_cases_v1.json \
  --protocol "$MERIT_ROOT/docs/revalue_object_presence_protocol_2026-10-26.md"
python experiments/revalue/object_presence_probe.py run \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_object_presence_v1_2026-10-26"
python experiments/revalue/object_presence_probe.py analyze \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_object_presence_v1_2026-10-26"
```

冻结协议、病例、两臂文本、schema、源帧收据、依赖代码和模型指纹；逐次保留原始响应、账本和技术状态。完成收据绑定 manifest、账本、资源和摘要，另保存配对分析与解释审阅。结果报告、代码和索引同步到本地发布目录，本次不执行 Git 推送。

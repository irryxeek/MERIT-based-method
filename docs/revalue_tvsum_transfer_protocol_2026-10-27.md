# ReVALUE 跨数据集测试：TVSum val10 冻结协议

## Material Passport

- Origin Skill: academic-research-suite 3.23.2 / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27（北京时间）
- Verification Status: UNVERIFIED；本页为运行协议，不代表实验已完成
- Version Label: tvsum_transfer_v1

## 1. 目的与授权范围

用户要求“进行跨数据集测试”。本轮执行刚讨论的第一阶段：固定 TVSum UMT val10，比较 REZE B0、P_pair、C_direct4、E_memory。检验具体 query 之外的通用重要性任务是否适合候选比较和决策记忆。它是开发迁移诊断，不是未见确认集，也不是 REZE 原论文的 TVSum 成绩。

本轮不执行 QV ROI、新回看、SVHighlights 或其他数据集，不改变旧失败结论和 R2 扩大门槛。后续阶段另行冻结适配协议。

## 2. 输入与四组条件

固定 UMT revision `bde05e3de39049bf90d8807f6b2a4ed8b42acbb0` 的 val10；从原始划分源码读取 ID，不按标注或模型表现选样。保留通用输入 `Highlight of this video.`，不加入标题、类别、音频、真实高光数量或摘要标签。

| 条件 | 内容 | 每视频新增请求上限 |
|---|---|---:|
| B0 | 核验并复用已完成 REZE 分数 | 0 |
| P_pair | 原 Yes/No top-20 概率双向比较与排序规则 | 20 |
| C_direct4 | 原任务解析与四态直接比较，无记忆 | 21 |
| E_memory | 同一任务解析、5 次事实提取、1 次决策记忆、20 次四态比较 | 27 |

任务解析在 C_direct4/E_memory 间共享，因此实际新增上限为每视频 **67 次**，十视频 **670 次**：600 次比较＋10 次任务解析＋50 次事实提取＋10 次记忆构建。没有普通记忆臂，也没有观察调用。

候选固定为 B0 原 Top-5。每候选最多四张 B0 已见帧；重新解码历史批次只用于逐像素与 PTS 核验，不增加模型所见源帧。所有方法使用同一候选与像素；仅置换候选已有分数，候选外完整评分序列不变。活动记忆含来源信息最多 1,024 tokenizer token；比较器仍读取旧帧，不属于纯文本 memory-only。

Qwen3-VL-8B-Instruct revision 固定为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、temperature 0。复用完整权重哈希凭证，检查文件大小和 mtime 未变，重新核验 tokenizer/config 等小文件哈希；不把该步骤称为本轮重新计算全部权重哈希。

P_pair 直接复用原提示、Yes/No token 与解析、0.6/0.4 双向支持规则及原分数置换。统一使用现有 ReVALUE 单请求引擎（6,144 上下文）；相较历史 P_pair 的批处理部署有所变化，记录配置，不宣称吞吐或逐位输出一致。

## 3. 对照隔离与失败处理

- 事实提取和记忆压缩不读取 query；C_direct4/E_memory 使用同一 TaskSpec、比较提示、四态 schema 与排名公式，仅可选记忆不同。
- 记忆构建语义失败不阻止 C_direct4；语义失败保留原响应，无重生成。
- P_pair 失败回退 B0；C_direct4/E_memory 失败沿用历史协议回退 P_pair。全部十视频保留，失败单列；回退方法的独立运行成本追加其使用的 P_pair 依赖。
- 分片独立负责五个视频，任务不重复。有效 tie/insufficient 与技术回退分开。
- 固定完整预测和收据后才读取标注评价。推理模型看不到参考标签、标题、成绩或选样分组。

## 4. 预算和监控

两张启动前确认空闲的 GPU，各一模型实例；按用户已有授权可使用任何空闲卡，不预留 GPU 6，不影响其他任务。独立视频并行，所有开销累计。

| 项目 | 上限 |
|---|---:|
| 正式请求／全部模型尝试 | 670；每分片 335 |
| 额外模型 smoke／重试 | 0；首个正式请求兼作运行观察 |
| 单请求输入＋输出预留 | 6,144 token，按真实多模态展开计算 |
| 总 token | 4,116,480；每分片 2,058,240 |
| 输出上限 | P_pair 1；任务 256；事实 512；记忆 1,024；四态比较 384 |
| GPU 生命周期 | 每分片至多 3,600 秒，总计至多 2 GPU 小时，含加载和清理 |

每 30 秒检查进程与账本进度；90 秒停滞仅提示。硬超时预留清理时间，仅终止本轮拥有的进程组。后台服务脱离 SSH 会话；正常结束、监督器收到终止信号或超时均清理所属模型进程。保留失败和部分产物；不自动重新启动失败轮或清零预算。

## 5. 评价和结果解释

主报固定 UMT val10 的 **Top-5 mAP**，核验原生评价器与独立实现一致；不将其称为摘要 F1。逐视频列出四组结果、配对差值、修正/退化数量、候选同分、未决率、技术回退和成本。

E_memory−C_direct4 用于记忆贡献诊断；P_pair−B0 用于原候选复评迁移诊断。十个视频各自是统计单位，20 名标注者、候选对和两方向不当作独立视频。主报描述性结果，不以十条中最高成绩宣称稳定泛化，也不据此降低 QV 的原门槛。

所有比较和排序按原始响应重放，保存记录哈希和完整分母。初始化缓存复用只减少本轮新增支出，不将该方法初始化成本归零；每视频正式初始化成本和新请求成本分列，原 B0 共享 smoke/失败开销另行说明。

若有可解释、可追溯的收益，再考虑锁定版本与独立样本；若候选已同分或缺少关键证据，记录适用范围，不在本轮偷偷扩充候选、更换提示或引入回看。机器合法引用不等于事实准确，独立人工语义复核状态保持未评。

## 6. 实现、运行和交付

新实现位于发布仓库 `baselines/zeroshot/experiments/revalue_crossdataset/`。保留原研究源码、历史计划、预测和 AGENTS.md；本轮按新冻结源码快照执行。

```bash
"$REPRO_PYTHON" experiments/revalue_crossdataset/run.py prepare \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27" \
  --baseline "$ZEROSHOT_ARTIFACT_ROOT/reze_tvsum_full50_v1" \
  --protocol "$MERIT_ROOT/docs/revalue_tvsum_transfer_protocol_2026-10-27.md" \
  --model-receipt "$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_v3_2026-10-26/model_verification.json"
"$REPRO_PYTHON" "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27/source_snapshot/experiments/revalue_crossdataset/run.py" run \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27"
```

启动前检查 `REZE_MODEL_PATH`，通过 CPU 测试和独立代码检查后冻结 manifest。交付媒体/候选缓存收据、原始请求响应、账本、完整预测、逐视频评价、资源退出收据和 Markdown 报告。用户后续要求推送时，报告、代码和索引一起同步。

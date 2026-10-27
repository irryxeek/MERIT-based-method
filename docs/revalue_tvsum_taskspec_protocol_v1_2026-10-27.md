# TVSum 通用 TaskSpec 对象约束：最小单变量验证协议 v1

## Material Passport

- Origin Skill: academic-research-suite 3.23.2 / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-27（北京时间）
- Verification Status: UNVERIFIED；CPU 输入准备通过，尚无新模型预测
- Version Label: tvsum_taskobject_v1_proposed
- Request Marker: `MERIT_TVSUM_OFFLINE_20261027_1500_V1`
- Execution Authorization: **PENDING；本协议不是运行授权或已启动任务**

## 1. 要检验的唯一问题

检验通用请求被误写为对象约束，是否影响四态比较的资格判断和观看价值排序。依据[离线诊断](revalue_tvsum_offline_diagnosis_v1_2026-10-27.md)，父轮十条 TaskSpec 均将 `Highlight of this video` 当作 `object`，但尚未证明这就是所有 tie 的原因。

**唯一干预是删除这一条对象约束。** 原 query、request_type、purpose 原样保留；不更换任务提取提示、不重跑任务解析，不改比较提示、输出 schema、记忆、排序、评价、候选、旧帧或模型。目的在于隔离该中间字段的作用，不是在本轮开发通用任务解析器。

```json
{
  "query": "Highlight of this video.",
  "request_type": "generic",
  "purpose": "Highlight of this video",
  "constraints": []
}
```

只允许从父轮精确的单条 `{"kind":"object","quote":"Highlight of this video","number":null}` 转为 `[]`。遇到不同 query、非 generic 类型、其他或额外约束、输入字段变化就停止准备；不得清空真实对象、事件、数量和时长限制。

## 2. 完整范围与条件

使用原 UMT val10 **全部十视频**，顺序与父轮冻结清单一致：

```text
JKpqYvAdIsw, JgHubY5Vw3Y, sTEELN-vY30, Yi4Ij2NM7U4,
fWutDQy1nnY, Bhxk-O1Y7Ho, Se3oxnaPsz0, cjibtmSLxQ4,
J0nA4VgnoCo, E11zDS9XGzg
```

不按 GT、原指标、是否有未知项或旧结果好坏选样。这是已查看结果的开发集，不是未见确认集。

| 条件 | TaskSpec | 记忆与输入 | 新增正式调用（待授权） |
|---|---|---|---:|
| C_direct4 | 父轮错误对象约束保留 | 原 C_direct4 缓存 | 0 |
| C_remove | 仅删除该对象约束 | 原候选旧帧，无记忆 | 200 |
| E_memory | 父轮错误对象约束保留 | 原 E_memory 缓存 | 0 |
| E_remove | 仅删除该对象约束 | 同一父轮记忆和候选旧帧 | 200 |

每新条件为 `10 视频 × 5×4 有向比较 = 200` 次，共 **400 次**。原 B0、P_pair 作为冻结参考，不重跑。事实、记忆和解析全部精确复用；旧比较响应只作为旧条件，**不能冒充改输入后的模型输出**。

主对比为 **C_remove−C_direct4**，用于减少错误记忆对约束效应解释的干扰；E_remove−E_memory 为预定的第二层检验。E_remove−C_remove 及两层效应差为辅助诊断，不在看完结果后挑最高条件作为唯一主结果。P_pair 不依赖 TaskSpec，此干预不检验其下降原因。

## 3. 必须固定的输入和实现

- 原 Top-5 候选及全部评分、NPZ 像素、PTS、帧元数据、每候选四张旧帧；无标题、类别、音频、ROI 或新源时刻。
- 原 query 与目的；原比较系统提示、正文模板、双侧引用集合、输出 schema/grammar、384 输出 token 上限；正常 tie/insufficient 均允许。
- 原 E_memory 活动视图、事实和引用，含已知的不确定性遗漏与可疑事实；本轮不修记忆、不增加来源压缩、不重建事实。
- Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、temperature 0、seed 0，冻结 ReVALUE 引擎及原软件版本；不新增跳过检查参数。
- 原双向一致胜负规则、原分数置换和候选外不变规则；不得直接把 unknown 改成 satisfied 或人为改写比较答案。
- 原生 TVSum UMT Top-5 mAP，包含原同分行为；既有稳定排序敏感性结果只作背景。本轮不更换主评价器以规避同分问题。

已准备的 400 份输入只在 `specification.text` 和 `rendered_prompt` 内替换同一个 TaskSpec JSON；其它字段逐项一致。输入长度会自然变化，后续实际展开 token 必须重新计费；本轮没有调用 tokenizer 或模型来估算新输入的真实多模态 token。

这是历史缓存对照，并非两组同时重新推理。启动前必须核验模型、环境、源代码、旧控制响应及请求签名；若无法保持等价条件，停止并修订协议，不能自动把 400 次扩大为 800 次。

## 4. 独立预算建议与授权边界

**以下全是新 TVSum 单变量轮的待授权建议，当前已用为 0。** 原 TVSum transfer 的 670/670 请求已耗尽，剩余 token/时间不能兑换新请求；QV 剩余额度也不转入本轮。

| 项目 | 建议硬上限 |
|---|---:|
| 正式比较 | 400：C_remove 200、E_remove 200 |
| 故障额外调用、语义重试、额外 smoke | **全部 0** |
| 所有模型尝试合计 | **400**，失败及未结算预留同样占用 |
| 单请求真实输入＋输出预留 | 6,144 token；输出至多 384 |
| 绝对新增 token 上限 | **2,457,600 = 400 × 6,144** |
| 总 GPU 生命周期 | **3,600 秒＝1 GPU 小时**，含加载、处理、等待、失败和清理 |
| 建议并行分片 | 两张空闲卡各五视频、200 请求、1,228,800 token、1,800 秒 |
| 新源帧／新增展示帧次 | 0／至多 3,200；展示旧帧也计成本 |
| 新解析／事实／记忆／P_pair 调用 | 全部 0 |

实际执行需要用户明确授权这一**独立预算及完整十视频范围**，并完成新运行器审查、输入绑定与资源检查。现有“可用空闲 GPU”授权不替代新轮的请求额度。预算不因换卡、重启、失败、进程中断或换目录而清零；跨分片共用总账，任何失联尝试保守占满其预留。

不分配 GPU、不预留队列、不启动服务。本协议没有自动续跑命令，也不授权任何远程模型 API 或新增付费服务。

## 5. 质量、停止和解释门槛

### 启动前

十条完整输入都通过精确约束检查，400 个新请求的非 TaskSpec 字段与对应旧请求一致；旧帧/候选/记忆/源文件收据、模型和环境检查通过。标签仍由独立评价阶段读取。核验失败则不启动，不删问题样本，不降级检查。

### 执行中

每次调用前累计预留真实多模态 token，并记录原始响应。遵守父轮语义失败规则：相应条件按原协议回退至已冻结的 P_pair，保留完整十视频分母并单列失败；不额外生成。运行时异常、来源漂移、数据串组、预算或时间到限则停止受影响运行，保存部分产物，不自动重新启动。正常 tie/insufficient 不算故障，也不触发重试。

状态收尾只写新轮自己的完成与退出收据；原分片遗留的 `generating` 状态不修改。独立运行器需明确完成凭证优先级并验证所属进程退出，不以 nanobind 退出提示自动重跑。

### 完成后

1. 先冻结完整预测，再计算同一原生指标。主报 C_remove−C_direct4；另报 E_remove−E_memory、逐视频改善/退化/相同、资格与四态转换、双向冲突、来源错误、技术回退和实际成本。
2. 审查全部新出现的胜负判断、因方向导致的矛盾、以及带来排序变化的依据。不得只审查指标改善的案例；新增胜负必须有对应旧帧支持，而不是从“无约束”推出“更好看”。机械合法与助手语义审查分别报告，独立人工审阅未完成仍注明未评。
3. **不以降低 tie 比例作为通过条件。** 如果仍合理并列，或只改变资格措辞而没有改善决策，保留该结果；如果增加无依据的胜负，视为未通过内容检查。
4. 作为下一次独立确认方案的候选，建议预先要求 C_remove 的原生 mAP 至少比 C_direct4 高 **1 pp**、零技术回退、全部新排序依据完成审查且没有已确认的无依据胜负。该建议是开发筛选规则，不代表本十视频能证明稳定收益，也不是已经获批的扩大条件。E_remove 的收益单列，不能替代主对比过关。
5. 未过门槛不自动改提示、改记忆或追加数据。即使过关，也只形成后续独立确认的建议；不自动启动 SVHighlights、QV ROI、A/B/C、100 条、R3 或其他数据集。

## 6. 已完成的离线准备与未完成事项

已完成：19 项 CPU 测试；400 份新输入字段差分；全部 400 份旧比较的来源校验；80 个合成 CPU 输出契约例；448 条事实到 50 个旧事实调用的追踪。**合成契约例不是模型调用、模型预测或实际解码可达性测试。**

本地准备目录：`$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_offline_v1_2026-10-27/`，包括 `audit.json`、`fact_lineage.json`、`budget_books.json`、`prepared_requests.jsonl`、`preparation_receipt.json`。所有准备输入明确标为 `authorized=false`，没有新预测文件。

| 固定凭证 | SHA-256 |
|---|---|
| 父轮 manifest | `b2be2e23a429a8c0f2663f344f035e8f768242d2939a2259808b6fef30d38317` |
| 父轮 predictions | `cadfa1941bc79ab62566352eddf311c76e31b2edbddc44313e8ad865c2610065` |
| 400 份准备输入 | `9c9309d8bb00208cebaad39dba8e06a551970a6c08a2df53bfa0b5c1f4bc91fb` |

未完成且未授权：新模型推理、GPU 分配、新轮运行器/监督器的正式执行验收和新指标。原[总体方案](revalue_current_experiment_plan_2026-10-26.md)、[transfer 协议](revalue_tvsum_transfer_protocol_2026-10-27.md)及[旧结果](revalue_tvsum_transfer_results_2026-10-27.md)保持原样。本协议与报告仅保存、同步到本地发布目录，不推送。

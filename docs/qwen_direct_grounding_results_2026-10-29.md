# QV20 Direct Grounding：定位结果与反例分析

> 发布说明（2026-10-30）：本报告随组会 PPT 和总结同步。保留原实验事实、历史状态与预算；机器路径改用别名，未包含的运行材料明确标为本地资料。文中原“不推送”描述属于编写时记录，本次推送依据后续用户授权。

日期：2026-10-29。状态：**20/20 推理完成、CPU 评价与独立审计通过，worker 已退出。**

本轮响应“做一下 qv 上 grounding 相关的 direct 测试，主要目的是找出反例”。原始授权、冻结规则见[协议 v1](qwen_direct_grounding_protocol_v1_2026-10-29.md)。全部预测及复核产物见结果包（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_qv20_v1_2026-10-29/README.md`）。

## 1. 实际做了什么

固定原有 QV20，Qwen3-VL-8B-Instruct BF16 每条调用一次：输入 query、整视频原 64 帧及真实 PTS，直接输出一个连续匹配区间 `[start,end]`，也允许 `not_found` 或 `uncertain`。模型、像素、采样、native video 输入均未改变；没有候选重排、记忆或回看。

**这是新的时间定位任务。** 旧实验要求生成密集评分数组或单目标分数，本轮要求直接给起止秒数；二者任务/输出均不同，不能把新 R1 与旧高光 mAP 当同指标比较，也不是严格单变量的时间理解因果试验。

这 20 条来自已有 train 开发诊断集，已用于先前 Direct 诊断；不等于完整 QV validation，也不是未见测试集。所有请求在 GT 内容读取前已冻结；评价前再次冻结预测与退出收据，随后才解析标签。

## 2. 完整 20 条结果

| 项目 | 结果 |
|---|---:|
| 技术成功 / 总样本 | 20 / 20 |
| 合法 found / not_found / uncertain | 20 / 0 / 0 |
| R1@IoU=0.5 | **70.00%（14/20）** |
| R1@IoU=0.7 | **50.00%（10/20）** |
| 全20 mean best-GT IoU | **0.6804** |
| 主要反例：IoU<0.5 | **6 / 20** |
| 仅更严格阈值失败：0.5≤IoU<0.7 | **4 / 20** |
| 完全无交集 | **0 / 20** |

R1 逐条对本地官方 `compute_mr_r1` 交叉验证：预测对任一 GT 区间取最大 IoU；多次出现只需匹配其中一次。mIoU 是本轮自定义诊断均值。协议规定弃答、技术失败和未尝试均保留在全20分母；本轮实际没有这些遗漏。

**已经找到反例，但集中在区间覆盖和边界上。** 6 条主要反例为 3 条覆盖偏窄、2 条部分重叠、1 条过宽。全部20条都有正重叠，这不能证明每条都正确理解了动作，但也不支持把这批失败概括成“完全没找对事件”。例如 qid1058 的“卧室里两名女性坐着交谈”预测与GT均为0–34秒。

## 3. 六个主要反例：全部保留

时间单位为秒；下表展示用于官方最大 IoU 的匹配 GT。完整多区间 GT 见第6节。

| qid | Query | 预测 | 匹配GT | IoU | 几何错误形态 |
|---|---|---|---|---:|---|
| 9429 | Rescuers helping people who were affected | 22–38 | 0–36 | 0.3684 | 覆盖偏窄 |
| 5370 | Boy in gray top is playing with action figure. | 19–25 | 16–22 | 0.3333 | 部分重叠／边界偏移 |
| 10272 | Brunette woman wears glasses during a interview. | 0–150 | 0–56 | 0.3733 | 范围过宽 |
| 8025 | There are statues of various Simpsons characters. | 100–110 | 90–106 | 0.3000 | 部分重叠／边界偏移 |
| 7795 | Man holds up his phone on the bus. | 63–70 | 60–76 | 0.4375 | 覆盖偏窄 |
| 7971 | Woman is walking down the street in a teal top. | 120–150 | 80–144 | 0.3429 | 覆盖偏窄 |

“偏窄/过宽/部分重叠”按冻结协议中的覆盖率和精度规则分类，只描述区间几何关系。没有按效果排除失败或替换样本，没有为了制造胜负重新调用模型。

## 4. 三个预定展示案例的画面核查

遵循运行前规则：按错误类别顺序，各取原 QV20 顺序首例，最多4个。本轮实际得到3个案例：10272、9429、5370。每张图均显示本轮输入的全部64帧，红框为预测覆盖，绿框为GT覆盖；没有新解码或给模型补帧。

### 4.1 短动作边界偏移：qid5370

Query：男孩穿灰色上衣玩玩偶。预测 **19–25秒**，匹配 GT **16–22秒**，IoU **0.3333**。两个区间长度都为6秒，但平移3秒便只剩3秒交集。

旧输入17.57、19.90秒可见男孩与玩偶，24.60秒已切到婴幼儿用餐画面。预测漏掉较早的匹配动作，并延伸到场景切换之后。这是适合后续边界细化验证的支持性案例。22.23秒仍可见男孩及手持物，稀疏帧不能据此确定逐帧真边界；其他4个GT实例不是本轮必须穷尽的召回目标。

qid5370的64帧与时间区间（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_qv20_v1_2026-10-29/counterexample_gallery/qv_5370_all64.png`）

### 4.2 覆盖偏窄：qid9429

Query：救援人员帮助受影响的人。预测 **22–38秒**，匹配 GT **0–36秒**，IoU **0.3684**；GT覆盖14/36，预测区间内匹配14/16。

输入17.55秒已有扶助画面，22.26–33.97秒是近距离救护/搀扶画面，支持“模型只取了较晚的一个局部”这一观察。0–36秒含切镜和不同救助画面，究竟应把哪些画面视为同一事件仍需人工核查；不能只凭此例证明某一种采样策略导致失败。GT的72–90秒为另一实例，不要求本轮全部召回。

qid9429的64帧与时间区间（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_qv20_v1_2026-10-29/counterexample_gallery/qv_9429_all64.png`）

### 4.3 过宽与多段标注口径：qid10272

Query：戴眼镜的深发色女性接受采访。模型输出 **0–150秒整条视频**，GT为0–56、58–60、70–122、134–150秒，最大单段 IoU **0.3733**。模型没有遵守“选一个连续实例、不跨空隙”的任务要求。

大部分画面属于同一访谈，123.04、125.36秒切到男性主持人。另一方面，GT的60–70秒等未标时间在若干输入帧仍能看到被访女性，**不能把所有标注空隙都断言为无关画面**。这里同时存在单实例选择和标注分段口径问题，应列入人工标注审阅，不能当作已证实的女性/眼镜识别错误。

qid10272的64帧与时间区间（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_qv20_v1_2026-10-29/counterexample_gallery/qv_10272_all64.png`）

以上视觉核查由助手完成，尚无独立人工复核；未修改GT、预测或指标。具体观察与不确定性保存在结果包 `assistant_case_review.json`。

## 5. 对下一步的实际启示

1. **先研究局部边界核查。** 以这轮 Direct 区间作为冻结起点，检查起点之前是否已有匹配事件、终点之后是否仍持续，以及是否已切到其他对象/场景。qid5370是明确的候选案例，qid7795和7971也有可量化的覆盖不足。
2. **单独处理连续事件与多实例分段。** qid10272需要先确认标签口径，再验证是否应拆段或选择一个实例；不要用整段访谈都相关的主观判断替换官方单段IoU。
3. **先把反例当诊断材料。** 本轮不能证明 memory、比较器或回看能改善结果，也没有证明稀疏采样、注意力或时间编码就是原因。下一轮应有冻结对照与独立预算；若只在这6个失败上改进，只能称为失败集修复，不能当完整QV20或全benchmark提升。

本轮只记录后续研究建议，未自动发起新的模型条件或回看实验。所有20条均为有GT的正查询，因此没有验证负query拒答能力。

## 6. 全部20条逐样本表

| qid | Query | 预测区间 | 全部GT区间 | best IoU | 判定 |
|---|---|---|---|---:|---|
| 9429 | Rescuers helping people who were affected | 22–38 | 0–36；72–90 | 0.3684 | 主要反例 |
| 5172 | Two young girls sitting on pool chairs | 45–58 | 42–60 | 0.7222 | 通过0.7 |
| 5370 | Boy in gray top is playing with action figure. | 19–25 | 16–22；64–66；74–76；122–126；136–140 | 0.3333 | 主要反例 |
| 10272 | Brunette woman wears glasses during a interview. | 0–150 | 0–56；58–60；70–122；134–150 | 0.3733 | 主要反例 |
| 8963 | Woman talks to the camera in a white nyc sweatshirt. | 28–150 | 30–150 | 0.9836 | 通过0.7 |
| 1058 | Two black women sit and talk in a bedroom. | 0–34 | 0–34 | 1.0000 | 通过0.7 |
| 881 | People stand by the pool in swimsuits before jumping in. | 58–81 | 58–80 | 0.9565 | 通过0.7 |
| 8025 | There are statues of various Simpsons characters. | 100–110 | 90–106；116–124；126–130 | 0.3000 | 主要反例 |
| 9196 | A reporter in a winter beanie, scarf and jacket is reporting near a large embankment of snow. | 0–15 | 2–14；116–132 | 0.8000 | 通过0.7 |
| 5073 | Woman makes herself some cereal. | 52–67 | 48–64 | 0.6316 | 仅0.7未达标 |
| 764 | A woman waits at an airport then walks to her plane. | 138–147 | 136–150 | 0.6429 | 仅0.7未达标 |
| 7795 | Man holds up his phone on the bus. | 63–70 | 60–76 | 0.4375 | 主要反例 |
| 5967 | Dresses laid out on a bed are shown. | 123–139 | 114–138 | 0.6000 | 仅0.7未达标 |
| 6468 | Two blonde women wearing white eat at a restaurant table overlooking the beach. | 0–24 | 0–26 | 0.9231 | 通过0.7 |
| 1910 | A family taking a boat to enjoy the sea view with babies | 23–67 | 30–68 | 0.8222 | 通过0.7 |
| 7050 | A lady in a white T-shirt is driving while talking about a house. | 14–150 | 12–150 | 0.9855 | 通过0.7 |
| 10249 | Man and woman have a interview in a library. | 34–82 | 32–82 | 0.9600 | 通过0.7 |
| 9841 | The queen is wearing a pink satin shirt. | 112–128 | 102–106；110–126 | 0.7778 | 通过0.7 |
| 7971 | Woman is walking down the street in a teal top. | 120–150 | 80–144 | 0.3429 | 主要反例 |
| 3202 | A young woman shows how to wear a Galo tribe conical traditional basket | 108–130 | 98–132 | 0.6471 | 仅0.7未达标 |

机器可读的逐条数据为 `per_query.csv`、`diagnostic.json`；6条主要失败与4条严格阈值失败另存 `counterexamples.json`。

## 7. 运行健康度、验证及独立成本

运行前11项CPU契约测试通过、18项grammar用例通过、6条真实Qwen词表CPU token-mask路径通过；源码审查及完成后的独立结果审计均通过。原64帧像素、PTS、metadata和真实native prompt token哈希逐条一致。GT读取时序和冻结链可审计。

CPU准备先发现xgrammar0.1.27把最大128秒误编译为可接受129秒；在推理前改用精确整数枚举0..duration，未改依赖或任务含义，失败产物保存在 `cpu_preparation_attempt1_failed/`，该阶段模型调用与GPU分配均为0。

| 本轮新账 | 实际 / 上限 |
|---|---:|
| 请求尝试 | 20 / 20 |
| 成功 / 技术失败 / 重试 | 20 / 0 / 0 |
| 输入 / 输出 token | 101,909 / 303 |
| 总 token | 102,212 / 163,840 |
| worker 启动 / 最大并发 GPU | 1 / 1 |
| GPU 生命周期 | 70.206870820 / 900 秒（0.01950 GPUh） |
| 请求内生成耗时之和 | 27.839813915 秒 |

数值上限由助手在本次新任务授权下保守设定，不冒称用户逐项预算。实际使用GPU0，加载、执行及清理共70.21秒；退出码0、所属进程为空、systemd为inactive/dead且Result=success。日志保留NCCL清理与nanobind引用计数提示，没有traceback；不能称为完全无警告，也不构成重跑理由。

旧账全部独立保留：

| 旧任务 | 累计请求 | token | GPU成本 |
|---|---:|---:|---:|
| 旧QV诊断 | 797/1000 | 398,929 | 0.5395204532 GPUh |
| 旧TVSum transfer | 670/670 | 1,106,932 | 2,102.263978秒 |
| 原Direct pilot | 26/30（25成功1OOM） | 140,457（含6,665失败预留） | 280.481364秒 |
| 时间键QV20 | 20/20 | 115,759 | 395.601399秒 |
| 单目标评分诊断 | 40/40 | 204,992 | 150.434879614秒 |

**622份受保护旧文件和10个历史账本哈希核对不变。** 旧QV剩203次、原Direct剩4次与其他剩余token/GPU时间均未转移；旧TVSum请求额度已耗尽，补跑仍单独未授权。新grounding的20次请求额度已用完，剩余token/GPU时间不允许追加调用。本轮没有推送。

## 8. 产物与复查入口

- 实验协议：[qwen_direct_grounding_protocol_v1_2026-10-29.md](qwen_direct_grounding_protocol_v1_2026-10-29.md)。
- 运行原件：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_grounding_qv20_v1_2026-10-29/`。
- 项目内结果包：grounding_qv20_v1_2026-10-29（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_qv20_v1_2026-10-29/README.md`）。
- 实现：experiments/direct_grounding（本地资料：`$MERIT_ROOT/experiments/direct_grounding/README.md`），包括CPU准备、契约、worker、监督器和评价；运行以 `source_snapshot/` 为准。
- 关键审计：`manifest.json`、`preparation_receipt.json`、`cpu_tests.json`、`runner_review.json`、`prediction_freeze.json`、`result_audit.json`、`budget_books_final.json`、`protected_after.json`及worker退出收据。
- 后处理联系图生成器：`experiments/direct_grounding_visualize.py`，不属于推理干预；仅使用已冻结64帧。

当前最直接的研究机会是**把已大致命中的事件定位得更完整、更精确，并明确多实例的分段口径**。这一判断来自固定开发20条及有限助手画面核查，后续仍需要独立条件与更大验证集验证。

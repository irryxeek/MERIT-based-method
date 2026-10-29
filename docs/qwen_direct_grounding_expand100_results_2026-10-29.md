# QV Direct Grounding 扩样：新增100条固定清单的结果与反例

> 发布说明（2026-10-30）：本报告随组会 PPT 和总结同步。保留原实验事实、历史状态与预算；机器路径改用别名，未包含的运行材料明确标为本地资料。文中原“不推送”描述属于编写时记录，本次推送依据后续用户授权。

日期：2026-10-29。**运行已停止：96次尝试，94条合法响应、1条OOM、1条非法区间，4条未尝试。** 这不是100条全部推理完成的结果。

## Material Passport

- Origin：academic-research-suite / experiment-agent，用户“多跑一点样本”的既定任务。
- Verification Status：ANALYZED。预测、成本及CPU指标复算通过；画面核查由助手完成，没有独立人工核验或独立模型重跑。
- 原20条和全部历史失败不改写；本轮不推送，不补跑TVSum。

## 1. 样本与方法

从既有无标签train元数据7218条中，剔除原QV20的查询及所有同源YouTube视频；以冻结SHA256规则排序、每个源视频至多一条，固定新增100条。有效候选池7160条，100个不同源视频，与旧20查询/源视频重叠均为0。选择在媒体可用性检查及GT读取前完成，没有按旧分数或结果挑样本。

这是查询哈希排序后按源视频去重的开发样本，不是源视频均匀随机抽样，也不是未触碰的官方validation。这里只保证与原Grounding QV20不重叠，不保证从未被其他历史实验或预训练使用。

沿用原Direct Grounding：Qwen3-VL-8B-Instruct BF16、整视频均匀64帧和真实PTS、相同像素处理与提示，输出一个连续区间或not_found/uncertain。没有评分数组、候选比较、记忆、回看、音频或模型训练。评价为单区间对任一GT的最大IoU，不计算HD mAP。

原协议：[expand100 v1](qwen_direct_grounding_expand100_protocol_v1_2026-10-29.md)；运行组织修订：[冷启动分批 v1.1](qwen_direct_grounding_expand100_runtime_amendment_v1_1_2026-10-29.md)。

## 2. 固定100条分母的结果

| 项目 | 结果 |
|---|---:|
| 实际尝试 / 固定清单 | 96 / 100 |
| 合法输出：found / not_found / uncertain | 91 / 3 / 0 |
| OOM / 非法区间 / 未尝试 | 1 / 1 / 4 |
| R1@IoU=0.5 | **42.00%（42/100）** |
| R1@IoU=0.7 | **28.00%（28/100）** |
| mean best-GT IoU，完整100分母 | **0.4255** |
| 合法输出但IoU<0.5的指标反例 | **52** |
| 仅0.7未通过，0.5≤IoU<0.7 | **14** |

所有缺失或非法输出计零，分母不缩为94。91条found均逐条核对官方compute_mr_r1；mIoU为自定义诊断。52条主要反例包含49条合法区间和3条not_found，不包括OOM、非法区间及4条未尝试。

旧QV20是70% / 50%，新100是42% / 28%，但样本集合和完成情况不同，不能把差值当方法退化或改进的因果对照，也不混成官方全量成绩。

## 3. 反例类型与解释边界

| 冻结几何/状态分类 | 数量 |
|---|---:|
| 与全部GT无交集 | 14 |
| 范围偏窄 | 11 |
| 部分重叠 | 11 |
| 范围过宽 | 13 |
| 返回不存在 | 3 |

**这些是相对于发布标注的失败，不能全部等同于语义理解错误。** 四个展示例按事前类别规则取首例，而非事后寻找最能支持某一机制的案例。它们揭示了连续视觉事件、标注分段及单区间评价之间的差异。

### qid2781：与全部GT无交集

Query：Guys throw boomerangs at targets inside a gymnasium.。预测 `[114, 150]`；GT `[[56, 60], [72, 98]]`；最大IoU **0.0000**。

GT56–60、72–98秒可见体育馆内人物投掷；预测114–150秒包含蜡烛特写、交谈及后段相似投掷画面。

相对发布GT为完全无交集；预测区间并非可据此断言全无查询相关内容。需要核查目标/动作实例与标注覆盖，不能直接认定模型完全理解错事件。

qid2781全部64帧（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_expand100_v1_2026-10-29/counterexample_gallery/qv_2781_all64.png`）

### qid9319：范围过宽

Query：Official brief press about NOVA SCOTIA HURRICANE。预测 `[20, 150]`；GT `[[20, 80], [82, 142], [144, 150]]`；最大IoU **0.4615**。

约20秒后大部分输入帧是同一女性在发布会上发言；GT80–82及142–144秒空隙附近也呈现相近画面。预测20–150秒跨越三个GT分段。

过宽是相对最大单段IoU的几何结论；仅凭稀疏视觉不能证明标注空隙全是无关事件，需要确认连续发布会与GT分段口径。

qid9319全部64帧（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_expand100_v1_2026-10-29/counterexample_gallery/qv_9319_all64.png`）

### qid2760：范围偏窄

Query：A man in a black shirt is talking in a warehouse.。预测 `[0, 10]`；GT `[[0, 92], [96, 138], [140, 150]]`；最大IoU **0.1087**。

预测0–10秒含开头黑衣男子讲解；GT0–92秒还含大量设备/图示插画，55.06、66.87、83.17秒又可见该男子。

按GT覆盖明显偏窄；视觉连续出镜与旁白/主题连续段的定义可能不同。当前无音频，未核验旁白，因此不能直接声称模型漏看了92秒的连续出镜。

qid2760全部64帧（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_expand100_v1_2026-10-29/counterexample_gallery/qv_2760_all64.png`）

### qid5466：部分重叠

Query：Video editing software showing how the video was made.。预测 `[38, 57]`；GT `[[0, 8], [48, 62]]`；最大IoU **0.3750**。

38.64秒起已经是视频编辑软件界面，直至59.73秒仍可见；GT第二段为48–62秒，预测38–57秒。

有终点覆盖不足，且预测起点前移部分仍符合软件界面描述；低IoU同时涉及GT起点口径，不宜把全部偏移解释为视觉时间定位错误。

qid5466全部64帧（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_expand100_v1_2026-10-29/counterexample_gallery/qv_5466_all64.png`）

## 4. 两次停止原因及运行健康度

1. 原连续worker在第26次qid9063发生视觉编码器CUDA OOM。25条成功和失败记录全部保留。日志符合此前跨请求视觉encoder缓存累积来源链，但本轮没有做新的缓存因果GPU对照。
2. 为完成同一固定清单，只把助手原设的一次worker改为最多5次累计：原1次加20/20/20/14四个顺序冷启动。提示、帧、请求哈希、engine及评价代码均未改，原26次尝试和成本逐项继承，不重复调用。三批各20条成功。
3. 最后一批第10条，即总第96次qid2305，原始输出为 `{"status":"found","window":[150,17]}`。两个数各自都在0–150内，但违反start<end；严格解析器正确拒绝。此为模型输出契约失败，不能由交换端点生成新的预测，也不是OOM。按预定首个新失败即停规则结束。

最后四个未尝试qid：2688、2498、437、6141。所有worker均已退出，无所属残留进程；systemd保留failed状态以反映停止原因，没有伪造inference_complete。累计worker上限已到，不自动启动第六次或补齐四条。

若后续处理区间倒置，应另立输出契约条件或授权重试，不将修订结果覆盖本轮。日志中的nanobind/NCCL清理提示单列保留，不作为重启理由。

## 5. 成本与验证

| 当前新增100范围累计账 | 实际 / 上限 |
|---|---:|
| 尝试 | 96 / 100 |
| 重试 | 0 / 0 |
| 94条合法响应token | 482,150 |
| 非法响应实际token | 5,141 |
| OOM保留的完整预留token | 7,168 |
| 总计费token | 494,459 / 819,200 |
| GPU生命周期，含加载和清理 | 361.064442033 / 900秒 |
| worker累计启动 / 同时GPU | 5 / 1 |

不能把父运行26次再加到续执行账本96次上：后者已包含前者。分批后新调用70次，其中69合法、1非法。旧QV、旧TVSum及其他Direct额度没有转移。

| 独立旧账 | 请求 | token | GPU成本 |
|---|---:|---:|---:|
| 旧QV诊断 | 797/1000 | 398929 | 0.5395204532 GPUh |
| 旧TVSum transfer | 670/670 | 1106932 | 2102.263978秒 |
| 原Direct pilot | 26/30 | 140457，含失败预留 | 280.481364秒 |
| 时间键QV20 | 20/20 | 115759 | 395.601399秒 |
| 单目标评分QV20 | 40/40 | 204992 | 150.434879614秒 |
| 首轮Grounding QV20 | 20/20 | 102212 | 70.206870820秒 |

新增100输入准备100/100通过；原7项CPU契约、25项grammar检查及15条真实词表CPU路径通过。运行修订增加8项CPU检查，验证失败前缀不能删除、旧成本不能重置、仅执行74条剩余目标及累计时间/token限制。最终996份保护文件与11个旧账本哈希不变；CPU独立算式复核指标通过。

## 6. 下一步依据

- 已有52个指标反例、14个严格阈值失败和1个非法输出，足以形成诊断集；当前不应继续仅为扩大数量重复调用。
- 先人工核查GT连续区间与视觉事件口径，特别是体育重复动作、采访/旁白和画面插入。否则把标注分段差异误当边界模型错误，会导致错误优化方向。
- 对确认存在真实边界遗漏的案例，再准备Direct区间前后局部核查对照；对完全选错实例的案例另做事件证据验证。不得把仅在失败子集上的修复率当全量性能。
- 语法上独立限制两个端点不足以保证start<end；后续可准备互相关联的区间输出契约，但需要保留本轮失败作冻结对照。本轮不启动该条件，也不自动补跑剩余样本。

## 7. 产物

- 项目结果包（本地资料：`$MERIT_ROOT/comparisons/qwen_direct/grounding_expand100_v1_2026-10-29/README.md`）：全部100条逐样本CSV、52条主要反例、14条严格阈值失败、原始响应、四张64帧图、账本与CPU审计。
- 原失败运行：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_grounding_expand100_v1_2026-10-29/`。
- 续执行与评价：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_grounding_expand100_v1_1_2026-10-29/`。
- 约2.8GB冻结帧数组留在SSD，以路径与SHA256引用；没有把视频或模型权重复制入项目。

## 附录：全部52条指标反例

按冻结清单顺序；非法区间与未运行项在per_query.csv中单列。GT和预测单位均为秒。

| qid | Query | 预测 | 全部GT | IoU | 类型 |
|---|---|---|---|---:|---|
| 2781 | Guys throw boomerangs at targets inside a gymnasium. | [114, 150] | [[56, 60], [72, 98]] | 0.0000 | 与全部GT无交集 |
| 2760 | A man in a black shirt is talking in a warehouse. | [0, 10] | [[0, 92], [96, 138], [140, 150]] | 0.1087 | 范围偏窄 |
| 5466 | Video editing software showing how the video was made. | [38, 57] | [[0, 8], [48, 62]] | 0.3750 | 部分重叠 |
| 9319 | Official brief press about NOVA SCOTIA HURRICANE | [20, 150] | [[20, 80], [82, 142], [144, 150]] | 0.4615 | 范围过宽 |
| 448 | The chickens are in a coop. | [130, 145] | [[92, 112]] | 0.0000 | 与全部GT无交集 |
| 3790 | Vlogger showing some peaceful ruler area of south India | [0, 25] | [[8, 150]] | 0.1133 | 部分重叠 |
| 3506 | A woman crossing a narrow unstable bridge made out of ropes and natural materials. | [13, 23] | [[10, 44]] | 0.2941 | 范围偏窄 |
| 7435 | A man in green is holding a bright orange snake. | [19, 30] | [[12, 38]] | 0.4231 | 范围偏窄 |
| 10293 | A group of Metro Police in bright yellow coats are apprehending someone. | [65, 130] | [[66, 98]] | 0.4923 | 范围过宽 |
| 5361 | Woman talks while hanging onto ta yellow cloth. | [55, 63] | [[62, 70], [78, 84]] | 0.0667 | 部分重叠 |
| 4324 | People are standing on the stage of a religious ceremony in a temple. | [40, 130] | [[50, 84]] | 0.3778 | 范围过宽 |
| 8894 | A lady is cooking and serving the naan bread with curry dish. | [81, 128] | [[56, 76], [78, 90]] | 0.1800 | 部分重叠 |
| 9491 | Clashes between police and  citizens of Hong Kong erupted during protest | [35, 45] | [[14, 24], [26, 52], [78, 116], [118, 120], [126, 134], [142, 144], [146, 148]] | 0.3846 | 范围偏窄 |
| 4165 | Man sits next to a large circle light. | [60, 135] | [[62, 90], [98, 100]] | 0.3733 | 范围过宽 |
| 4008 | Cute little girl sharing her breakfast menu | [38, 43] | [[18, 44]] | 0.1923 | 范围偏窄 |
| 6548 | A woman is having a coffee drink from a fancy machine. | [119, 130] | [[82, 102]] | 0.0000 | 与全部GT无交集 |
| 6264 | All angles shown of a van interior. | [0, 43] | [[12, 84], [98, 100], [108, 110], [120, 122], [148, 150]] | 0.3690 | 部分重叠 |
| 4296 | Man in blue shirt holds a camera in his right hand. | [0, 150] | [[0, 50], [60, 72], [82, 150]] | 0.4533 | 范围过宽 |
| 3676 | A woman wearing a santa hat is setting up a small Christmas tree. | [0, 24] | [[118, 144]] | 0.0000 | 与全部GT无交集 |
| 10032 | People wearing red uniforms are performing in the street while standing side by side in a row. | [117, 147] | [[70, 106], [108, 126], [144, 150]] | 0.2308 | 部分重叠 |
| 22 | Man is talking to his friend wearing a y'all sweatshirt. | [137, 147] | [[64, 82]] | 0.0000 | 与全部GT无交集 |
| 4348 | A woman is introducing a market and is browsing a jewelry store there. | [127, 145] | [[58, 80]] | 0.0000 | 与全部GT无交集 |
| 4024 | A man holding a coffee cup during a convention | [0, 150] | [[0, 62]] | 0.4133 | 范围过宽 |
| 1869 | There are different stores in a market in Budapest. | [110, 148] | [[84, 86], [110, 126], [130, 150]] | 0.4500 | 范围过宽 |
| 4605 | A little Asian girl taking out a towel from a bag and putting it on top of a chair | [] | [[72, 108]] | 0.0000 | 返回不存在 |
| 5951 | Man in Hawaiian shirt walks on the beach. | [135, 145] | [[74, 76], [120, 128], [132, 140], [144, 148]] | 0.3846 | 部分重叠 |
| 2603 | Girls making breakfast and having fun | [137, 150] | [[74, 118]] | 0.0000 | 与全部GT无交集 |
| 6285 | A three women is showing products inside of a beige big box | [140, 150] | [[50, 150]] | 0.1000 | 范围偏窄 |
| 6833 | A man in white t-shirt is talking in front of the camera | [137, 142] | [[82, 88], [90, 108], [114, 138]] | 0.0357 | 部分重叠 |
| 9350 | A newscast shows night time flood scenes in Hyderabad | [139, 150] | [[88, 140]] | 0.0161 | 部分重叠 |
| 4703 | A man talks up close to the camera with his mask on. | [112, 128] | [[72, 100]] | 0.0000 | 与全部GT无交集 |
| 7038 | Man holds his baby to his chest. | [53, 72] | [[52, 54], [64, 82]] | 0.2759 | 部分重叠 |
| 8975 | Man and woman walk along the beach at dusk. | [129, 150] | [[90, 130]] | 0.0167 | 部分重叠 |
| 2677 | A woman in a white tank top talks about chocolate slices. | [0, 150] | [[54, 104]] | 0.3333 | 范围过宽 |
| 5267 | Girl is wearing a plastic poncho. | [10, 16] | [[8, 46]] | 0.1579 | 范围偏窄 |
| 658 | Man and woman hold different camera equipment against a blue background. | [0, 15] | [[2, 4], [24, 30], [42, 48], [58, 64], [88, 94], [112, 116], [126, 134], [138, 146]] | 0.1333 | 范围过宽 |
| 2650 | A crowd sings and cheers as they watch a sporting event on television. | [10, 15] | [[10, 48]] | 0.1316 | 范围偏窄 |
| 4725 | A woman walking with her friends outside St. Peter's Basilica | [129, 145] | [[8, 12], [80, 114]] | 0.0000 | 与全部GT无交集 |
| 5538 | Man in white hat walks around outside. | [] | [[90, 98], [102, 116]] | 0.0000 | 返回不存在 |
| 3631 | A man gesturing to his female friend who he then sits next to against a purple wall. | [43, 57] | [[120, 150]] | 0.0000 | 与全部GT无交集 |
| 9105 | People are running on the beach. | [75, 81] | [[60, 64], [68, 72], [76, 88]] | 0.3846 | 范围偏窄 |
| 9160 | Cows are standing in the middle of the road. | [19, 25] | [[54, 68]] | 0.0000 | 与全部GT无交集 |
| 1835 | Man and woman in a pink top car dance with a American flag in the background. | [109, 147] | [[72, 102]] | 0.0000 | 与全部GT无交集 |
| 6506 | A couple is visiting a historical stone palace with ornate arches and steps on vacation. | [51, 130] | [[48, 80]] | 0.3537 | 范围过宽 |
| 5206 | Woman eats cheese on bread. | [73, 127] | [[74, 88]] | 0.2593 | 范围过宽 |
| 4249 | A woman is syncing a device to her phone and then putting it in a case. | [108, 124] | [[88, 124]] | 0.4444 | 范围偏窄 |
| 4601 | A luxury hotel room is covered with natural design elements such as branches and leaves. | [126, 133] | [[122, 150]] | 0.2500 | 范围偏窄 |
| 8649 | A rider narrates his journey as he rides his bike along the road on a beautiful sunny day. | [0, 150] | [[132, 150]] | 0.1200 | 范围过宽 |
| 5491 | Woman in white top stands around a pool. | [62, 67] | [[22, 46]] | 0.0000 | 与全部GT无交集 |
| 4699 | A group of girls are playing in a fountain together. | [] | [[100, 124]] | 0.0000 | 返回不存在 |
| 4660 | Two men walk down the street side by side. | [128, 147] | [[86, 108]] | 0.0000 | 与全部GT无交集 |
| 8105 | Woman in striped shift is riding a inner tube. | [0, 150] | [[2, 4], [8, 12], [20, 26], [28, 38], [40, 44], [46, 50], [54, 60]] | 0.0667 | 范围过宽 |

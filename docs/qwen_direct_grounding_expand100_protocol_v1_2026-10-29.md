# Direct Grounding 新增100条诊断协议 v1

> 发布说明（2026-10-30）：本报告随组会 PPT 和总结同步。保留原实验事实、历史状态与预算；机器路径改用别名，未包含的运行材料明确标为本地资料。文中原“不推送”描述属于编写时记录，本次推送依据后续用户授权。

## Material Passport

- Origin Skill：academic-research-suite 3.23.2 / experiment-agent；Mode：run。
- Origin Date：2026-10-29；Version：direct-grounding-expand100-v1。
- 用户授权：本会话最新指令“多跑一点样本”，承接已完成QV20定位反例实验。
- Verification Status：UNVERIFIED（冻结时尚未推理；CPU契约和实际运行收据另存）。

## 1. 目的与固定方法

新增100条query继续发现Direct时序grounding反例。100是助手在“多跑一点”授权下采用的适度扩量，不冒称用户明确指定100，也不沿用旧实验余额。

保持上一轮Grounding的Qwen3-VL-8B-Instruct BF16、确定性采样、任务文本、JSON契约、整视频64帧取样/缩放、真实PTS、8192 context和2048 output cap。一次请求返回一个最有依据的连续事件区间，或not_found/uncertain；不输出评分、不加memory、比较器或回看。旧20不重跑。

## 2. 不读取GT的样本冻结

使用现有无标签清单 `$VIDEOARM_ROOT/reproduction/qvhighlights/manifests/train.jsonl`，7218行；只允许qid、query、vid、duration、media_path、timeline_id、num_clips、split等元数据字段。数据来源是train开发集，不是完整validation或未触碰test。

排序键为UTF-8字符串 `qwen-direct-grounding-expand100-v1-20261029|qid|vid` 的SHA256，哈希相同则按qid。先排除旧20的qid及全部YouTube源视频，再按该顺序每个源视频最多取一条，取前100。源视频ID由vid末尾两个数字时间字段移除得到。选择不使用query语义、GT边界、旧模型分数、失败类别或媒体可用性。

`selection_rule.json`、`selected_inputs.json` 已在媒体存在性检查前写入；此后没有替换资格，读帧失败不从后续队列补选。新100内部源视频唯一，与旧20源视频完全去重。该规则是内容盲的确定性query-hash抽取再去重，不能称为均匀随机抽取源视频。

## 3. CPU输入准备与审查

从每条视频全时段64个等宽格的中点取源帧，以实际PTS定位，使用原bicubic/antialias缩放到512×320外框且尺寸为32倍数。有效采样时长取原文件结束与标称时长的较小值，保持旧规则。新视频当然产生新像素，但不改变采样机制。

逐条保存视频/源像素/缩放像素哈希、原帧索引、PTS、metadata、native video grid和CPU prompt token哈希。CPU通过真实AutoProcessor和已有时间token展开器双重校验。模型worker再次核对同一实际token哈希后才预留请求。

JSON仍为 `{"status":"found","window":[start,end]}` 或明确弃答；整数秒域使用上一轮修复后的精确枚举，严格验证0≤start<end≤duration，不修补模型输出。仅新增不同duration的CPU grammar检查，不重复已完成的旧模型实验。

## 4. 独立预算与停止规则

助手采用的保守上限：**100次总尝试、0重试、0额外故障请求、819200 token；1次worker、最多1张空闲卡、GPU生命周期900秒**，包括加载、推理及清理。新额度归属本轮扩量，所有旧账不变。

一次冷启动连续处理冻结100条；只选当前空闲RTX4090，无空闲即停止、不排队。首个OOM、格式错误、哈希/上下文漂移或预算不满足即结束，没有自动重启、重复调用、替换样本或新增条件。正常uncertain/not_found不是技术错误，继续剩余清单。每次generate前预留input+2048，异常无法可靠结算时保留预留；监督器预留20秒清理且只清理所属进程组。

原20完成不保证连续100条一定稳定，故完整保留故障与未尝试分母。模型效果差或同分等科学结果不触发改提示。本轮不补TVSum、不下载/修改依赖、不推送、不干预其他用户进程。

## 5. 评价、反例和展示

运行退出并冻结全部预测/账本后才解析GT。主结果单独报告**新增100条完整分母**：R1@IoU0.5、R1@IoU0.7和自定义全100 mean best-GT IoU。每个found输出对本地官方compute_mr_r1核对；对任一GT取最大IoU，弃答/技术遗漏/未尝试计0，后三者与有效区间错误分开。旧20只作历史参照，不将两批差值解释成方法改进或退步。

IoU<0.5的有效模型输出全部列为主要失败，0.5≤IoU<0.7单列。无交集、过宽、过窄和部分重叠按旧协议几何规则分型；该标签仅对失败解释，不证明具体视觉原因。正常弃答不是错误时间区间的直接证据。

展示沿用原规则：依无交集/过宽/过窄/部分重叠/not_found/uncertain顺序，每类取冻结清单的首例，最多4个。完整失败表不删减；展示图只用本轮输入的64帧，不新增模型或采样。助手核查与独立人工复核分开，GT划分疑点保留、不改标注。不按GT挑有利推理子集，不为凑反例强迫非空。

## 6. 原账与交付

旧QV797/1000、398929 token、0.5395204532 GPUh；旧TVSum670/670、1106932 token、2102.263978 GPU秒；Direct pilot26/30、140457计费token、280.481364 GPU秒；时间键QV20 20/20、115759 token、395.601399 GPU秒；单目标评分40/40、204992 token、150.434879614 GPU秒；上一轮Grounding20/20、102212 token、70.206870820 GPU秒。旧余额不转移，余量不授权新条件。

运行目录：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_grounding_expand100_v1_2026-10-29/`。保留选择清单、采样帧、源码快照、CPU检查、原始响应、请求/生命周期账本、全部逐样本评价和报告。帧数组只留SSD，项目内归档小型预测/报告/源码/展示图和哈希引用。完成后给出新增100的结果、反例类型、代表案例和实际成本；后续改进实验另定。

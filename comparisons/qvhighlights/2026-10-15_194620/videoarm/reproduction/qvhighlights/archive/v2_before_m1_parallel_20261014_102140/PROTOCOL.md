# 开发协议 v2_stop_contract（正式 val 前尚需冻结）

唯一开发修订：停止动作只需 action=stop/start=0/end=0，question 可省略；视觉工具仍需 question。修订仅依据 train 开发格式诊断，模型、名单、预算与输出规则不变。v1 原件、预测和失败保留，详情见 DEVELOPMENT_AMENDMENT_V2.md。

主 MR、次 HD，分开 run_id/数据库/评分。模型、开发列表、初始像素和输出契约固定；任何正式成绩前，执行 `freeze_qvh.py` 保存 code/config/environment/media/model/prompt/evaluator 身份和正式 run_id 注册表。

- 初始化：MR 全片至多 24 帧的 query overview；HD 完整 `int(duration/2)` clip 向量，每批至多 24 张、每 clip 一个真实代表帧。所有方法共享精确身份的初始化。
- 每张图片单独输入，相邻文字说明图片位置和实际 PTS，HD 另标 ID 与 `[2i,2(i+1))`。像素最长边 672，JPEG quality 90，均匀请求在实际可见区间取中点并映射到最近有效 PTS；最后部分 clip 只取该 clip 真实帧。
- B1 五个固定时间范围，HD 向下/上对齐 clip 网格，最多 24 clips。M1/A1 两工具共用十次/120 局部帧，Controller 自选区间并可提前停。
- 每请求至多 24 个逻辑帧；Controller 至多 14；全部请求至多 32（含初始化、终答、重试）；MR 呈现上限 512，HD 640。单请求最多一次原请求重试；解析错误不修改原始输出。
- context 32768；Controller/最终输出 4096、Scene 1024、Clip 2048；temperature 0、seed 42。明确 seed 不保证 BF16/服务逐位确定性。
- 每 query/method 输入 token 262144、输出 65536、实际活动时间 1800 秒。每次请求之前 fail-closed 预留 `context-max_output` 输入与最大输出，计入图片/请求；保留终答一请求、24 图片、4096 输出和上下文余量。未知使用量保留整个预约，不通过恢复清零。
- controller 因预算无法继续时停止搜索并尝试保留的终答额度；合法结果标记旁路 budget interruption，不能说是自主高置信提前停。
- 固定摘要仅影响模型上下文，完整记录旁路保存：result 事件前 256 字符、不确定性前 128 字符、working 目的前 128 字符；HD 初始各 batch 事件前 256 字符但保留完整 clip 分数。Sensory 摘要只给区间与 PTS，文件路径/SHA 不反复塞进 prompt。Controller 图片只复用已有初始/最新工具像素，并在每个请求重新编号位置。MR 另外保存候选区间内是否有直接采样 PTS 的诊断；该诊断不判定语义正确，也不修改预测。
- SQLite WAL + synchronous FULL：预算预约与请求日志同事务；响应和使用量同事务；状态检查点与预测 JSONL 原子写入。中断 running 请求被标 uncertain 并保留完整预约，最多原阶段剩余一次尝试。succeeded/failed/budget_exhausted 终态不选择性重跑。
- 语义键包括 task、qid、query hash、vid、media/PTS SHA、固定模型 revision、完整实际消息和像素、完整采样/协议、seed/temperature/max_tokens。恢复用原请求文字与有哈希的原图片；代码/配置/manifest 改变则拒绝恢复。A1 没有历史缓存回查入口。
- MR 按 confidence 降序，只去重复 start/end（保留最高分），最多 10 个。空候选原样保存为无证据失败，官方 R1 不支持空列表，不伪造区间。
- HD 必须精确 ID、顺序、长度、有限值；未细查留初始化；合法局部最新分数替换，不合法不覆盖。完整初始化失败则样本失败，不输出伪全零。
- 推理读严格 whitelist manifest，score_qvh 独立读取 GT，test 推理前不打开 test 标签。Val 正式前冻结，不根据 val/test 调参或挑最优重复运行。
- 官方 evaluator 原件只读；完整 qid/query/vid 检查后评分。MR 报告 R1@0.5/0.7、mAP@0.5/0.75/平均及短中长组；HD Fair=2、Good=3、VeryGood=4 的 mAP/Hit1，VeryGood 为主表，三位标注者聚合完全使用官方函数。
- 全计划失败按指标记零的汇总明示自定义，保留分母。另报 B1/M1 共同合法子集的官方分数与覆盖率。逐 query 指标取官方未舍入函数，核对聚合后做 seed42、10,000 次源视频聚类配对 bootstrap。

真实能力门槛：24-picture 颜色/位置绑定、固定真实视频 Scene→Clip→final 接口探针（不计入效果）、两任务自主 smoke8、dev50 四方法与 B1/M1 低/高点、实际 token/frame/latency 记录。全量媒体审计完成后冻结，按同一协议运行 val 的 B0/B1/M1，再运行 test 的 B1/M1。任何失败保留，只有完全合法覆盖才叫完整集合官方成绩。

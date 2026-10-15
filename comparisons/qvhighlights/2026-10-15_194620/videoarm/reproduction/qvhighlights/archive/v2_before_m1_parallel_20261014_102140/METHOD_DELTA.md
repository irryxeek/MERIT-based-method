# 方法差异

2026-10-14 修订 v2_stop_contract：v1 错误地将工具 question 设为 stop 的必填字段；267 次保存的控制器 schema 失败全为三字段 stop，正常完成且可解析。新协议给 stop 条件格式，原模型/名单/预算不变，不改写 v1。旧 M1/A1 低覆盖受接口错误影响，不能据此判定原方法效果。详情见 DEVELOPMENT_AMENDMENT_V2.md。

本实现是 VideoARM 在 QVHighlights 上的开源模型迁移评测，training-free、video-only。原论文没有该数据集成绩可逐项恢复。

| 项目 | 论文/固定公开源码 | 此次迁移 |
|---|---|---|
| 任务与模型 | 长视频 QA；公开默认 Controller=o3、视觉工具=gpt-4.1 | 查询条件 MR 和独立 HD；Controller、Scene、Clip 同一固定 Qwen3-VL-8B BF16 |
| HM³ | 论文 §3.2：sensory 的长/短感知池、按时间排列 result、working 中推理与目标；公开 `agent.py` 以 scene/audio/clip 三个列表注入上下文 | sensory 保存不可变初始图像映射与最新观察范围；result 保存动作、帧与结构化工具证据；working 保存简短行动目的；不生成内部推理链，不宣称完整恢复论文所有机制 |
| Controller 图像 | 公开源码在重建 HM³ 后重新附最新图像消息，图片通常是 3×2 拼图 | 初始 24 张独立图片；后续至多 12 张初始图像和 12 张最新局部图像，逐张相邻位置/PTS 文本，不复用失败的 SVH 图像绑定提示 |
| 工具 | 公开 Scene/Clip/audio 与帧编号接口；论文另描述 Interval Localizer/Clip Explorer | Scene/Clip 的 start/end 秒范围整合原 scoping 功能，程序按 PTS 解码；禁用 audio/ASR，Clip 检查事件存在、持续、重复、边界 |
| 自适应性 | observe→think→act→memorize | Controller 自选工具/区间/停止；输出 schema-bounded JSON，不需要服务原生 tool parser；仍是任务适配后的自主控制器 |
| MR | 原自然语言最终 QA | 多个不连续区间、最多 10 个、有排序置信分数；只去完全重复边界、保留最高分，不做 NMS、合并或网格化 |
| HD | 原论文没有 QVH 的全片排序协议 | 完整 2 秒 clip 初始化、每批最多 24 个真实代表帧；合法局部结果按动作顺序替换对应分数，不做窗口归一化或补零 |
| 上下文管理 | 论文 working/result 清理与外化；公开三个列表的 JSON 注入 | 固定摘要：每条 result 事件前 256 字符、不确定性前 128 字符、working 目的前 128 字符，完整候选/clip 分数；HD 各初始 batch 事件前 256 字符。模型内只传必要 PTS；文件路径和 SHA 留旁路。最多 10 动作，全文和原图映射都在 trace 与 SQLite。没有基于 GT 的回查或搜索量 |
| No-History | 本次消融 | 不可变初始化 + 最新合法工具结果；最新工具图像；只有去语义化 start/end 覆盖与必要预算。旧语义、旧行动目的不交给 Controller 或 MR 最终聚合；HD 累积输出状态不交回模型 |

方法：B0 MR 仅 overview 后终答、HD 仅完整初始化；B1 加五个均匀时间安排的 Clip；M1 自选 Scene/Clip，最多十动作、120 局部帧；A1 同 M1 但移除跨轮历史语义，仅用于开发。

MR/HD 初始化请求按精确语义身份共享，使各方法获得同一初始化证据。共享仅限 overview/完整 HD 初始化，不共享控制器或局部语义。共享命中也记录等效冷启动请求、token 和图像成本；另计实际新增模型请求/token/图像。重复实际展示图片再次计入呈现数，独立帧按 media SHA + PTS 去重。

Controller 后续图片复用初始及最新工具已经观察的像素，不额外采出新的局部帧绕过工具帧上限。HD Controller 的初始图像取完整初始化 clip 代表帧的均匀子集，复用同一 PTS；每个请求的图片位置重新编号为连续 0..N−1，避免初始/局部图片出现重号。

低/高成本诊断只针对 dev50 的 B1/M1。默认配置作为高点（120 局部帧、每动作 24 帧）；低点 60 局部帧、每动作 12 帧，B1 仍五次均匀观察但窗口宽度为默认 0.6，HD 对齐真实 clip 范围。低/高既改变采样也改变可观察范围，按独立协议报告，不能把收益全部归于自适应或记忆。

工程与接口测试不构成方法有效性证据；forced 工具调用只验证执行接口，不能计入自主方法得分。完整成效比较需要原计划覆盖率、总成本、B1/M1 配对差值与源视频聚类区间。

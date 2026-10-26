# REZE 下一阶段执行记录：QFVS 收尾、QV 候选对照与长视频基线

核对时间：2026-10-23 01:10:18（北京时间）。执行依据为[下一阶段方案 v2](reze_next_stage_execution_plan_v2_2026-10-23.md)。本文记录本轮启动与冻结状态，最终指标以各运行完成后的评价报告为准。

## 1. 已执行与当前状态

| 阶段 | 设备 | 当前状态 | 范围 |
|---|---|---|---|
| Q1：QFVS 概念提示 × 时间选择 | GPU 4 | `running`；正式评分 9,245/20,356 | 原四查询、四条完整视频；完成后自动生成四条件及 Uniform 比较 |
| L0：SVHighlights 原方法恢复 | GPU 5 | `running`；正式评分 480/19,014 | 固定七场、15.84 小时视频；已通过 21 个 CPU 与 21 个 GPU 探针 |
| V1：QV 候选分散对照 | 接续 GPU 4 | `waiting_for_Q1`；正式 0/100 查询 | 旧 B0/C0 与全部输入帧已核验；Q1 终止后检查空闲设备，先工程检查再正式推理 |

本轮未占用 GPU 6、7。原 GPU 6 QFVS 计时器已按用户要求停用；旧 SVHighlights 失败队列也已停用。新队列不终止其他用户进程。

## 2. 调度迁移与预算保留

QFVS 新目录是 `$ZEROSHOT_ARTIFACT_ROOT/reze_qfvs_concept_pilot4_gpu4_v2/`。只改变 GPU 绑定，提示、清单、帧、模型、选择器和评价协议保持原冻结设置。原 `reze_qfvs_concept_pilot4_v1/` 留存；其一次加载失败及 **22.33 秒**活动时间计入新目录，仍只有 **两次剩余 worker 尝试**。没有清空请求账本以恢复预算。

Q1 正式请求 20,356，工程请求 12，累计请求最多 20,416；活动 worker 总上限 12 小时。迁移后重新通过全部 12 个媒体探针和 GPU 工程调用；正式窗口继续逐批核对原 B0 像素。

SVHighlights 新目录是 `$ZEROSHOT_ARTIFACT_ROOT/reze_svhighlights_pilot7_gpu5_v2/`。保留旧 v1 的三次失败，并从新任务活动时长预算中扣除已消耗的 **31.93 秒**。旧任务未执行有效模型请求。新任务使用数值 GPU 索引并核对实际 UUID，增加首中尾 21 次 GPU 工程检查。**19,014 正式＋21 工程＋中断补试共用 19,062 次总请求上限**，不另增工程预算。

## 3. V1 已实现的对照

新实现目录：`$ZEROSHOT_ROOT/experiments/reze_agent_v2/`；发布副本：`baselines/zeroshot/experiments/reze_agent_v2/`。运行目录：`$ZEROSHOT_ARTIFACT_ROOT/reze_agent_v2_v1_explore100/`。

- C0 复用原分数 Top-5 和原 P_pair 完整结果；重新验证父运行收据、原始概率、分数置换和身份。
- C1 按原分数贪心取候选，抑制 clip ID 距离 ≤2 的邻居；不足五个时按原排序补满。半径固定，不按结果搜索。
- 两组使用原比较提示、0.6/0.4 阈值及 20 次有向比较；每候选最多四张 B0 旧帧。
- 新协议支持任意合法五候选，保留原分数集合及非候选分数；旧 Top-5 校验器和冻结快照不变。
- 100 条查询的 C0/C1 帧、实际时间戳和像素已重新解码核验，全部通过；尚未产生 C1 正式指标。

工程阶段固定五条查询，共 100 次比较，只查格式和证据一致性；正式阶段 100 条、2,000 次比较。工程最多 100 次，正式尝试最多 2,100 次，累计活动 GPU 时间最多两小时；中断预留继续收费。冻结 B0 总 token 为 **2,896,024**，新增调用的保守 token 上限为 **14,480,120**，端到端 token 不超过 B0 的六倍。

完成后自动计算官方 VeryGood mAP、Hit@1、Fair/Good、逐查询差值、候选覆盖及源视频配对 bootstrap（10,000 次，seed42）。`decision.json` 按未四舍五入数值判定：C1 相对 C0 的 mAP ≥+1 pp、Hit@1 ≥−1 pp、回退不增加且成本合格才选 C1；否则 V2 使用 C0。

## 4. S0 审计与后续边界

已重新核对 explore100 与 confirm100 的源视频交集为零；在两个活跃产物根目录找到的 **51 份预测 JSONL** 中，未发现 confirm100 的完整 query/video 身份匹配。此检查没有读取准确率作选型；不能覆盖未知归档或人工查看历史，进入 V3 前仍需最终审计。

QFVS 的四份 H5 与原下载哈希一致，分别含 2,783 / 3,692 / 2,152 / 3,588 个 2,816 维特征；均为有限值，shot 数与当前网格一致。平铺和分组数组的差异小于 3e-8，符合 float64/float32 精度差异。已有说明将特征标为 SlowFast/CLIP，但缺少精确编码器权重来源、提取时间戳及 shot 聚合记录。**Q2 语义去重分支暂未放行**，不把文件存在视为零样本资格已核实，也不自动下载新模型。

V2 的候选事实卡、规划器、均匀/定向观察器仍待实现和工程验证，**没有加入当前自动队列**。当前自动链到 V1 评价与候选选型为止；随后按该选型落实四组 Agent 消融。V3、Q3、QV 全量重跑、QFVS 180 查询及 MR.HiSum 全量重跑均未启动。

## 5. 后台执行与验证

启用的用户级计时器：

- `reze-qfvs-concept-pilot4-gpu4-v2.timer`
- `reze-svh-pilot7-gpu5-v2.timer`
- `reze-agent-v2-v1-explore100.timer`

约每分钟检查一次；各自有进程锁、原子检查点、有界尝试和 900 秒无进度超时。已核对 `Linger=yes`，SSH 断开后继续运行；重启恢复要求 SSD、数据路径及分配设备仍可用。请求或活动时间触及上限会停止，不无限重启。V1 只在 Q1 已生成完整比较或明确停止记录后接续。

验证结果：V1 六项针对性测试通过，覆盖同分/候选不足、任意候选、旧排序等价、非候选不变、标签隔离和中断预算；SVHighlights 十项测试通过，覆盖时间网格、设备选择、旧像素及 GPU smoke 的收费/复用。V1 的 GPU 工程检查尚待接续执行。

SVHighlights 当前冻结快照的官方评价器经历了 import 排序与无插值 f-string 简化；十个函数经规范化 AST 比对均等价，指标逻辑未变。维护源码已恢复官方原文件，运行快照原样保留，差异及哈希见 `evaluator_equivalence_audit.json`。

## 6. 查阅入口与发布

本机运行指针：`docs/reze_qfvs_adaptation_current_run.local.json`、`docs/reze_svhighlights_current_run.local.json`、`docs/reze_agent_v2_current_run.local.json`。这些机器路径文件保留本地。

运行目录内保留 `binding.json`、清单、模型/代码/媒体哈希、`status.json`、请求及 worker 账本；QFVS 最终写 `comparison.json/.md`，SVHighlights 写 `evaluation.json` / `REPORT.md`，QV V1 写 `results/evaluation.json`、`results/REPORT.md` 与 `decision.json`。

本报告、索引和维护源码已同步到本地 Git 发布副本；本轮未提交或推送。正式新指标未完成时不覆盖此前结果总表。

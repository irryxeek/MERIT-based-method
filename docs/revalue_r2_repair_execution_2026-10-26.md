# ReVALUE R2 修复重跑执行记录

日期：2026-10-26。依据[修复协议 v3](revalue_r2_repair_protocol_v3_2026-10-26.md)；问题与旧轮结果见[诊断报告](revalue_r2_diagnosis_2026-10-26.md)。

## 已完成的验证

- 70 项 CPU 回归通过，ReVALUE 目录 Ruff 检查通过。
- 真实解析检查为旧 7 条失败加最早空规划样本，共 8 条；修复后 8/8 通过。
- 首轮真实检查完成 53 次请求，但新观察写回后的比较发生一次非法记忆引用，整轮工程验收失败并保留。
- 修复本环境字符串枚举编译问题后，第二轮 53/53 请求通过。模型仍返回空动作，控制器按已声明的规则查证候选 42、38；两组各两动作、每动作 24 张新增源帧，实际观察后各完成 20 次比较，无回退。
- 两观察组分别产生 5、7 条非 unknown 的可见事实。它们是结构与流程检查，不代替人工语义审核；本次工程检查没有计算高光精度。
- 两轮工程检查累计 106 次请求、233,110 token；按用户服务启动至退出日志计费共 403.77 GPU 秒（约 6.73 分钟），包含失败与退出过程，低于 1 GPU 小时修复检查上限。

## 正式重跑设置

同一 explore100、原五候选和旧输入帧，五个条件及 B0 参考。原 v2 的 100 条结果保留；新运行将独立评价，不拼接旧失败样本修复前后的预测。

仅复用 558 次 query-independent 事实/记忆和第二轮检查中的 8 次 TaskSpec。其余任务解析、全部比较和新观察按新协议运行。冷启动最多 9,200 次，新增正式请求最多 8,634 次，另有 460 次有界瞬时重试；各方法 token ≤ B0 的 6 倍，新运行占卡上限 6 GPU 小时。

正式推理已于 **2026-10-26 04:06:39（北京时间）** 启动，首批使用 GPU **5、6、7**，三个 worker 按 query 并行。全 100 条源视频/父输入与 566 次缓存请求均已核验；50 个源码/文档文件冻结。启动时完成数为 0/100，尚无本轮精度结果；实时状态以运行目录 status.json 为准。

运行绑定 SHA-256：`b730f089cade44be3d64c7e8166ed3867f014ad9bda2396659145a0df91e2cb1`。启动和工程成本收据见 `prelaunch_audit.json`、`automation_receipt.json`；旧运行、两轮检查及本次正式推理各自保留账本。

## 本地产物与服务

以下是机器上的产物来源，发布仓库不附带模型、视频或完整缓存。

- 正式运行：`$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_v3_2026-10-26`。
- 历史运行：`$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_v2_2026-10-25`。
- 失败检查：`$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_repair_smoke_v3_2026-10-26`。
- 通过检查：`$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_repair_smoke_v3b_2026-10-26`。
- 诊断和工程累计成本：`$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_repair_v3_2026-10-26/{diagnosis,engineering_usage}.json`。
- 正式完成后指标：运行目录的 `R2/results/REPORT.md`、`evaluation.json`、各条件预测与逐 query 分析。

`$ZEROSHOT_ARTIFACT_ROOT` 为本地运行产物根目录。正式队列使用 `revalue-r2-v3.service` 和 `revalue-r2-v3.timer`，可使用任意空闲 GPU；按 query 独占领取、共享预算、断点恢复，不终止其他任务。完成后自动评估并停止，不自动进入 R3/R4。

本次修改与报告同步到本地发布工作区；Git 推送需按用户的推送请求执行，此次不自动推送。

# REZE 在 QFVS 上的推理执行记录

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](report_publication_workflow.md)。

日期：2026-10-20。固定 **4 查询 pilot 已完成**：每条视频选 1 个查询，覆盖完整 P01–P04，总视频时长 16.96 小时。全量 180 查询约需 916,020 次窗口评分，本轮控制为 **20,356 次正式评分**，另做 12 次 GPU 工程检查，未扩展样本范围。

**完成于 18:49 CST**：4/4 查询完成，Oracle 语义 F1 **38.63%**，三位用户平均 F1 **36.26%**。worker 用时约 35.05 分钟，无 worker 重启。后续 CPU 诊断发现，同预算均匀/随机选片 F1 为 **50.15% / 50.62%**，当前结果未超过简单对照，详见[结果分析](reze_qfvs_analysis_2026-10-20.md)。未启动全量 180 查询。

历史启动快照：18:14 CST 启动，12 次真实 GPU 检查通过；18:16:49 CST 完成 1,088 / 20,356 个窗口，当时估计剩余 35–50 分钟。该快照保留作执行记录，不代表当前仍在运行。

## 固定范围

每个视频取 `sha256("42:vid:query_name")` 最小的查询，不使用摘要位置、语义标签或历史指标筛选。

| 视频 | 查询 | 3 秒评分窗口数 | 5 秒输出片段数 | 摘要片段数 |
|---|---|---:|---:|---:|
| P01 | Desk_Face | 4,637 | 2,783 | 55 |
| P02 | Building_Boat | 6,153 | 3,692 | 73 |
| P03 | Book_Lady | 3,586 | 2,152 | 43 |
| P04 | Garden_Hall | 5,980 | 3,588 | 71 |
| 合计 | 4 个查询 | 20,356 | 12,215 | 242 |

## 方法与必要适配

模型仍为 `Qwen3-VL-8B-Instruct`，固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、无训练。原 QV B0 的 REZE 概率计算、模型后端及读帧参考模块逐字节复用：3 秒窗口、每完整窗口 6 帧、严格动作核验提示、top-20 Yes/No 概率及原逐秒分数曲线。

QFVS 查询改写为 `The video shows either desk or face.` 这样的固定句式。使用“任一概念”是因为 QFVS 将至少含一个查询概念的片段视为相关；仅展开三个复合概念名称，不根据标注改写查询。[QFVS 任务定义](https://arxiv.org/html/1707.04960)

将原 REZE 的逐秒曲线在每个原生 5 秒片段内求均值。P01–P03 最后片段只对有效整秒取均值；P04 额外 0.2 秒尾部不新增标注项。原视频和标注保持原样，边界详情见[下载校验记录](qfvs_raw_videos_2026-10-20.md)。

按固定预算选取 `floor(片段数 × 0.02)` 个高分片段，采用 [UniVTG 的默认 2% 预算](https://github.com/showlab/UniVTG/blob/32659ac7aeba21742a63274f30eba785fc57e247/main/config.py#L177)。同分时先选时间较早的片段，规则在推理前固定。未加入记忆、重排、事件图、回看或额外多样性模块。

这是 **REZE 向 QFVS 查询摘要任务的迁移适配**，不能称为 REZE 论文已报告的 QFVS 原始成绩。

## 评估与检查

主指标为对 Oracle 摘要的语义 Precision、Recall、F1；另报告对三位人工摘要的平均指标。使用固定版本的[语义 IoU 与最大权二分匹配实现](https://github.com/showlab/UniVTG/blob/32659ac7aeba21742a63274f30eba785fc57e247/eval/qfvs.py)，只补充完全零语义交集时返回零的保护，避免原函数除零。两种参考摘要的结果分开报告，不与 QV 的 HD-mAP 混用。

- 19 项针对性测试通过：时间边界、原生片段数量、摘要预算、同分规则、语义指标、检查点隔离、GPU 编号、与 QV 读帧像素一致性。
- 新增停滞监控后，2 项 GPU 调度回归检查通过；Ruff 静态与格式检查通过。
- 四条视频首、中、尾共 12 个窗口通过原生模型输入检查，最大输入为 570 tokens，低于 4,096 上限。
- 模型和视频重新计算文件指纹；运行器、配置、查询和评估输入均已冻结。
- GPU worker 先执行 12 次真实模型检查，通过后自动处理全部正式窗口；评估标签仅在独立 CPU 评估阶段解析。

## 自动执行与产物

实现：`$ZEROSHOT_ROOT/experiments/reze_qfvs/`。

运行目录：`$ZEROSHOT_ARTIFACT_ROOT/reze_qfvs_pilot4_v1/`。

本机入口：运行指针（本地资料：`reze_qfvs_current_run.local.json`）。

用户级 `reze-qfvs-pilot4.timer` 已启用；退出登录后继续，重启后可恢复。调度仅选择 GPU 4–7 中没有其他计算任务的卡，最多使用 1 张。每 16 个窗口保存检查点；中断批次仍计入请求预算，已保存批次校验后复用。最多 3 次 worker 尝试、12 小时累计 worker 时间，15 分钟无进度时终止本任务并按剩余预算重试。

此次修正了现有 SVHighlights 启动器暴露的 vLLM 0.12 设备参数兼容问题：传入数字 CUDA 索引，并在模型加载前核对实际 UUID。修正位于本次 QFVS 独立实现；旧 SVHighlights 的失败记录保持原样。

```bash
systemctl --user status reze-qfvs-pilot4.service
cat $ZEROSHOT_ARTIFACT_ROOT/reze_qfvs_pilot4_v1/status.json
```

运行中可查看 `worker_attempt_*.log`、`gpu_smoke.json`、`chunks/`、`request_attempts/` 与 `predictions/`。完成后自动生成 `evaluation.json`、`predictions.json`、`REPORT.md` 和 `complete.json`。4 查询结果只用于迁移诊断，不能代表完整 180 查询成绩。

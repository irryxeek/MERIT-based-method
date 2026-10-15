# VideoARM 在 QVHighlights 上的开源模型迁移评测

产物核对时间：2026-10-14 18:50:06 +0800。依据文件和独立 SQLite 状态库生成。

已实现独立 MR/HD 推理、标签隔离、PTS 采样、记忆摘要、B0/B1/M1/A1、预算、恢复、严格预测校验、固定官方评分和源视频聚类 bootstrap。

2026-10-14 开发协议修订一次为 v2_stop_contract：v1 将 stop 缺少 question 误判为失败，旧分数受接口错误影响，不能用来判定 VideoARM 算法收益。旧源码、预算、预测、失败和评分保留；后续正式结果使用 v2，同一训练开发名单、模型和预算不变。详见 DEVELOPMENT_AMENDMENT_V2.md。

离线验证：`{"status": "passed", "count": 46, "revision": "v2_stop_contract_m1_parallel_v1", "pytest_log": "reproduction/qvhighlights/logs/pytest_m1_parallel_v1_final.log", "ruff_log": "reproduction/qvhighlights/logs/ruff_m1_parallel_v1_final.log", "scope": "CPU contract/scoring/recovery checks plus overlapping query ledger/cache isolation, pause/resume budgets, pixel identity and parallel GPU time; not model quality", "verified_unix": 1791945407.525629}`。真实自主运行数据库 28；已有评分文件 22；执行状态 `paused_by_user_or_service_stop`。

当前用户范围：仅执行 M1 的 MR/HD smoke8、dev50、完整 val/test；其他方法与低预算实验暂缓，已有结果保留为历史。正式计划 6,184 次样本×方法推理。执行并发与像素缓存见 M1_SCOPE_AND_EFFICIENCY.md；论文原始数据和协议对照见 PAPER_COMPARISON.md。论文没有 QVHighlights 成绩，Accuracy 与本任务 AP/IoU 不可相减。

真实推理和完整评测仅按下表及 progress.json 认定。作者示例和 CPU 合成测试不属于本方法成绩。空表表示暂无已评分真实运行。

| run_id | 提交状态 | 成功/计划 query | 官方指标 |
|---|---|---:|---|
| dev_low_v1_mr_dev50_B1 | official_success_subset_only | 41/50 | {"MR-full-R1@0.5": 21.95, "MR-full-R1@0.7": 4.88, "MR-full-mAP": 7.0, "MR-full-mAP@0.5": 21.1, "MR-full-mAP@0.75": 3.35, "MR-long-mAP": 6.92, "MR-middle-mAP": 10.71, "MR-short-mAP": 0.73} |
| dev_low_v1_mr_dev50_M1 | official_success_subset_only | 12/50 | {"MR-full-R1@0.5": 25.0, "MR-full-R1@0.7": 0.0, "MR-full-mAP": 4.47, "MR-full-mAP@0.5": 11.39, "MR-full-mAP@0.75": 4.17, "MR-long-mAP": 11.25, "MR-middle-mAP": 1.88, "MR-short-mAP": 1.0} |
| dev_v1_hd_smoke8_A1 | official_success_subset_only | 2/8 | {"HL-min-Fair-mAP": 32.86, "HL-min-Fair-Hit1": 50.0, "HL-min-Good-mAP": 22.37, "HL-min-Good-Hit1": 50.0, "HL-min-VeryGood-mAP": 9.13, "HL-min-VeryGood-Hit1": 50.0} |
| dev_v1_hd_smoke8_B0 | complete_official_submission | 8/8 | {"HL-min-Fair-mAP": 54.09, "HL-min-Fair-Hit1": 62.5, "HL-min-Good-mAP": 45.68, "HL-min-Good-Hit1": 62.5, "HL-min-VeryGood-mAP": 33.64, "HL-min-VeryGood-Hit1": 62.5} |
| dev_v1_hd_smoke8_B1 | complete_official_submission | 8/8 | {"HL-min-Fair-mAP": 61.42, "HL-min-Fair-Hit1": 50.0, "HL-min-Good-mAP": 52.71, "HL-min-Good-Hit1": 50.0, "HL-min-VeryGood-mAP": 36.19, "HL-min-VeryGood-Hit1": 50.0} |
| dev_v1_hd_smoke8_M1 | official_success_subset_only | 2/8 | {"HL-min-Fair-mAP": 24.98, "HL-min-Fair-Hit1": 50.0, "HL-min-Good-mAP": 18.62, "HL-min-Good-Hit1": 50.0, "HL-min-VeryGood-mAP": 4.97, "HL-min-VeryGood-Hit1": 0.0} |
| dev_v1_mr_dev50_A1 | official_success_subset_only | 12/50 | {"MR-full-R1@0.5": 25.0, "MR-full-R1@0.7": 0.0, "MR-full-mAP": 5.5, "MR-full-mAP@0.5": 18.33, "MR-full-mAP@0.75": 4.17, "MR-long-mAP": 11.25, "MR-middle-mAP": 1.88, "MR-short-mAP": 1.88} |
| dev_v1_mr_dev50_B0 | official_success_subset_only | 34/50 | {"MR-full-R1@0.5": 35.29, "MR-full-R1@0.7": 17.65, "MR-full-mAP": 17.55, "MR-full-mAP@0.5": 32.6, "MR-full-mAP@0.75": 17.4, "MR-long-mAP": 40.56, "MR-middle-mAP": 12.72, "MR-short-mAP": 2.27} |
| dev_v1_mr_dev50_B1 | official_success_subset_only | 40/50 | {"MR-full-R1@0.5": 50.0, "MR-full-R1@0.7": 27.5, "MR-full-mAP": 24.1, "MR-full-mAP@0.5": 42.73, "MR-full-mAP@0.75": 21.4, "MR-long-mAP": 39.38, "MR-middle-mAP": 25.56, "MR-short-mAP": 0.4} |
| dev_v1_mr_dev50_M1 | official_success_subset_only | 13/50 | {"MR-full-R1@0.5": 15.38, "MR-full-R1@0.7": 7.69, "MR-full-mAP": 7.62, "MR-full-mAP@0.5": 9.23, "MR-full-mAP@0.75": 8.46, "MR-long-mAP": 22.5, "MR-middle-mAP": 5.62, "MR-short-mAP": 0.0} |
| dev_v1_mr_smoke8_A1 | official_success_subset_only | 2/8 | {"MR-full-mAP": 7.5, "MR-full-mAP@0.5": 75.0, "MR-full-mAP@0.75": 0.0, "MR-full-R1@0.5": 100.0, "MR-full-R1@0.7": 0.0, "MR-short-mAP": null, "MR-middle-mAP": 7.5, "MR-long-mAP": null} |
| dev_v1_mr_smoke8_B0 | official_success_subset_only | 6/8 | {"MR-full-R1@0.5": 33.33, "MR-full-R1@0.7": 33.33, "MR-full-mAP": 17.92, "MR-full-mAP@0.5": 29.17, "MR-full-mAP@0.75": 20.83, "MR-long-mAP": 70.0, "MR-middle-mAP": 18.0, "MR-short-mAP": 0.0} |
| dev_v1_mr_smoke8_B1 | official_success_subset_only | 6/8 | {"MR-full-R1@0.5": 66.67, "MR-full-R1@0.7": 33.33, "MR-full-mAP": 22.08, "MR-full-mAP@0.5": 58.33, "MR-full-mAP@0.75": 20.83, "MR-long-mAP": 90.0, "MR-middle-mAP": 18.5, "MR-short-mAP": 0.83} |
| dev_v1_mr_smoke8_M1 | official_success_subset_only | 2/8 | {"MR-full-mAP": 0.0, "MR-full-mAP@0.5": 0.0, "MR-full-mAP@0.75": 0.0, "MR-full-R1@0.5": 0.0, "MR-full-R1@0.7": 0.0, "MR-short-mAP": null, "MR-middle-mAP": 0.0, "MR-long-mAP": null} |
| dev_v2_hd_dev50_M1 | official_success_subset_only | 45/50 | {"HL-min-Fair-mAP": 57.54, "HL-min-Fair-Hit1": 60.0, "HL-min-Good-mAP": 47.84, "HL-min-Good-Hit1": 55.56, "HL-min-VeryGood-mAP": 29.42, "HL-min-VeryGood-Hit1": 44.44} |
| dev_v2_hd_smoke8_M1 | complete_official_submission | 8/8 | {"HL-min-Fair-mAP": 54.08, "HL-min-Fair-Hit1": 62.5, "HL-min-Good-mAP": 48.24, "HL-min-Good-Hit1": 62.5, "HL-min-VeryGood-mAP": 33.97, "HL-min-VeryGood-Hit1": 62.5} |
| dev_v2_mr_dev50_M1 | official_success_subset_only | 37/50 | {"MR-full-R1@0.5": 32.43, "MR-full-R1@0.7": 18.92, "MR-full-mAP": 18.72, "MR-full-mAP@0.5": 30.86, "MR-full-mAP@0.75": 18.69, "MR-long-mAP": 34.17, "MR-middle-mAP": 14.35, "MR-short-mAP": 1.5} |
| dev_v2_mr_smoke8_A1 | official_success_subset_only | 6/8 | {"MR-full-R1@0.5": 50.0, "MR-full-R1@0.7": 16.67, "MR-full-mAP": 11.25, "MR-full-mAP@0.5": 37.5, "MR-full-mAP@0.75": 4.17, "MR-long-mAP": 0.0, "MR-middle-mAP": 24.0, "MR-short-mAP": 0.0} |
| dev_v2_mr_smoke8_B0 | official_success_subset_only | 6/8 | {"MR-full-R1@0.5": 33.33, "MR-full-R1@0.7": 33.33, "MR-full-mAP": 17.92, "MR-full-mAP@0.5": 29.17, "MR-full-mAP@0.75": 20.83, "MR-long-mAP": 70.0, "MR-middle-mAP": 18.0, "MR-short-mAP": 0.0} |
| dev_v2_mr_smoke8_B1 | official_success_subset_only | 6/8 | {"MR-full-R1@0.5": 66.67, "MR-full-R1@0.7": 33.33, "MR-full-mAP": 22.08, "MR-full-mAP@0.5": 58.33, "MR-full-mAP@0.75": 20.83, "MR-long-mAP": 90.0, "MR-middle-mAP": 18.5, "MR-short-mAP": 0.83} |
| dev_v2_mr_smoke8_M1 | official_success_subset_only | 6/8 | {"MR-full-R1@0.5": 33.33, "MR-full-R1@0.7": 16.67, "MR-full-mAP": 7.08, "MR-full-mAP@0.5": 20.83, "MR-full-mAP@0.75": 4.17, "MR-long-mAP": 0.0, "MR-middle-mAP": 19.0, "MR-short-mAP": 0.0} |
| formal_c0c8ae9799ad_mr_val_M1 | official_success_subset_only | 1113/1550 | {"MR-full-R1@0.5": 42.32, "MR-full-R1@0.7": 25.61, "MR-full-mAP": 20.7, "MR-full-mAP@0.5": 37.76, "MR-full-mAP@0.75": 19.58, "MR-long-mAP": 31.9, "MR-middle-mAP": 16.85, "MR-short-mAP": 2.48} |

## 当前 M1 的覆盖率与全计划口径

上表保留历史版本和其他方法结果。下表仅列当前 v2 或正式 M1；失败贡献零的聚合是自定义保守指标，不是完整集合官方成绩。

| run_id | 合法预测/原计划 | 全计划失败记零指标（自定义） |
|---|---:|---|
| dev_v2_hd_dev50_M1 | 45/50 | {"HL-min-Fair-mAP": 51.78561918934883, "HL-min-Fair-Hit1": 54.0, "HL-min-Good-mAP": 43.052287231844005, "HL-min-Good-Hit1": 50.0, "HL-min-VeryGood-mAP": 26.47571415078685, "HL-min-VeryGood-Hit1": 40.0} |
| dev_v2_hd_smoke8_M1 | 8/8 | {"HL-min-Fair-mAP": 54.07909426353158, "HL-min-Fair-Hit1": 62.5, "HL-min-Good-mAP": 48.235000913597645, "HL-min-Good-Hit1": 62.5, "HL-min-VeryGood-mAP": 33.966186605269336, "HL-min-VeryGood-Hit1": 62.5} |
| dev_v2_mr_dev50_M1 | 37/50 | {"MR-full-mAP": 13.85, "MR-full-mAP@0.5": 22.83333333333333, "MR-full-mAP@0.75": 13.833333333333336, "MR-full-R1@0.5": 24.0, "MR-full-R1@0.7": 14.0} |
| dev_v2_mr_smoke8_M1 | 6/8 | {"MR-full-mAP": 5.3125, "MR-full-mAP@0.5": 15.625, "MR-full-mAP@0.75": 3.125, "MR-full-R1@0.5": 25.0, "MR-full-R1@0.7": 12.5} |
| formal_c0c8ae9799ad_mr_val_M1 | 1113/1550 | {"MR-full-mAP": 14.863844046634368, "MR-full-mAP@0.5": 27.116784302913317, "MR-full-mAP@0.75": 14.059015190305516, "MR-full-R1@0.5": 30.387096774193548, "MR-full-R1@0.7": 18.387096774193548} |

## 数据与时间轴

| split | query | 数据集片段 | 源 YouTube 视频 | 有媒体 query |
|---|---:|---:|---:|---:|
| train | 7218 | 7100 | 2214 | 7218 |
| val | 1550 | 1519 | 474 | 1550 |
| test | 1542 | 1529 | 476 | 1542 |

dev50 完整 PTS 与首尾解码通过 50/50；全量检查 10148/10148，失败 0。

初始 0.25 秒一致性检查隔离 222 个片段；标注减媒体时长差异 −0.411078 至 1.742875 秒，原始检查保留。登记的 identity-PTS 映射要求每个官方 clip 有真实帧，不改变标注、像素或 clip 数量。部分末尾片段的可见时长不足 2 秒；单帧不能证明整段动作。预测与解码使用当前数据集片段起点，不叠加文件名中的源偏移。

## 评测、成本与方法限制

固定官方评测器 commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`。作者 1,550-query 示例结果与参考文件完全一致，这只验证评测器安装。逐 query 使用同一官方 AP/IoU/三位标注者函数，并核对聚合。开发子集缺少某个时长组时，该组为未定义，使用非空组的官方函数，未修改官方文件。

完整合法覆盖才标记完整官方提交；部分结果另外报告 B1/M1 共同成功子集和原计划分母的失败指标记零自定义汇总。不会伪造预测使评测运行。常量或全零 HD 向量仍有官方排序分数，不能直接当作零指标。配对 bootstrap 保留原始精度，seed=42、10,000 次，按源 YouTube 视频聚类。

此次 sensory/result/working-objective 表达与论文 HM³ 及公开代码的三个工具列表存在差异。working 只保留短行动目的；时间范围工具接口、独立 MR/HD 输出、HD 全片初始化与固定摘要为任务适配，不称为论文所有机制或其 QVHighlights 分数的完整恢复。详见 METHOD_DELTA.md。

A1 只传不可变初始化、最新工具语义和去语义化覆盖记账。HD 累积输出分数保存在程序状态中，不交回 Controller；MR 最终聚合也只看最新工具记录。历史像素和全文保留在旁路 trace，A1 无回查接口。

当前 M1 保留默认 120 局部帧预算。原低/高预算 B1/M1 实验的已完成结果保留，尚未执行的低预算条件因用户范围变更暂缓。

真实 smoke/dev 成本测量后才生成 checks/full_run_estimate.json，提供完整 val/test 的时间、存储和 GPU 小时估计；该文件缺失时，没有实测估计。冷启动请求/token/图像和共享初始化后的实际新增成本分别记账。GPU 服务监控包含启动和空闲时间。

## 剩余问题与恢复

执行阻塞：`{"status": "paused_by_user_or_service_stop", "phase": "P3"}`。实时资源见 checks/gpu_readiness.json；仅空闲且无计算进程的两张 GPU 可用于默认 BF16 TP=2。

独立 videoarm-qvh-worker.service 只推进 M1：真实 binding/forced 接口检查、MR/HD smoke8/dev50、实测估时、冻结、val/test、评分和报告。跨 query 并行执行，每 query 内动作顺序不变。接口错误保存具体原因；预算和正式 run_id 不重置。videoarm-qvh-media-audit.service 已独立完成 CPU 审计。

现有 SVHighlights 推理继续暂停；其源码、配置、timer、ledger 和预测保持冻结。所有环境、运行、评分、恢复和暂停命令见 README.md。尚未完成的工作由 progress.json 和各 run/state.sqlite 标识；工程完成不能代替真实推理验收。

## B1/M1 配对统计与实际成本

下表为 M1−B1，单位为百分点；共同成功集与自定义全计划失败记零口径分开。原始完整指标、三阈值 HD 与覆盖率均在 checks/paired_*.json。暂无真实结果时，下表为空。

| 比较文件 | 口径 | 指标 | 差值 | 95% CI | 有效源视频簇 |
|---|---|---|---:|---|---:|
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | common_success_ci | HL-min-Fair-mAP | -7.791358121546797 | [-10.498575498575496, -5.084140744518098] | 2 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | common_success_ci | HL-min-Fair-Hit1 | 0.0 | [0.0, 0.0] | 2 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | common_success_ci | HL-min-Good-mAP | -3.2430763812561825 | [-4.515669515669519, -1.9704832468428468] | 2 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | common_success_ci | HL-min-Good-Hit1 | 0.0 | [0.0, 0.0] | 2 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | common_success_ci | HL-min-VeryGood-mAP | -3.731496586657519 | [-5.982905982905982, -1.4800871904090553] | 2 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | common_success_ci | HL-min-VeryGood-Hit1 | -50.0 | [-100.0, 0.0] | 2 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | custom_full_plan_failure_zero_ci | HL-min-Fair-mAP | -55.18068864241265 | [-76.04170532028601, -33.06002642199013] | 8 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | custom_full_plan_failure_zero_ci | HL-min-Fair-Hit1 | -37.5 | [-75.0, -12.5] | 8 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | custom_full_plan_failure_zero_ci | HL-min-Good-mAP | -48.05275478070185 | [-67.23082291952106, -27.095782224644854] | 8 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | custom_full_plan_failure_zero_ci | HL-min-Good-Hit1 | -37.5 | [-75.0, -12.5] | 8 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | custom_full_plan_failure_zero_ci | HL-min-VeryGood-mAP | -34.944696327863326 | [-50.16207399626518, -19.882220980748453] | 8 |
| paired_hd_smoke8_dev_v1_hd_smoke8_M1.json | custom_full_plan_failure_zero_ci | HL-min-VeryGood-Hit1 | -50.0 | [-87.5, -12.5] | 8 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | common_success_ci | MR-full-mAP | -3.425 | [-16.06729166666667, 10.241666666666665] | 10 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | common_success_ci | MR-full-mAP@0.5 | -11.333333333333334 | [-43.0, 15.666666666666666] | 10 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | common_success_ci | MR-full-mAP@0.75 | 1.2500000000000007 | [-9.999999999999998, 14.583333333333334] | 10 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | common_success_ci | MR-full-R1@0.5 | 0.0 | [-40.0, 40.0] | 10 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | common_success_ci | MR-full-R1@0.7 | -10.0 | [-30.0, 0.0] | 10 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP | -4.668333333333333 | [-9.3435, -0.5232500000000012] | 50 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.5 | -14.566666666666665 | [-24.867499999999996, -4.733333333333333] | 50 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.75 | -1.75 | [-6.75, 2.0833333333333335] | 50 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.5 | -12.0 | [-24.0, 0.0] | 50 |
| paired_mr_dev50_dev_low_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.7 | -4.0 | [-10.0, 0.0] | 50 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | common_success_ci | MR-full-mAP | -4.625 | [-27.416666666666668, 15.875] | 12 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | common_success_ci | MR-full-mAP@0.5 | -20.416666666666668 | [-51.666666666666664, 7.5] | 12 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | common_success_ci | MR-full-mAP@0.75 | -0.8333333333333334 | [-25.0, 23.333333333333332] | 12 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | common_success_ci | MR-full-R1@0.5 | -25.0 | [-58.333333333333336, 8.333333333333334] | 12 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | common_success_ci | MR-full-R1@0.7 | -8.333333333333334 | [-33.333333333333336, 16.666666666666668] | 12 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP | -17.298333333333336 | [-27.538333333333334, -7.77825] | 50 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.5 | -31.78333333333333 | [-45.133750000000006, -18.883333333333336] | 50 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.75 | -14.916666666666668 | [-26.6, -4.216666666666667] | 50 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.5 | -36.0 | [-50.0, -22.0] | 50 |
| paired_mr_dev50_dev_v1_mr_dev50_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.7 | -20.0 | [-32.0, -8.0] | 50 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | common_success_ci | MR-full-mAP | -11.25 | [-12.5, -10.0] | 2 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | common_success_ci | MR-full-mAP@0.5 | -100.0 | [-100.0, -100.0] | 2 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | common_success_ci | MR-full-mAP@0.75 | 0.0 | [0.0, 0.0] | 2 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | common_success_ci | MR-full-R1@0.5 | -100.0 | [-100.0, -100.0] | 2 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | common_success_ci | MR-full-R1@0.7 | 0.0 | [0.0, 0.0] | 2 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP | -16.5625 | [-39.0625, -2.5] | 8 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.5 | -43.75 | [-75.0, -12.5] | 8 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.75 | -15.625 | [-40.625, 0.0] | 8 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.5 | -50.0 | [-87.5, -12.5] | 8 |
| paired_mr_smoke8_dev_v1_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.7 | -25.0 | [-62.5, 0.0] | 8 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | common_success_ci | MR-full-mAP | -15.0 | [-47.5, 6.666666666666667] | 6 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | common_success_ci | MR-full-mAP@0.5 | -37.5 | [-79.16666666666667, 4.166666666666667] | 6 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | common_success_ci | MR-full-mAP@0.75 | -16.666666666666668 | [-50.0, 0.0] | 6 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | common_success_ci | MR-full-R1@0.5 | -33.333333333333336 | [-66.66666666666667, 0.0] | 6 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | common_success_ci | MR-full-R1@0.7 | -16.666666666666668 | [-50.0, 0.0] | 6 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP | -11.25 | [-35.625, 4.375] | 8 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.5 | -28.125 | [-62.5, 3.125] | 8 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-mAP@0.75 | -12.5 | [-37.5, 0.0] | 8 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.5 | -25.0 | [-62.5, 0.0] | 8 |
| paired_mr_smoke8_dev_v2_mr_smoke8_M1.json | custom_full_plan_failure_zero_ci | MR-full-R1@0.7 | -12.5 | [-37.5, 0.0] | 8 |

| run_id | 冷启动等效/新增请求 | 输入/输出 token | 新增输入/输出 token | 呈现/独立帧 | P50/P95 秒 | 活动 GPU 小时 |
|---|---:|---:|---:|---:|---|---:|
| dev_low_v1_mr_dev50_B1 | 359/310 | 1175457/26904 | 955875/18976 | 5616/4140 | [22.17684010748053, 32.00870546206424] | 0.6549699262134769 |
| dev_low_v1_mr_dev50_M1 | 397/348 | 1767397/25127 | 1547815/17199 | 7968/1716 | [23.342776103003416, 49.56170228464761] | 0.6155394504712766 |
| dev_v1_hd_smoke8_A1 | 98/66 | 399455/13022 | 271567/4341 | 1458/600 | [18.880745563976234, 68.28999148770527] | 0.13470898774387832 |
| dev_v1_hd_smoke8_B0 | 32/32 | 127888/8681 | 127888/8681 | 600/600 | [44.42702958700829, 47.44993895004445] | 0.1986799599916493 |
| dev_v1_hd_smoke8_B1 | 72/40 | 260086/17852 | 132198/9171 | 1200/600 | [48.95094047446037, 52.142556462463105] | 0.21740625953722176 |
| dev_v1_hd_smoke8_M1 | 98/66 | 416097/12608 | 288209/3927 | 1458/600 | [18.918859665020136, 65.54781536808002] | 0.12766509747119725 |
| dev_v1_mr_dev50_A1 | 396/347 | 2009917/25674 | 1790335/17746 | 9504/2086 | [17.211855409521377, 51.536564654740495] | 0.6804462895738672 |
| dev_v1_mr_dev50_B0 | 116/108 | 550856/24017 | 514978/22848 | 2784/1200 | [12.090762847976293, 35.331237240077435] | 0.5228558616461103 |
| dev_v1_mr_dev50_B1 | 358/309 | 1708685/35707 | 1489103/27779 | 8592/6089 | [32.96262670098804, 57.94468113231702] | 1.036621871911468 |
| dev_v1_mr_dev50_M1 | 391/342 | 2025547/29095 | 1805965/21167 | 9384/2175 | [16.298604702024022, 53.058163125251276] | 0.7542143957760223 |
| dev_v1_mr_smoke8_A1 | 64/56 | 324596/3632 | 288718/2463 | 1536/311 | [21.418874581548152, 47.111841514933616] | 0.1004627970349975 |
| dev_v1_mr_smoke8_B0 | 18/17 | 85430/2383 | 80970/2209 | 432/192 | [11.433797915000468, 27.68039187702815] | 0.06010508254500261 |
| dev_v1_mr_smoke8_B1 | 59/51 | 282734/4564 | 246856/3395 | 1416/990 | [31.260063103516586, 46.243116082021146] | 0.1520609092261293 |
| dev_v1_mr_smoke8_M1 | 64/56 | 332140/3474 | 296262/2305 | 1536/311 | [22.075943293020828, 45.49663294153869] | 0.10077074511005776 |
| dev_v2_hd_dev50_M1 | 429/230 | 1905500/72202 | 1107744/16117 | 7236/3771 | [22.074141987483017, 117.70334143619691] | 0.23820157031499548 |
| dev_v2_hd_smoke8_M1 | 60/28 | 258376/10395 | 130488/1714 | 1022/600 | [25.46616570846527, 34.944781821506325] | 0.03718522316274983 |
| dev_v2_mr_dev50_M1 | 278/229 | 1423272/22422 | 1203690/14494 | 6672/1799 | [29.45936105248984, 95.53490743923757] | 0.28260474624833376 |
| dev_v2_mr_smoke8_A1 | 49/41 | 248519/2720 | 212641/1551 | 1176/311 | [14.502651615999639, 35.37003538239223] | 0.07698468095939866 |
| dev_v2_mr_smoke8_B0 | 18/10 | 85430/2381 | 49552/1212 | 432/192 | [5.153572112030815, 21.362116046997823] | 0.03709130300778068 |
| dev_v2_mr_smoke8_B1 | 59/51 | 282734/4564 | 246856/3395 | 1416/990 | [31.276409502519527, 46.26271821504051] | 0.15233305271999498 |
| dev_v2_mr_smoke8_M1 | 54/46 | 278703/3031 | 242825/1862 | 1296/311 | [17.409195550513687, 42.02892075912677] | 0.0898051715332743 |
| formal_c0c8ae9799ad_mr_val_M1 | 7056/7056 | 35758307/797569 | 35758307/797569 | 169335/51473 | [35.213078676431905, 116.0589778418187] | 10.29402614062998 |

未知请求用量的预算账保留保守预约，不能将它当成精确实际 token。峰值显存（5 秒采样可能漏掉短峰）、服务生命周期 GPU 小时、重试、格式失败、空候选、预算中断、工具频率、提前停止、缓存命中与逐 query 失败原因保留在 checks/report_snapshot.json、各配对文件和 run/traces。未运行不填数值。

开发低/高预算质量与上表实际成本一起报告；方法的请求数、Controller 图像呈现和实际 token 并不完全匹配，因此不能将额外推理计算的收益全部归于调度或记忆。

失败实例：
- dev_low_v1_mr_dev50_B1, qid=479, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_B1, qid=2216, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_B1, qid=5220, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_B1, qid=5409, status=failed: ValueError: final invalid after at most one retry: MR candidate outside segment-relative timeline; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_B1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_M1, qid=140, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_M1, qid=479, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_M1, qid=1206, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_M1, qid=1441, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_low_v1_mr_dev50_M1, qid=1680, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_A1, qid=2953, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_A1, qid=6099, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_A1, qid=6415, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_A1, qid=8429, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_A1, qid=8796, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_M1, qid=2953, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_M1, qid=6099, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_M1, qid=6415, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_M1, qid=8429, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_hd_smoke8_M1, qid=8796, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_A1, qid=140, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_A1, qid=479, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_A1, qid=1206, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_A1, qid=1347, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_A1, qid=1680, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B0, qid=1206, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B0, qid=1441, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B0, qid=1680, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B0, qid=2216, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B0, qid=5220, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B1, qid=1206, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B1, qid=2216, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B1, qid=4994, status=failed: ValueError: final invalid after at most one retry: MR requires an array of at most ten candidates; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B1, qid=5220, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_B1, qid=5409, status=failed: ValueError: final invalid after at most one retry: MR candidate outside segment-relative timeline; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_M1, qid=140, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_M1, qid=479, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_M1, qid=1206, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_M1, qid=1441, status=failed: ValueError: final invalid after at most one retry: MR candidate outside segment-relative timeline; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_dev50_M1, qid=2216, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_A1, qid=5095, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_A1, qid=6099, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_A1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_A1, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_A1, qid=8429, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_B0, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_B0, qid=9291, status=failed: ValueError: final invalid after at most one retry: MR requires an array of at most ten candidates; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_B1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_B1, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_M1, qid=5095, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_M1, qid=6099, status=failed: ValueError: controller_0 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_M1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_M1, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v1_mr_smoke8_M1, qid=8429, status=failed: ValueError: controller_1 invalid after at most one retry: controller action schema mismatch; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_hd_dev50_M1, qid=1206, status=failed: ValueError: controller_0 invalid after at most one retry: HD local clip range must contain 1..24 clips; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_hd_dev50_M1, qid=1441, status=failed: ValueError: controller_0 invalid after at most one retry: HD local clip range must contain 1..24 clips; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_hd_dev50_M1, qid=1821, status=failed: ValueError: controller_0 invalid after at most one retry: HD local clip range must contain 1..24 clips; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_hd_dev50_M1, qid=9217, status=failed: ValueError: controller_0 invalid after at most one retry: HD local clip range must contain 1..24 clips; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_hd_dev50_M1, qid=9908, status=failed: ValueError: controller_0 invalid after at most one retry: HD local clip range must contain 1..24 clips; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_dev50_M1, qid=1206, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_dev50_M1, qid=1441, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_dev50_M1, qid=2216, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_dev50_M1, qid=3839, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_dev50_M1, qid=5409, status=failed: ValueError: final invalid after at most one retry: MR requires an array of at most ten candidates; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_A1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_A1, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_B0, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_B0, qid=9291, status=failed: ValueError: final invalid after at most one retry: MR requires an array of at most ten candidates; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_B1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_B1, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_M1, qid=6415, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- dev_v2_mr_smoke8_M1, qid=6536, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- formal_c0c8ae9799ad_mr_val_M1, qid=12, status=failed: ValueError: final invalid after at most one retry: MR candidate outside segment-relative timeline; 原请求、图像与解析状态在对应 trace/SQLite。
- formal_c0c8ae9799ad_mr_val_M1, qid=39, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- formal_c0c8ae9799ad_mr_val_M1, qid=68, status=failed: ValueError: overview invalid after at most one retry: truncated/nonterminal model response; 原请求、图像与解析状态在对应 trace/SQLite。
- formal_c0c8ae9799ad_mr_val_M1, qid=76, status=failed: ValueError: final invalid after at most one retry: empty MR candidates: official R1 cannot score this query; 原请求、图像与解析状态在对应 trace/SQLite。
- formal_c0c8ae9799ad_mr_val_M1, qid=137, status=failed: ValueError: final invalid after at most one retry: MR candidate outside segment-relative timeline; 原请求、图像与解析状态在对应 trace/SQLite。

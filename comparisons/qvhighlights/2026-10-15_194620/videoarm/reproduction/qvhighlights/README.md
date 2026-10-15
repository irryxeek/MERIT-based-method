# VideoARM 在 QVHighlights 上的开源模型迁移评测

2026-10-14 用户范围修订：后续只执行 **M1**，MR/HD 分别跑 smoke8、dev50、完整 val/test；B0/B1/A1 与低预算条件暂缓。正式计划 6,184 次推理。4 路真实请求重放实测约 3.58× 串行吞吐；完整 query 提速以 dev50 实测为准。见 [M1_SCOPE_AND_EFFICIENCY.md](M1_SCOPE_AND_EFFICIENCY.md)、[PAPER_COMPARISON.md](PAPER_COMPARISON.md)。

当前开发协议为 v2_stop_contract。v1 的 stop 缺少 question 被误判为失败，旧源码与结果保留；修订记录见 [DEVELOPMENT_AMENDMENT_V2.md](DEVELOPMENT_AMENDMENT_V2.md)。不改写旧失败或以旧受影响分数判定方法收益。

独立源码 `../../open_source_qvh/videoarm/qvhighlights/`；所有新增日志、清单、协议、官方原件和报告在本目录。`QVH_DATA_ROOT` 默认 `/path/to/storage/videoarm-repro/qvh_data`，大文件、逐请求图片、PTS、缓存与 run 状态保存在该目录。共享媒体和 Qwen 权重只读复用。

先看 [FINAL_REPORT.md](FINAL_REPORT.md)、[progress.json](progress.json)、[PROJECT_AUDIT.md](PROJECT_AUDIT.md)、[DATA_AUDIT.md](DATA_AUDIT.md)、[PROTOCOL.md](PROTOCOL.md)、[METHOD_DELTA.md](METHOD_DELTA.md)。未运行真实推理时没有本方法成绩。官方作者示例文件不属于本方法预测。

## 当前环境和 CPU 检查

```bash
cd /path/to/storage/videoarm-repro
export QVH_DATA_ROOT=/path/to/storage/videoarm-repro/qvh_data
QVH_PY=/path/to/storage/longvt-repro/env/inference/bin/python
$QVH_PY -m pytest reproduction/qvhighlights/tests -q
$QVH_PY -m ruff check open_source_qvh reproduction/qvhighlights/tools reproduction/qvhighlights/tests --select F,E9
$QVH_PY reproduction/qvhighlights/tools/prepare_qvh.py --probe dev50_mapping --workers 3
$QVH_PY reproduction/qvhighlights/tools/run_qvh.py --task mr --method M1 --split smoke8 --run-id dry_mr_m1 --dry-run
$QVH_PY reproduction/qvhighlights/tools/run_qvh.py --task hd --method M1 --split dev50 --run-id dry_hd_m1 --dry-run
```

重新做全量时间轴审计可用 `prepare_qvh.py --probe all --workers 12`。已审计媒体按路径、size、mtime 的缓存身份恢复 PTS，正式阶段另核对冻结的媒体/PTS SHA。失败按原计划保留。

环境未升级。完整 freeze 在 `checks/client_environment_20261013.txt`、`checks/server_environment_20261013.txt`；仅客户端所需依赖见 `configs/requirements-client.lock`。重建一个新客户端环境时，可 `python3.10 -m venv "$QVH_DATA_ROOT/env-client"` 后在新环境安装这份 lock。模型服务使用已有兼容 vLLM 0.17 环境；完整包锁可能包含本地 wheel/direct URL，迁移到另一主机需保留那些 wheel，不能把复制虚拟环境当作重建。当前命令默认直接复用已核实环境。

## 独立推进与恢复

```bash
$QVH_PY reproduction/qvhighlights/tools/launch_qvh_service.py
$QVH_PY reproduction/qvhighlights/tools/advance_qvh.py --wait-resources
```

`advance_qvh.py` 只管理 `videoarm-qvh-*`。正式开发预算只用 default_v2 与 dev_low_v2，不扩展模型、种子、训练或 AV。真实 image binding 和 forced 工具日志在 `qvh_data/probes/`，不计入自主方法效果。开发四方法的真实运行目录为 `qvh_data/runs/dev_v2_{task}_{split}_{method}`；低点为 `dev_low_v2_{task}_dev50_{method}`。

系统服务状态与日志：

```bash
systemctl --user status videoarm-qvh-worker.service videoarm-qvh-media-audit.service
journalctl --user -u videoarm-qvh-worker.service -n 50 --no-pager
$QVH_PY reproduction/qvhighlights/tools/refresh_progress.py
$QVH_PY reproduction/qvhighlights/tools/write_handoff.py
```

断电恢复只需启动已有 worker 或重新执行同一 `advance_qvh.py`。run_qvh 总是读原状态库；成功、失败和预算中断终态保留，只有 pending/running/paused 按原请求 ledger 恢复。若接口/格式门槛导致 worker 退出，应先检查 `checks/worker_status.json` 和 `logs/*worker.log` 中的具体错误，不直接换 run_id 重做失败的正式样本。

手动开发推理：

```bash
$QVH_PY reproduction/qvhighlights/tools/run_qvh.py --task mr --method M1 --split dev50 --run-id dev_v2_mr_dev50_M1 --workers 4 --resume
$QVH_PY reproduction/qvhighlights/tools/run_qvh.py --task hd --method M1 --split dev50 --run-id dev_v2_hd_dev50_M1 --workers 4 --resume
```

每个 run 包含 `state.sqlite`、`identity.json`、`predictions.jsonl`、`traces/{qid}/`、`frames/{qid}/`、`progress.json`。所有 stage raw response、解析状态、预算预约、实际 usage、重试和原图片映射均保留。MR 与 HD 不混用 run_id。

## 校验、评分、报告

```bash
$QVH_PY reproduction/qvhighlights/tools/validate_predictions.py --task mr --manifest reproduction/qvhighlights/manifests/smoke8.jsonl --predictions "$QVH_DATA_ROOT/runs/dev_v2_mr_smoke8_M1/predictions.jsonl" --allow-partial
$QVH_PY reproduction/qvhighlights/tools/score_qvh.py --task mr --manifest reproduction/qvhighlights/manifests/smoke8.jsonl --predictions "$QVH_DATA_ROOT/runs/dev_v2_mr_smoke8_M1/predictions.jsonl" --allow-partial --out "$QVH_DATA_ROOT/runs/dev_v2_mr_smoke8_M1/scores.json"
$QVH_PY reproduction/qvhighlights/tools/report_qvh.py --run "$QVH_DATA_ROOT/runs/dev_v2_mr_smoke8_M1" --out reproduction/qvhighlights/checks/cost_mr_smoke8_M1.json
```

校验器不会修补预测。部分集评分必须显式 `--allow-partial`，输出覆盖率、原计划分母的自定义失败记零指标及 `.per_query.jsonl` 未舍入指标。当前仅执行 M1，report 记录实际成本；历史 B1/M1 配对统计保留，不据缺少当前对照的新实验断言方法收益。HD 把 `--task mr` 改为 `--task hd`，使用对应独立 run 的预测。

完整 val 前，worker 根据真实 dev 成本保存 `checks/full_run_estimate.json`，`freeze_qvh.py` 验证门槛并写 `checks/formal_freeze.json`；正式 run_id 从其中的 `run_ids` 读取。run_qvh 在正式阶段拒绝任意新 run_id、`--limit`、协议/模型/媒体身份漂移。test GT 只在独立 scorer 内读取，不能通过推理端传入 GT 路径。

## 暂停本任务

```bash
systemctl --user disable --now videoarm-qvh-worker.service
systemctl --user stop videoarm-qvh-model.service
```

worker 和 run_qvh 的中断保留请求预约和状态；在服务器响应不确定时，不退还预约。如需同时暂停 CPU 审计，另执行 `systemctl --user stop videoarm-qvh-media-audit.service`。不要启动、停止或改动 `videoarm-svh-*`，现有 SVHighlights 推理仍暂停。

执行并发由 `configs/execution_m1.json` 和正式 freeze 固定；命令行正式运行需传 `--workers 4`。原已终态 MR smoke8 由 worker 原样保留，不在新源码下重新清零执行。吞吐重放完整记录见 `checks/concurrency_benchmark_m1.json`。

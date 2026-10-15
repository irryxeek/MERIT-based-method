"""Generate a report from evidence. Absent model runs never become scores."""

import _bootstrap  # noqa: F401
import json
import os
import time
from pathlib import Path

from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    code_identity,
    file_hash,
)
from report_qvh import costs
from refresh_progress import refresh


def main():
    root = Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
    now = time.strftime("%Y-%m-%d %H:%M:%S %z")
    progress = refresh(root)
    source = json.loads((ARTIFACTS / "checks/source_lock.json").read_text())
    inventory = json.loads((ARTIFACTS / "checks/data_inventory.json").read_text())
    development = json.loads(
        (ARTIFACTS / "checks/data_audit_dev50_mapping.json").read_text()
    )
    tests = ARTIFACTS / "checks/offline_tests.json"
    test_info = (
        json.loads(tests.read_text()) if tests.exists() else {"status": "pending"}
    )
    worker_path = ARTIFACTS / "checks/worker_status.json"
    worker = (
        json.loads(worker_path.read_text())
        if worker_path.exists()
        else {"status": "not_started"}
    )
    results, run_costs = [], {}
    for p in sorted((root / "runs").glob("*/scores.json")):
        result = json.loads(p.read_text())
        results.append(
            {
                "run_id": p.parent.name,
                "status": result["status"],
                "coverage": result["coverage"],
                "brief": (result["official_metrics"] or {}).get("brief"),
                "score_path": str(p),
            }
        )
        run_costs[p.parent.name] = costs(p.parent)
    databases = list((root / "runs").glob("*/state.sqlite"))
    paired = {
        p.name: json.loads(p.read_text())
        for p in sorted((ARTIFACTS / "checks").glob("paired_*.json"))
    }
    atomic_json(
        ARTIFACTS / "checks/report_snapshot.json",
        {
            "generated": now,
            "results": results,
            "costs": run_costs,
            "paired_comparisons": paired,
            "worker": worker,
            "progress": progress,
            "autonomous_run_databases": len(databases),
        },
    )
    media = progress["stages"]["P0"]["full_timeline"]
    lines = [
        "# VideoARM 在 QVHighlights 上的开源模型迁移评测",
        "",
        f"产物核对时间：{now}。依据文件和独立 SQLite 状态库生成。",
        "",
        "已实现独立 MR/HD 推理、标签隔离、PTS 采样、记忆摘要、B0/B1/M1/A1、预算、恢复、严格预测校验、固定官方评分和源视频聚类 bootstrap。",
        "",
        "2026-10-14 开发协议修订一次为 v2_stop_contract：v1 将 stop 缺少 question 误判为失败，旧分数受接口错误影响，不能用来判定 VideoARM 算法收益。旧源码、预算、预测、失败和评分保留；后续正式结果使用 v2，同一训练开发名单、模型和预算不变。详见 DEVELOPMENT_AMENDMENT_V2.md。",
        "",
        f"离线验证：`{json.dumps(test_info, ensure_ascii=False)}`。真实自主运行数据库 {len(databases)}；已有评分文件 {len(results)}；执行状态 `{worker['status']}`。",
        "",
        "真实推理和完整评测仅按下表及 progress.json 认定。作者示例和 CPU 合成测试不属于本方法成绩。空表表示暂无已评分真实运行。",
        "",
        "| run_id | 提交状态 | 成功/计划 query | 官方指标 |",
        "|---|---|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['run_id']} | {r['status']} | {r['coverage']['predicted']}/{r['coverage']['planned']} | {json.dumps(r['brief'], ensure_ascii=False)} |"
        )
    lines += [
        "",
        "## 数据与时间轴",
        "",
        "| split | query | 数据集片段 | 源 YouTube 视频 | 有媒体 query |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, v in inventory["splits"].items():
        lines.append(
            f"| {split} | {v['queries']} | {v['segments']} | {v['source_videos']} | {v['media_queries']} |"
        )
    lines += [
        "",
        f"dev50 完整 PTS 与首尾解码通过 {development['probe']['passed']}/{development['probe']['planned']}；全量检查 {media.get('completed', 0)}/{media.get('planned', 10148)}，失败 {media.get('failed', 0)}。",
        "",
        "初始 0.25 秒一致性检查隔离 222 个片段；标注减媒体时长差异 −0.411078 至 1.742875 秒，原始检查保留。登记的 identity-PTS 映射要求每个官方 clip 有真实帧，不改变标注、像素或 clip 数量。部分末尾片段的可见时长不足 2 秒；单帧不能证明整段动作。预测与解码使用当前数据集片段起点，不叠加文件名中的源偏移。",
        "",
        "## 评测、成本与方法限制",
        "",
        f"固定官方评测器 commit `{source['official_commit']}`。作者 1,550-query 示例结果与参考文件完全一致，这只验证评测器安装。逐 query 使用同一官方 AP/IoU/三位标注者函数，并核对聚合。开发子集缺少某个时长组时，该组为未定义，使用非空组的官方函数，未修改官方文件。",
        "",
        "完整合法覆盖才标记完整官方提交；部分结果另外报告 B1/M1 共同成功子集和原计划分母的失败指标记零自定义汇总。不会伪造预测使评测运行。常量或全零 HD 向量仍有官方排序分数，不能直接当作零指标。配对 bootstrap 保留原始精度，seed=42、10,000 次，按源 YouTube 视频聚类。",
        "",
        "此次 sensory/result/working-objective 表达与论文 HM³ 及公开代码的三个工具列表存在差异。working 只保留短行动目的；时间范围工具接口、独立 MR/HD 输出、HD 全片初始化与固定摘要为任务适配，不称为论文所有机制或其 QVHighlights 分数的完整恢复。详见 METHOD_DELTA.md。",
        "",
        "A1 只传不可变初始化、最新工具语义和去语义化覆盖记账。HD 累积输出分数保存在程序状态中，不交回 Controller；MR 最终聚合也只看最新工具记录。历史像素和全文保留在旁路 trace，A1 无回查接口。",
        "",
        "预算默认高点为 120 局部帧；登记低点为 60 局部帧、每动作最多 12 帧，B1 仍执行五个均匀局部观察，窗口宽度缩至默认 0.6，HD 区间按完整 clip 对齐。这是开发成本敏感性实验，不能声称两方法总计算完全匹配。",
        "",
        "真实 smoke/dev 成本测量后才生成 checks/full_run_estimate.json，提供完整 val/test 的时间、存储和 GPU 小时估计；该文件缺失时，没有实测估计。冷启动请求/token/图像和共享初始化后的实际新增成本分别记账。GPU 服务监控包含启动和空闲时间。",
        "",
        "## 剩余问题与恢复",
        "",
        f"执行阻塞：`{json.dumps({k: v for k, v in worker.items() if k in ('status', 'error', 'phase', 'free_indices')}, ensure_ascii=False)}`。实时资源见 checks/gpu_readiness.json；仅空闲且无计算进程的两张 GPU 可用于默认 BF16 TP=2。",
        "",
        "独立 videoarm-qvh-worker.service 等待资源，通过真实 binding/forced 接口检查后执行自主 smoke8/dev50、低/高预算诊断，保存估时、冻结、val/test 预测、评分和报告。接口错误会保存具体原因并停止该依赖阶段，不清零预算或用新的正式 run_id 重试。videoarm-qvh-media-audit.service 独立完成 CPU 审计。",
        "",
        "现有 SVHighlights 推理继续暂停；其源码、配置、timer、ledger 和预测保持冻结。所有环境、运行、评分、恢复和暂停命令见 README.md。尚未完成的工作由 progress.json 和各 run/state.sqlite 标识；工程完成不能代替真实推理验收。",
    ]
    lines += [
        "",
        "## B1/M1 配对统计与实际成本",
        "",
        "下表为 M1−B1，单位为百分点；共同成功集与自定义全计划失败记零口径分开。原始完整指标、三阈值 HD 与覆盖率均在 checks/paired_*.json。暂无真实结果时，下表为空。",
        "",
        "| 比较文件 | 口径 | 指标 | 差值 | 95% CI | 有效源视频簇 |",
        "|---|---|---|---:|---|---:|",
    ]
    for name, value in paired.items():
        comparison = value.get("paired", {})
        for scope in ("common_success_ci", "custom_full_plan_failure_zero_ci"):
            for metric, stats in comparison.get(scope, {}).items():
                lines.append(
                    f"| {name} | {scope} | {metric} | {stats['estimate']} | {stats['ci95']} | {stats['clusters']} |"
                )
    lines += [
        "",
        "| run_id | 冷启动等效/新增请求 | 输入/输出 token | 新增输入/输出 token | 呈现/独立帧 | P50/P95 秒 | 活动 GPU 小时 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, c in run_costs.items():
        b = c["totals"]
        lines.append(
            f"| {name} | {b['requests']}/{b['actual_requests']} | {b['input_tokens']}/{b['output_tokens']} | {b['actual_input_tokens']}/{b['actual_output_tokens']} | {b['images']}/{c['unique_frames_sum_over_queries']} | {c['elapsed_p50_p95_seconds']} | {c['gpu_hours']} |"
        )
    lines += [
        "",
        "未知请求用量的预算账保留保守预约，不能将它当成精确实际 token。峰值显存（5 秒采样可能漏掉短峰）、服务生命周期 GPU 小时、重试、格式失败、空候选、预算中断、工具频率、提前停止、缓存命中与逐 query 失败原因保留在 checks/report_snapshot.json、各配对文件和 run/traces。未运行不填数值。",
        "",
        "开发低/高预算质量与上表实际成本一起报告；方法的请求数、Controller 图像呈现和实际 token 并不完全匹配，因此不能将额外推理计算的收益全部归于调度或记忆。",
        "",
        "失败实例：",
    ]
    for name, c in run_costs.items():
        for failure in c["failures"][:5]:
            lines.append(
                f"- {name}, qid={failure['qid']}, status={failure['status']}: {failure['error']}; 原请求、图像与解析状态在对应 trace/SQLite。"
            )
    if not run_costs:
        lines.append(
            "真实推理未运行，暂无本方法失败实例。离线测试曾发现的浮点 R1 调用和失败原因覆盖错误已修复，旧测试日志保留，不能把合成失败当成真实模型失败案例。"
        )
    (ARTIFACTS / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        ARTIFACTS / "checks/implementation_snapshot.json",
        {
            "date": now,
            "code": code_identity(),
            "configs": {
                p.name: file_hash(p) for p in (ARTIFACTS / "configs").glob("*.json")
            },
            "scope": "engineering snapshot, not the pre-val formal freeze",
        },
    )
    print(
        json.dumps(
            {
                "report": "FINAL_REPORT.md",
                "real_scored_runs": len(results),
                "worker": worker["status"],
            }
        )
    )


if __name__ == "__main__":
    main()

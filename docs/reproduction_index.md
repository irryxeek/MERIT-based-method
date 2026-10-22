# 实验实现与复现入口

更新：2026-10-22。本页对应发布仓库；历史指标见[结果汇总](experimental_results_report_2026-10-22.md)。代码补齐不代表新增实验完成。新机器的路径、源码哈希和运行标识会变化，应创建新运行并重新冻结输入。

## 方法、实现与结果对应表

| 方法 / 数据集 | 实现入口 | 配置 / 输入 | 已有结果与范围 |
|---|---|---|---|
| MERIT / QVHighlights | [`src/merit_qvh/`](../src/merit_qvh/) | [`main_gpu_v1.json`](../configs/main_gpu_v1.json) | [原适配](merit_qvhighlights_adaptation.md)，完整 val 1,550 |
| MERIT 密集初始化 / 局部复评 | [`hd_adapter`](../experiments/hd_adapter/README.md) | [`hd_adapter.example.json`](../configs/hd_adapter.example.json) | [适配记录](merit_hd_adapter_implementation_2026-10-16.md)，固定 train100 |
| 高光记忆、证据与读取实验 | [`highlight_evidence`](../experiments/highlight_evidence/README.md) | [四类 cue](../configs/highlight_value_schema.example.json)、父运行缓存 | [结果总汇](reproduction_results_summary_2026-10-19.md)，固定 train20；R2 未运行 |
| CLIP / VTG-GPT / QVHighlights | [`零样本工作区`](../baselines/zeroshot/README.md) | `baselines/zeroshot/configs/` | [基线报告](zeroshot_baselines/RESULTS_2026-10-19.md)，完整 val 1,550 |
| REZE B0 / QVHighlights | [`launch_reze.py`](../baselines/zeroshot/scripts/launch_reze.py) | [`reze_protocol_v2.json`](../baselines/zeroshot/configs/reze_protocol_v2.json) | [基线报告](zeroshot_baselines/REZE_RESULTS_2026-10-19.md)，完整 val 1,550 |
| REZE P_abs / P_pair | [`全量运行器`](../baselines/zeroshot/experiments/reze_fullval/runner.py)、[`比较模块`](../baselines/zeroshot/experiments/reze_evidence/) | [`计划模板`](../configs/reze_evidence_fullval_v1.example.json)、完成的 B0 原始记录 | [全量报告](reze_fullval_results_2026-10-20.md)，完整 val 1,550 |
| REZE / QFVS | [`reze_qfvs`](../baselines/zeroshot/experiments/reze_qfvs/README.md) | `--scope pilot4` / `full180` | [报告](reze_qfvs_analysis_2026-10-20.md)，仅四查询完成 |
| REZE / MR.HiSum | [`reze_mrhisum`](../baselines/zeroshot/experiments/reze_mrhisum/README.md) | `--count 100` / `--all-verified` | [全量可用集报告](reze_mrhisum_full_execution_2026-10-21.md)，1,888 / 2,000 |
| REZE / TVSum | [`reze_tvsum`](../baselines/zeroshot/experiments/reze_tvsum/README.md) | 固定 UMT 标注、val10 / all50 | [报告](reze_tvsum_execution_2026-10-22.md)，50 条完成 |
| REZE / SVHighlights | [`reze_svhighlights`](../baselines/zeroshot/experiments/reze_svhighlights/README.md) | 历史配对池与七场清单 | [范围与失败记录](reze_cross_dataset_2026-10-20.md)，无有效指标；发布版修复 GPU 寻址，尚未重跑 |

TVHighlights 与 TVSum 是不同数据集。TVHighlights 尚无可用数据和实验实现，不能用 TVSum 结果代替。

## 环境与机器路径

从仓库根目录操作，使用独立环境。不要对正在运行历史实验的环境直接升级依赖。

```bash
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install -r baselines/zeroshot/requirements-reze.txt
python -m pip install --no-deps -e .
source configs/paths.example.sh
```

系统还需要 `ffmpeg` / `ffprobe`；GPU 推理需要兼容的 NVIDIA 驱动。版本以依赖清单为准，未在本轮重新安装或验证所有平台的组合。

[`paths.example.sh`](../configs/paths.example.sh) 导出数据、模型、源码、产物路径。可复制为被 Git 忽略的 `configs/paths.local.sh`。在加载脚本前设置 `REPRO_DATA_ROOT`、`MERIT_ARTIFACT_ROOT`、`ZEROSHOT_ARTIFACT_ROOT` 可指向数据盘。将 `REPRO_GPU_UUIDS` 设置为本次分配设备的完整 UUID，逗号分隔；需要数字索引的旧入口由 `--gpus` 显式传入。脚本不会自行分配 GPU。

JSON 模板中的 `${变量}` 不能直接交给旧运行器，先展开为本机配置：

```bash
python scripts/materialize_config.py configs/hd_adapter.example.json configs/hd_adapter.local.json
```

展开工具拒绝覆盖已有文件。HD 模板保留历史排除清单的路径要求；要复现固定划分，需准备这些清单并重新做 preflight。不要删除排除条件后仍声称复现了同一划分。历史硬件 preflight 的哈希已从模板中移除，必须使用本机实际检查结果。

## REZE 的 QVHighlights 新运行

将固定 revision 的 Qwen3-VL-8B-Instruct 放在 `$REZE_MODEL_PATH`，完整视频放在 `$REPRO_DATA_ROOT/QVHighlights/videos_full/videos/`。模型 revision、16 个文件的大小与 SHA-256 来自[模型核验模板](../baselines/zeroshot/manifests/reze_model_audit.example.json)。

```bash
cd "$ZEROSHOT_ROOT"
"$REPRO_PYTHON" scripts/prepare_reze_inputs.py
CUDA_VISIBLE_DEVICES="$REPRO_GPU_INDEX" "$REPRO_PYTHON" scripts/check_reze_backend.py
CUDA_VISIBLE_DEVICES="$REPRO_GPU_INDEX" "$REPRO_PYTHON" scripts/check_reze_sustained.py
"$REPRO_PYTHON" scripts/launch_reze.py --run "$ZEROSHOT_ARTIFACT_ROOT/runs/reze_val_v2" --gpus "$REPRO_GPU_INDEX"
```

`REPRO_GPU_INDEX` 需事先设为同一张已分配卡的数字索引。第一条 Python 命令只核验权重、生成无标签推理清单和评估文件哈希；后续三条会进行 GPU 推理。最后一步要求持续推理检查完成 21 条，再执行全量和官方 HD 评估。新准备入口不依赖先跑 CLIP 或 VTG-GPT。已有完整 QV 清单、标签及实验输出均不由本轮发布重写。

B0 完成后，创建绑定新 B0 的比较计划：

```bash
"$REPRO_PYTHON" scripts/prepare_pair_plan.py \
  --baseline-run "$ZEROSHOT_ARTIFACT_ROOT/runs/reze_val_v2" \
  --store "$ZEROSHOT_ARTIFACT_ROOT/reze_pairs_new" \
  --output "$MERIT_ROOT/configs/reze_evidence_fullval_v1.json" --gpus "$REPRO_GPU_INDEX"
"$REPRO_PYTHON" experiments/reze_fullval/runner.py --prepare "$MERIT_ROOT/configs/reze_evidence_fullval_v1.json"
"$REPRO_PYTHON" experiments/reze_fullval/runner.py --run "$ZEROSHOT_ARTIFACT_ROOT/reze_pairs_new/runs/val1550_pairs"
```

计划生成器核对 B0 完成凭证、记录哈希并绑定新文件；准备阶段进一步检查模型、像素来源、标签身份和完整分母。模板中的旧结果哈希仅用于追溯，不能通过字符串替换绕过检查。比较仍只运行 P_abs / P_pair，不自动进入记忆或回看阶段。

## 跨数据集入口

仅运行跨数据集时，用 `scripts/prepare_reze_inputs.py --model-only` 准备模型核验，不要求 QV 原始视频。数据准备材料必须位于配置路径下，并通过各数据集的原有审计：

- **QFVS**：`raw/video_audit.json`、四条视频、原始 Oracle/User summaries 和 `source/eval/Tags.mat`。下载与审计脚本位于根目录 `scripts/`，来源见[视频记录](qfvs_raw_videos_2026-10-20.md)。
- **MR.HiSum**：`availability.json`、媒体 receipts、`annotation_audit.json` 和原始 H5 / split / metadata。根目录保留下载、补试工具；原作者评估代码和许可证随实验发布。
- **TVSum**：原视频审计、UMT 标注 archive / receipt / dataset audit；`dataset.py` 核对固定 50 条和标注网格。
- **SVHighlights**：还需要 `$SVH_PAIRED_RUN` 的历史预测 ID、视频 receipts/counts，以及 `$SVH_INFERENCE_MANIFEST`。当前实现保留固定七场协议，不能直接作为官方 320 条入口。

QFVS 示例（从零样本工作区执行）：

```bash
"$REPRO_PYTHON" experiments/reze_qfvs/prepare.py --scope pilot4 --run "$ZEROSHOT_ARTIFACT_ROOT/qfvs_new"
"$REPRO_PYTHON" "$ZEROSHOT_ARTIFACT_ROOT/qfvs_new/source_snapshot/preflight_media.py" --run "$ZEROSHOT_ARTIFACT_ROOT/qfvs_new"
"$REPRO_PYTHON" "$ZEROSHOT_ARTIFACT_ROOT/qfvs_new/source_snapshot/dispatch.py" --run "$ZEROSHOT_ARTIFACT_ROOT/qfvs_new"
```

MR.HiSum 使用相同 prepare → preflight → dispatch 流程，准备命令改为 `experiments/reze_mrhisum/prepare.py --count 100 --run ...`；单分片可用全量改用 `--all-verified`。历史双分片脚本另需相应 `plan.json`，不能拿旧计划接续新路径的运行。

TVSum 用 `experiments/reze_tvsum/dataset.py` 审计后，`launch.py --run 新目录 --unit-prefix 新名称` 创建两分片任务和 systemd 用户定时器。发布版将路径环境写入该运行的 `publication.env`，使后台准备过程使用相同配置。QFVS / MR.HiSum / SVHighlights 的 dispatcher 是一次有界执行，持续自动恢复需另行配置用户 timer；重复调用会复用已校验的检查点。

## 历史实验与验证范围

MERIT 的粒度、纯记忆、R1/R2 和 REZE A 阶段依赖前一轮完整缓存、固定查询划分及其哈希。源码已发布，这些未公开原始缓存仍需自行提供；R1 默认保留原父协议限制。冻结结果和 `comparisons/` 内代码快照仅用于历史核验，新入口不会改写它们。

CPU 测试按独立工作区执行，避免各数据集的同名 `shared` / `media` 模块互相覆盖：

```bash
cd "$MERIT_ROOT"
python -m pytest tests -q
cd "$ZEROSHOT_ROOT"
python -m pytest tests -q
python -m pytest experiments/reze_qfvs/tests -q
python -m pytest experiments/reze_mrhisum/tests -q
python -m pytest experiments/reze_tvsum/tests -q
python -m pytest experiments/reze_svhighlights/tests -q
```

每条命令单独启动解释器。缺少原视频或 tokenizer 时，相关集成检查显式跳过；CPU 测试通过不代表已重新验证 GPU 精度或取得新的指标。发布完整性检查见 `scripts/verify_publication.py`。

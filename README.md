# MERIT / REZE 视频高光检测实验

本项目从 MERIT 的三键记忆检索、时序邻域过滤和完整时间轴高光评分出发，保存 QVHighlights 适配实验及后续零样本对照。原论文评估视频问答，**MERIT 原适配并非复现论文已有的 QVHighlights 分数**。下方运行命令对应原 MERIT 主实验；REZE 最新结果单独归档。

**发布约定：以后每次推送更新，一并同步新增或修改的实验报告、结果分析、研究方案和文档索引。** 详见[报告同步发布约定](docs/report_publication_workflow.md)和[本次同步清单](docs/report_publication_manifest.json)。

**资源约定：[后续新运行可使用任意空闲 GPU](docs/gpu_resource_policy_2026-10-25.md)。** 可独立拆分的任务优先多卡并行，已有冻结运行保留原配置。

**本周组会汇报（2026-10-21–27）：[进度周报](docs/weekly_experiment_progress_2026-10-21_to_2026-10-27.md) · [更新版 15 页 PPT](docs/presentations/ReVALUE_实验进度汇报_2026-10-21至10-27_v2_TaskSpec补充.pptx) · [原 14 页版](docs/presentations/ReVALUE_实验进度汇报_2026-10-21至10-27.pptx)。** 更新版第 4 页介绍 TaskSpec；配套逐页讲稿见 [PPT 交付目录](docs/presentations/revalue_weekly_2026-10-21_to_2026-10-27/README.md)。

**最新比较器讨论：[四态比较器诊断](docs/revalue_four_state_comparator_diagnosis_2026-10-27.md)。** 保留 QV / TVSum 比较状态分布、回看触发及已知结论边界，无新增实验模型调用。

**最新离线推进：[TVSum 来源链与失败诊断](docs/revalue_tvsum_offline_diagnosis_v1_2026-10-27.md)。** 本轮零新增模型调用，19 项 CPU 测试通过；仅改变通用对象约束的[单变量协议](docs/revalue_tvsum_taskspec_protocol_v1_2026-10-27.md)已准备，400 次正式调用及独立预算仍待授权。[CPU 实现](baselines/zeroshot/experiments/revalue_tvsum_offline/README.md)。

**最新跨数据集结果：[ReVALUE / TVSum val10](docs/revalue_tvsum_transfer_results_2026-10-27.md)。** 四组、670 次完成；B0 / P_pair 为 61.32 / 58.38，C_direct4/E_memory 与 B0 的评分完全相同。通用任务约束和比较决策需修订；含同分敏感性、事实抽查及成本。见[冻结协议](docs/revalue_tvsum_transfer_protocol_2026-10-27.md)与[实现入口](baselines/zeroshot/experiments/revalue_crossdataset/README.md)。

**阶段总结：[实验成果与问题（2026-10-27）](docs/experimental_progress_and_issues_2026-10-27.md)。** REZE 候选复评已在 val1550 提升指标；ReVALUE 记忆与定向观察尚无稳定精度收益，当前优先验证视觉证据可靠性。跨数据集范围及限制一并列出。

**此前定向核查：[ReVALUE 定向物体证据核查](docs/revalue_object_presence_results_2026-10-27.md)。** 60/60 次请求完成，150 项 CPU 测试通过；两臂状态完全相同，14 个明确正例全部保留，六个模糊例均未选择无法判断。更长提示未改善判断，内容关未通过；未启动新 HD 评估。见[冻结协议](docs/revalue_object_presence_protocol_2026-10-26.md)。

**此前容量与输出结果：[ReVALUE 来源压缩与 37 次观察输出探针](docs/revalue_memory_compaction_results_2026-10-26.md)。** 14 视图节省 25.30% token，累计补回 104 个旧事实 ID；135 项 CPU 测试通过。字符限制修复改善句子完整性，但视觉内容仍未通过，A/B/C 和新 100 条未启动。原[下一轮计划](docs/revalue_next_experiment_plan_2026-10-26.md)保留为快照，实际顺序见新协议。

**此前来源修复：[ReVALUE 来源绑定修复与 672 次单帧验证](docs/revalue_frame_grounding_repair_results_2026-10-26.md)。** 来源时间、片外引用与空描述入库已修复，86 项测试通过；固定跨场景错误得到纠正，但仍有视觉误判与记忆容量代价，单帧观察未晋级正式评分。见[本轮冻结协议](docs/revalue_frame_grounding_repair_protocol_2026-10-26.md)、[此前 28 次观察对照](docs/revalue_r2_observation_probe_results_2026-10-26.md)及[离线诊断](docs/revalue_r2_offline_diagnosis_2026-10-26.md)。R3/R4 未启动。

**整体框架与阶段快照：[现阶段方案](docs/revalue_current_experiment_plan_2026-10-26.md)。** 父 R2 修复版完成 100/100、零技术回退、7 条配对观察，观察组 mAP / Hit@1 为 44.41 / 71.00，精度未过门槛；后续观察诊断未新增 HD 指标，下一轮细化安排见上方新计划。[总体设计 v3](docs/revalue_execution_plan_v3_2026-10-23.md)、[R2 修复协议](docs/revalue_r2_repair_protocol_v3_2026-10-26.md)、[故障诊断](docs/revalue_r2_diagnosis_2026-10-26.md)及[启动快照](docs/revalue_r2_repair_execution_2026-10-26.md)保留。[ReVALUE 代码入口](baselines/zeroshot/experiments/revalue/README.md)。

**QFVS 全量验证已完成：[结果报告](docs/reze_qfvs_full_results_2026-10-25.md)。** 两组均 180/180；概念加时间分散 Oracle F1 为 44.86，均匀对照为 46.62，已单列剩余 176 条。原[启动记录](docs/reze_qfvs_full_validation_2026-10-23.md)保留冻结配置。

## 实验代码与复现入口（2026-10-22 补齐）

[统一复现索引](docs/reproduction_index.md) 对应 MERIT 适配与记忆实验、REZE QVHighlights 复评及跨数据集实现；[代码补齐报告](docs/repository_completion_2026-10-22.md)记录部署修改、280 项通过的 CPU 测试及 12 项依赖原媒体或 tokenizer 的跳过。新源码入口位于 `experiments/` 和 `baselines/zeroshot/`，历史结果仍在 `comparisons/`。

- [MERIT 密集初始化与局部复评](experiments/hd_adapter/README.md)、[高光记忆与证据实验](experiments/highlight_evidence/README.md)。
- [CLIP / VTG-GPT / REZE 零样本工作区](baselines/zeroshot/README.md)，包含 QFVS、MR.HiSum、TVSum、SVHighlights。
- [路径配置](configs/paths.example.sh)、[代码来源清单](docs/code_publication_manifest.json)、[发布完整性检查](scripts/verify_publication.py)。

10 月 22 日代码补齐本身未运行 GPU 实验；10 月 23 日已完成 SVHighlights 七场基线和 QFVS 四查询优化；QFVS 全量验证已于 10 月 24 日完成，TVHighlights 数据尚不可用。历史父运行缓存的要求见复现索引。

## 最新实验结果（2026-10-22）

[数据集说明](docs/datasets_inputs_outputs_metrics_2026-10-22.md) 整理六个数据集的特点、输入输出和评价指标，区分原任务与本项目适配协议，并列出当前实验覆盖。

[实验结果汇总报告](docs/experimental_results_report_2026-10-22.md) 汇总 QVHighlights 主实验、MERIT 开发对照及 REZE 跨数据集测试。TVSum 固定 val10 / all50 的 Top-5 mAP 为 **61.32 / 54.89**；MR.HiSum 可用 test 1,888 条的 mAP50 / mAP15 / 摘要 F1 为 **57.00 / 23.90 / 38.27**；QFVS 四查询 Oracle F1 为 **38.63**。不同指标和样本范围分别讨论，含计算开销、适配分析及尚未完成项目。

[会议纪要后的推进计划](docs/next_steps_after_meeting_2026-10-22.md) 将下一阶段分为补齐 SVHighlights 基线、候选对照、同预算定向补证及长视频粗到细验证，包含 MERIT / LensWalk 分工和推进门槛；最新启动状态见上方执行记录。详细数据集报告与 MERIT 开发实验已补齐，见[文档索引](docs/README.md)。

REZE B0、P_abs、P_pair 已完成全部 1,550 条 validation 查询，VeryGood mAP / Hit@1 分别为 **42.99 / 71.68、43.57 / 74.06、43.76 / 76.84**。本轮仅重排前五候选，未加入记忆或回看；此前 A 阶段未通过门槛的结论保留。

- [QVHighlights 全量实验报告](docs/reze_fullval_results_2026-10-20.md)：完整指标、配对区间、CLIP / VTG-GPT 对照与成本边界。
- [REZE 完整执行方案](docs/reze_execution_plan_2026-10-19.md)、[研究方向](docs/reze_research_directions_2026-10-19.md)：比较、记忆、回看和确认阶段的原始设计，附最新执行状态。
- [结果包与 CPU 复算](comparisons/qvhighlights/reze_fullval_2026-10-20/README.md)：预测、逐查询结果、冻结代码和来源校验。
- [文档索引](docs/README.md)、[此前结果总汇](docs/reproduction_results_summary_2026-10-19.md)：查阅 MERIT、LongVT、VideoARM 历史结果。

## 资料与实现

- 论文：[Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding](https://arxiv.org/abs/2608.07663v2)。[本地 PDF](papers/MERIT_2608.07663v2.pdf)、[来源记录](papers/MERIT_2608.07663v2.metadata.json)。
- [官方源码](https://github.com/choi-yeeun/MERIT)位于 `third_party/merit/`，提交 `6edea6e1ea55af0f6d2e32c46ac170699e955993`；参考其 LVBench 三键方案，未改上游代码。
- `src/merit_qvh/`：数据检查、视频采样、模型调用、三键检索、评分与评估适配器。
- `configs/main_gpu_v1.json`：当前 GPU 主实验配置；`main_v1.json` 保留之前的 CPU 检索配置，两者仅检索设备不同。
- `tests/test_protocol.py`：时间轴、检索、标签隔离、输出验证、失败处理及恢复测试。
- [执行记录](docs/execution_status.md)：已完成检查、当前运行位置和进度。
- [自动执行与恢复](docs/automation.md)：每分钟监控、服务恢复、断点续跑和 systemd 管理命令。
- [历史实验对比数据](comparisons/qvhighlights/README.md)：LONGVT 四组完整验证结果、VideoARM 的 28 个历史运行、预测、请求账本及逐查询对齐表。
- [后续实验方向报告](docs/research_directions_summary_2026-10-15.md)、[研究方案详版](docs/research_directions_2026-10-15.md)：2026-10-16 的计划快照，包含适配差异、初始化审计、MERIT 新评分方案及并行计划；后续状态以结果报告为准。
- [统一结果分析](comparisons/qvhighlights/analysis_2026-10-15/README.md)：MERIT 完整 val、LongVT 对照、VideoARM 同样本子集和时间开销。
- [MERIT 最终预测与报告](comparisons/qvhighlights/merit_main_2026-10-15/README.md)：1,550 条完整 val，VeryGood mAP **21.67**、Hit@1 **31.23**，含 9 条失败回退。

## 原 MERIT 主实验的固定协议

| 项目 | 设置 |
| --- | --- |
| 记忆及评分模型 | `Qwen/Qwen3-VL-8B-Instruct`，BF16、temperature 0 |
| 检索模型 | `Qwen/Qwen3-Embedding-4B`，BF16，无指令前缀，最大序列 512 |
| 记忆 | 每 10 秒一个片段，1 FPS；每秒中点采样，最长边 448 像素 |
| 记忆生成 | 先看图生成英文描述，再仅根据描述提取 event/object/summary 三键 |
| 检索 | 三键余弦相似度取最大；每轮 3 个新锚点，最多 3 轮，邻域 ±1 个记忆片段 |
| 评分 | 全视频描述、检索帧和邻域证据共同评分；每 2 秒一个 0–100 分数，每批 15 个区间 |
| 数据 | 训练集固定 5 个不同视频做工程检查；随后验证集全部 1,550 查询、1,519 视频 |
| 指标 | 官方 VeryGood mAP、Hit@1；同时记录 Fair/Good、失败率、并列分数和调用成本 |

记忆与查询无关，不使用音频、ASR 或对话键。未检索片段仍从描述获得评分。输出长度必须为 `int(duration / 2)`；仅允许一次格式重试。查询失败时保留明确失败状态，整条查询使用零分回退且保留评估分母。零分回退不代表指标一定为零。

推理清单只包含 `qid/vid/query/duration/video_path`；标签保存在独立目录，只由评估函数读取。缓存绑定源码、配置、模型版本和视频哈希；修改代码或配置会建立新运行目录。

## 克隆与数据准备

```bash
git clone --recurse-submodules git@github.com:irryxeek/MERIT-based-method.git
cd MERIT-based-method
# 已克隆但未获取依赖时：
git submodule update --init --recursive
```

本仓库保留原 MERIT 主实验的 `src/`，并补齐后续实验源码。新增发布入口调整了机器路径和部署参数，历史代码快照保留在 `comparisons/`。自动化配置和 systemd 单元提供模板；`AGENTS.md` 原样保留。

- 视频自行准备到 `data/QVHighlights/videos/<vid>.mp4`。
- 官方标注和评估器由 `third_party/moment_detr` 子模块提供。
- 视觉模型放在 `models/Qwen3-VL-8B-Instruct/`；检索模型放在 `models/Qwen3-Embedding-4B/`。
- 运行产物写入 `artifacts/`。视频、权重和完整运行缓存不上传 Git；精选预测、报告和对比记录保存在 `comparisons/`。

可将上述目录链接到数据盘，或修改配置路径。两份主实验配置的模型与数据路径应保持一致，所有命令从仓库根目录执行。路径变化也会改变协议哈希，新机器不能直接接续历史运行。

## 环境与命令

原实验使用 Python 3.10.12、PyTorch 2.9.0、Transformers 4.57.6 和 vLLM 0.12.0，依赖见 [requirements-runtime.txt](requirements-runtime.txt)。新机器需要兼容的 NVIDIA 驱动，可创建独立环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt "vllm==0.12.0"
python -m pip install --no-deps -e .
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir models/Qwen3-VL-8B-Instruct
```

保留 Hugging Face 下载元数据供模型审计。检索下载脚本读取 `configs/main_v1.json`，使用固定版本并核验权重 SHA-256。准备数据和视觉权重后执行：

```bash
export PYTHONPATH="$PWD/src"
MERIT_PY=python
MERIT_CONFIG=configs/main_gpu_v1.json

$MERIT_PY -m pytest -q
$MERIT_PY -m ruff check src tests scripts
$MERIT_PY -m ruff format --check src tests scripts

$MERIT_PY -m merit_qvh.cli prepare --config "$MERIT_CONFIG"
$MERIT_PY -m merit_qvh.cli frames --config "$MERIT_CONFIG" --split smoke
$MERIT_PY -m merit_qvh.cli frames --config "$MERIT_CONFIG" --split val
$MERIT_PY scripts/fetch_embedding.py
$MERIT_PY -m merit_qvh.cli audit-models --config "$MERIT_CONFIG"
$MERIT_PY -m merit_qvh.cli where --config "$MERIT_CONFIG"
```

`prepare` 校验数据并生成无标签清单；`frames` 解码、检查时长并缓存采样帧；`audit-models` 验证模型版本与权重校验和；`where` 打印当前协议对应的输出目录。权重下载未完成时，可用 `audit-models --model-scope vlm` 单独核验视觉模型。

旧辅助脚本 `scripts/prepare_frames.py --workers 8 --split val` 固定读取 CPU 配置；GPU 主实验请使用上方显式传入配置的 `frames` 命令。已核对全部 1,519 个验证视频：其中 13 个尾段比标注时长短 1–2 秒。记忆按实际可见时长截取，评分保留官方 2 秒网格；差异写入媒体审计报告。缺失达到完整 2 秒区间的视频会报错。

原实验使用物理 GPU 4、5，视觉服务端口 8036。以下 GPU 编号仅为示例，启动前替换为实际分配的设备，并确保检索模型有足够显存：

```bash
MERIT_GPU_IDS=4,5 MERIT_VLM_GPU_MEMORY=0.56 \
  MERIT_VLM_MODEL="$PWD/models/Qwen3-VL-8B-Instruct" \
  bash scripts/serve_vlm.sh
```

视觉服务支持每请求 90 张图、32768 上下文、BF16 和两卡张量并行。显存预算为每卡 56%，为物理 GPU 4 上的检索模型留出空间。在另一个终端执行：

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
MERIT_PY=python
MERIT_CONFIG=configs/main_gpu_v1.json
MERIT_ENDPOINT=http://127.0.0.1:8036/v1
$MERIT_PY -m merit_qvh.cli memory --config "$MERIT_CONFIG" \
  --split smoke --endpoint "$MERIT_ENDPOINT"
CUDA_VISIBLE_DEVICES=4 $MERIT_PY scripts/run_main.py \
  --config "$MERIT_CONFIG" --endpoint "$MERIT_ENDPOINT"
$MERIT_PY -m merit_qvh.cli evaluate --config "$MERIT_CONFIG" --split val
```

`run_main.py` 依次检查模型共用显存、运行五视频工程检查、执行全量验证并评估；各阶段使用独立进程，确保模型显存完全释放。进度见运行目录的 `execution_status.json` 和 `val_progress.json`。同一协议有进程锁防止重复运行。`cuda:0` 经上述环境变量映射到物理 GPU 4。

`memory` 子命令仅生成描述与键，每次最多 10 张图。所有调用串行进行。完整验证推理要求五视频工程检查先通过；记忆生成、索引或评分失败均记录整条查询的零分回退，连续三条查询失败会停止并保留日志。相同命令可以恢复已完成工作，失败请求不会在恢复时无限重试。首次启动 `run_main.py` 前必须先生成五视频记忆，上方 `memory` 命令完成此步骤。

`scripts/serve_vlm.sh` 不会停止已有服务。`scripts/check_retrieval.py` 保留原 CPU 配置，可独立检查已生成训练集记忆的真实检索，不读取高光标签。

原机器已启用 `merit-qvh-supervisor.service` 自动监控。新机器需填写 `configs/automation_v1.example.json` 和 `scripts/merit-qvh-supervisor.service.example` 中的路径、GPU UUID 及协议标识，按自动执行文档安装；克隆仓库不会自动启动实验或服务。进程异常退出后通过 `scripts/resume_validation.py` 只处理未完成查询；原有成功结果和明确失败结果均保留。使用 `systemctl --user status merit-qvh-supervisor.service` 查看服务状态，详细恢复规则见[自动执行与恢复](docs/automation.md)。

## 数据与输出

发布版使用以下示例布局，数据只读使用：

- 视频：`data/QVHighlights/videos/`
- 标注和官方评估器：`third_party/moment_detr/`
- 评估器版本：`b7e553ac3b0c898ee6b85e03ee507c064eab89ca`

模型放在 `models/`，生成的大文件放在 `artifacts/`。每次运行目录包含 `protocol.json`、源码快照 `source_snapshot/`、数据/模型审计、`memories/`、原始请求 `requests/`、逐查询轨迹 `queries/`、`*_predictions.jsonl` 和 `*_report.json`。报告区分模型正常输出与失败回退。官方示例预测已单独核对评估器；其分数不是本项目结果。

历史进度文档仅代表其中注明的时间点。2026-10-15 主实验已完成，公开的[最终报告](comparisons/qvhighlights/merit_main_2026-10-15/val_report.json)与[预测](comparisons/qvhighlights/merit_main_2026-10-15/val_predictions.jsonl)保持原始字节；完整请求、记忆和图片缓存未随本次导出。后续实验结果见上方最新报告与文档索引。

## 开发约定

使用 Python 四空格缩进、`snake_case` 函数/变量和 `test_*.py` 测试命名；Ruff 负责格式与静态检查。增加影响推理或评估的改动时补充相应协议测试。当前目录原有 `AGENTS.md` 按用户要求保留不修改，实际命令与布局以本文为准。

## 历史对比归档

对比数据位于 [comparisons/qvhighlights/](comparisons/qvhighlights/README.md)，约 134 MiB。公开副本替换了机器路径与设备标识，并分别保存原始文件和公开文件的校验和；逐查询预测及指标保持一致。原始本地归档独立保留。

LONGVT 包含 compact v1 和 GroundVTS 启发式 v2b 两套协议，各有 Global/Tool 完整验证预测。VideoARM HD 仅有 419 条成功预测，已明确标为部分结果；MR、HD 和开发集指标分别保存，不能混作完整高光检测基线。

## QFVS 优化验证（2026-10-22）

[本轮执行报告](docs/reze_qfvs_adaptation_2026-10-22.md)记录固定四查询的概念提示与时间选择消融。原分数增加时间抑制后 Oracle F1 为 40.41，仍低于同预算 Uniform 50.15；新提示任务已配置自动等待 GPU 6。该结果属于开发诊断，不能代表全量 180 查询。

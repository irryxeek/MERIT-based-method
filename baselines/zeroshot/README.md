# 零样本高光检测基线

本目录补齐原 QVHighlights-ZeroShot 工作区的 CLIP、VTG-GPT、REZE B0、P_abs / P_pair 及跨数据集实现。保持内部 `scripts/`、`experiments/`、`tests/` 布局；机器路径通过根目录的 `configs/paths.example.sh` 配置。

统一入口、依赖安装、数据准备与结果对应关系见[复现索引](../../docs/reproduction_index.md)。在运行任何命令前，从仓库根目录加载路径配置，并在独立环境中安装 `requirements-reze.txt`。

| 目录 | 用途 |
|---|---|
| `scripts/reze_{core,backend,media}.py` | REZE 的评分、模型调用、视频采样 |
| `scripts/prepare_reze_inputs.py` | 新机器权重核验与无标签清单 |
| `scripts/prepare_pair_plan.py` | 将新比较计划绑定到已完成的 B0 |
| `experiments/reze_evidence/`、`experiments/reze_fullval/` | 独立复评、双向比较、固定预算与结果核验 |
| `experiments/reze_{qfvs,mrhisum,tvsum,svhighlights}/` | 各任务的数据网格、评价与恢复 |
| `third_party/CLIP/`、`third_party/VTG-GPT/` | 固定提交的上游子模块 |

CLIP / VTG-GPT 使用 `scripts/prepare.py` 准备作者 caption、query 和数据审计，再用 `run_clip.sh` / `run_vtg.sh` 运行。VTG-GPT 仍复用作者中间结果，没有新增描述模型推理。该准备入口会写入自身清单与 `configs/protocol_v1.json`，应在新工作副本中执行，不用于修改正在运行的 REZE 绑定。

跨数据集 vendor 中的官方评价器和许可证保留。REZE 核心公式、提示词与采样实现保持原方案；部署调整、GPU 寻址修复和验证记录见[代码补齐报告](../../docs/repository_completion_2026-10-22.md)。本目录不附原始视频、权重和完整请求缓存。

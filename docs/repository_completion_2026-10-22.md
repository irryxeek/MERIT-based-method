# GitHub 仓库代码补齐与复现入口整理

日期：2026-10-22。目标仓库：`irryxeek/MERIT-based-method`，分支 `main`。本轮整理代码和文档，没有启动新的 GPU 推理、下载数据或改写已有实验结果。

## 补齐范围

- MERIT：补齐 `experiments/hd_adapter/` 和 `experiments/highlight_evidence/`，包括密集初始化、局部复评、四类 cue 记忆、粒度消融、可靠性修复、纯记忆评分与 R1/R2 读取实验。
- 零样本工作区：在 `baselines/zeroshot/` 保留原 `scripts/`、`experiments/`、`tests/` 结构，补齐 CLIP、VTG-GPT、REZE B0、P_abs / P_pair，以及 QFVS、MR.HiSum、TVSum、SVHighlights 的实现。
- 配套材料：补齐下载、数据审计、结果分析、CPU 测试、配置示例、模型文件校验信息与依赖清单。
- 上游依赖：增加固定提交的 CLIP、VTG-GPT 和 MR.HiSum 子模块。已有 MERIT、moment_detr 子模块不变；各跨数据集评价器的原作者文件和许可证保留。
- 首页与文档：增加[统一复现索引](https://github.com/irryxeek/MERIT-based-method/blob/main/docs/reproduction_index.md)，将方法、数据集、输入条件、代码入口、样本范围和结果报告对应起来。

源码来自两个研究工作区，来源文件与公开副本的 SHA-256 记录在 `docs/code_publication_manifest.json`。机器路径通过环境变量和 JSON 模板展开，原研究目录和运行缓存不随发布迁移。

## 为可运行性补上的内容

1. `configs/paths.example.sh`：统一配置数据、模型、产物、源代码和分配 GPU 的 UUID。机器配置另存为忽略文件。
2. `scripts/materialize_config.py`：将 `${变量}` 展开为新的本机 JSON，缺少路径时失败，拒绝覆盖已有配置。
3. `prepare_reze_inputs.py`：校验固定模型文件，生成完整 1,550 查询的无标签推理清单、五查询工程清单和独立评价文件哈希；提供 `--model-only` 供跨数据集准备使用。
4. QV B0 启动：直接核验原视频 SHA，解除先运行 CLIP、读取其帧缓存的隐式依赖。持续推理工程检查保存源码绑定，正式运行仍要求 21 条工程检查完成。
5. `prepare_pair_plan.py`：把新比较计划绑定到已完成 B0 的原始记录、完成凭证、模型、源码与评估器；保留原 Top-5 比较方案和调用预算。
6. 部署修复：MERIT 根据配置 UUID 查询数字卡号；TVSum 将必要路径环境写入本次运行的 systemd 环境文件；REZE A 阶段显式接收 GPU 参数。
7. SVHighlights：将 UUID 直接传给旧 vLLM 的方式改为数字 CUDA 索引，并在 worker 中核对预期 UUID。CPU 回归覆盖空闲卡选择和环境传递，但没有进行新的真实 GPU 验证。

上述部署调整没有修改 REZE 的评分公式、提示词、原始帧采样及官方指标函数。`reze_core.py`、`reze_media.py` 和各原作者评价器保持原字节；`reze_backend.py` 仅调整模型路径来源与格式；正式 worker 使用冻结核验记录中的模型路径，避免后续环境变量变化切换模型。跨数据集 vendor 的 REZE 核心与当前公共评分器逐文件校验一致。

## CPU 验证

各工作区分别启动测试进程，`CUDA_VISIBLE_DEVICES` 为空，不进行模型前向计算。

| 范围 | 通过 | 跳过 |
|---|---:|---:|
| MERIT 与配置发布工具 | 151 | 0 |
| QV 零样本与 REZE | 57 | 7 |
| QFVS | 18 | 1 |
| MR.HiSum | 23 | 3 |
| TVSum | 23 | 0 |
| SVHighlights | 8 | 1 |
| 合计 | **280** | **12** |

跳过项需要原始媒体、历史 receipts 或模型 tokenizer；没有将这些项视为通过。测试使用已有推理解释器及只读的 h5py 安装，未升级运行环境。新克隆的完整环境安装和实际 GPU 推理仍需按复现索引执行。

另有 25 个 CLI 帮助入口检查通过，覆盖配置工具、准备、推理协调、CPU 评估与 MERIT 模块。Ruff 检查通过；原作者 vendor 文件不作全库格式重写；`.gitattributes` 对导入的许可证与评价代码保留原字节和上游空白。新增回归包括配置拒绝覆盖、路径缺失拒绝写入、无标签输入隔离、可变卡号、冻结模型路径优先及 SVHighlights 数字索引传递。

## 结果与复现边界

- 已有报告、数值、预测和冻结快照保持其历史含义。`comparisons/` 与原 `AGENTS.md` 不修改；既有冻结计划发布凭证保留。
- QV 全量与 TVSum 全量已经完成；MR.HiSum 为可用的 1,888 / 2,000；QFVS 仅四查询；SVHighlights 仍无有效指标。代码补齐不改变这些状态。
- TVHighlights 尚无可用数据与实现，不以 TVSum 代替。
- MERIT 后续读取实验、REZE A 阶段及 SVHighlights 配对协议依赖历史父运行、划分或媒体 receipts。源码发布不等于这些原始材料已全部公开；相关入口保留原检查条件。
- 原始视频、权重、完整记忆 / 请求缓存、私人配置和运行中状态不上传。下载脚本需要相应公开来源与本机清单，不会因安装或克隆而自动下载。

## 后续维护

新开发优先在 Git 管理的发布仓库内进行；原研究工作区保留为历史来源。需要从原工作区同步时，使用发布清单核对来源与公开副本，不以工作树干净代替全工作区覆盖检查。

`scripts/verify_publication.py` 校验代码、报告、索引和文档链接；提供两个研究目录参数后，还会检测新增或变化的源码与报告。每次推送继续包含相关报告和方案，见[发布约定](https://github.com/irryxeek/MERIT-based-method/blob/main/docs/report_publication_workflow.md)。

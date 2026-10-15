# P0 项目与资源审计

任务：VideoARM 在 QVHighlights 上的开源模型迁移评测。始于 2026-10-13，执行跨至 2026-10-14。

本目录和 `open_source_qvh/videoarm/qvhighlights/` 为新建独立实现。没有复制虚拟环境，也没有修改 `upstream/` 或 SVHighlights 实现。应用了实际路径祖先 `/path/to/workspace/AGENTS.md` 中的通用风格与保存要求；其中 LongVT 项目结构、既有 GPU 授权及实验成绩不迁移为本任务授权或成绩。工作区根不是 Git 仓库，`upstream/` 是固定的独立 Git checkout。

- VideoARM 固定源码：`MILVLG/videoarm@af1973ad8ffdb1ae3815c1f4e3f22b8c82858ef6`；现场 `git status` 干净。
- QVHighlights 官方源码与 evaluator：`jayleicn/moment_detr@b7e553ac3b0c898ee6b85e03ee507c064eab89ca`，独立 detached checkout 在 `external/moment_detr/`。未修改 evaluator 数学定义。
- 论文使用 [VideoARM v2](https://arxiv.org/abs/2512.12360v2) 和 [QVHighlights v2](https://arxiv.org/abs/2107.09609v2)，本地 PDF/文本和文件哈希在 `checks/`。
- 模型只读复用 `/path/to/storage/MemDreamer/models/Qwen3-VL-8B-Instruct`；下载 metadata revision 为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。重新计算四个 safetensors 文件 SHA-256，全部与既有记录相符；证据 `checks/model_hash_verification.json`。
- 客户端环境只读复用 `/path/to/storage/longvt-repro/env/inference`；服务环境只读复用 `/data/linyifei/videoarm-repro/svhighlights/tf_selector/qwen35_9b/env-vllm017`。包清单独立保存在 `checks/*environment_20261013.txt`，未安装或升级依赖。
- 客户端原 `pip check` 有既有 `latex2sympy2/ANTLR` 版本冲突与 decord 平台检查失败；本任务不调用这两条路径，PyAV、官方评分和独立测试均实际通过，不声称整个客户端依赖检查通过。服务环境没有 pip/uv，用 `importlib.metadata` 获取 193 个包及依赖关系，未发现声明依赖冲突；vLLM 0.17.0、torch 2.10.0+cu128、transformers 4.57.6。GPU 兼容性仍须真实启动与探针确认。
- 初查 SSD 剩余约 850 GB；原始压缩包约 134 GB 已存在，解压原视频已存在，媒体不复制。模型不量化；默认 BF16、TP=2、单副本、单请求并发。

SVHighlights 暂停记录 `reproduction/svhighlights/INFERENCE_PAUSE_2026-10-13.md` 已读取。既有 full inference/automation 没有被启动；端口 8016/8017 未被使用。本任务服务限定 `videoarm-qvh-*`，模型端口 8026，数据库、缓存与锁均在独立 `QVH_DATA_ROOT`。原保护文件的执行前哈希在 `checks/protected_artifacts_before.json`，执行后核对在 `checks/protected_artifacts_after.json`。

实时 GPU 初查只有 index 6 / `GPU-REDACTED-01` 无计算进程、约 15 MiB 占用。其余设备由其他用户的训练、ASR 或 Qwen2.5-VL 服务占用；即使利用率为零，也未当作可用。没有可复用且身份匹配的 Qwen3-VL-8B 本地服务，没有终止其他用户进程，也没有改成 TP=1。持续资源记录在 `checks/gpu_readiness.json`；只有两张设备同时满足无计算进程且占用不超过 200 MiB，才允许启动 QVH-owned 服务。启动后以 service cgroup 的计算进程核对实际 NVML GPU UUID。

GitHub commits API 初次获取返回 403 rate-limit，原请求错误保留在工具记录；随后直接 Git clone 并固定 HEAD，来源身份已落实，不构成数据或评测阻塞。官方 README 仍有旧 test 说明，但 [评测 README](https://github.com/jayleicn/moment_detr/blob/b7e553ac3b0c898ee6b85e03ee507c064eab89ca/standalone_eval/README.md) 已注明 Codalab 关闭与 test GT 公开；推理只使用 `highlight_test_release.jsonl`。

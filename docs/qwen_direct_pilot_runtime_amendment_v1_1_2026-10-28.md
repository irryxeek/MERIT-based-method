# Direct pilot v1.1：仅修复 GPU 设备映射

> 发布说明（2026-10-30）：本报告随组会 PPT 和总结同步。保留原实验事实、历史状态与预算；机器路径改用别名，未包含的运行材料明确标为本地资料。文中原“不推送”描述属于编写时记录，本次推送依据后续用户授权。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run。
- Origin Date：2026-10-28；Version Label：qwen-direct-pilot30-v1.1。
- Verification Status：UNVERIFIED；本修订不含新实验结果。

## 失败记录

v1 在本地 vLLM 0.12 初始化时，因内部将 `CUDA_VISIBLE_DEVICES` 转为整数而无法接受 GPU UUID，退出码为 1。模型尚未加载，`ledger.json` 的 attempts 为空：**0 模型请求、0 token**。该 worker 的 GPU 生命周期为 **15.08502735500224 秒**，退出收据确认没有所属残留进程。

旧运行目录 `$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_2026-10-28` 及协议、日志、源快照全部保留，不修改其失败状态。

## 修复及预算承接

新版本仅使用当前 `nvidia-smi` 确认的 GPU 数字编号传入 `CUDA_VISIBLE_DEVICES`，设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；预期 UUID 仍独立传入，并在 worker 中通过 PyTorch 核对。其余模型、候选范围、64 帧像素、提示词、输出语法、排序/区间规则与指标保持相同；新版本直接复用冻结输入，不重抽帧，不重选样本。

这是对“完成本轮实验”授权范围内的启动兼容修复，不是模型输出重试。v1 的一次 worker 启动已经失败，本修订记录**另一次且仅一次**兼容修复启动；正式请求仍总计 **30**，额外模型请求及输出重试均 **0**，绝对 token 上限仍 **245,760**。累计 GPU 生命周期仍不超过 **3,600 秒**，新版本最多可用 **3,584.914972644998 秒**；硬时限保留清理余量，不重置历史耗时。

新目录：`$ZEROSHOT_ARTIFACT_ROOT/qwen_direct_pilot30_v1_1_2026-10-28`。保存旧版本退出收据、空请求账本、输入及协议的哈希，新增设备映射 CPU 回归检查。QV 与 TVSum 旧研究账本继续独立，不追加使用。

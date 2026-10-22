# REZE 在 QVHighlights 上的复现协议

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](../report_publication_workflow.md)。

## 范围与来源

目标是用已有 **Qwen3-VL-8B-Instruct**，在完整 QVHighlights **val 1,550 条查询**上运行
REZE 的高光检测主实验，与本目录 CLIP、VTG-GPT 和历史 MERIT 的同集合结果比较。
本轮不训练、不调参，不运行 MR 区间提取器。

方法依据：[REZE: Recognition-Based Zero-Shot Extraction for Video Temporal Grounding](https://arxiv.org/html/2608.04480v1)，
重点采用 Appendix A 的提示词、概率提取与时间轴聚合代码。
截至本次检索，未找到可核验的完整官方代码仓库，因此本轮属于**按论文附录独立实现**。
论文 HTML 与 PDF 已归档到 SSD 的 `papers/reze/2608.04480v1.{html,pdf}`；HTML SHA256 写入配置，
PDF 下载凭证和 SHA256 保存在同目录 `pdf_receipt.json`。
论文 Qwen3-VL-8B 的 test 43.37 / 71.98 不与这里的 val 分数直接相减。

## 固定推理流程

1. 读取原始 query 和视频；推理清单仅含 qid、vid、query、duration、video_path。
   不向模型提供 GT、VTG-GPT 的 caption 或改写 query。
2. 视频从起点划分连续、不重叠的 3 秒窗口，以 2 fps 取帧；完整窗口包含 6 帧。
   丢弃不足 0.3 秒的最后残段。
3. Qwen 通过原生视频输入联合读取窗口内的帧。沿用论文严格 Yes/No 动作核验提示词；
   greedy 解码仅生成 1 token，取完整词表 top-20 log probabilities。
4. Yes 与 No 同时在 top-20 时，计算二者 softmax 得到 P(Yes)。仅 Yes 存在时取其完整词表概率，
   Yes 缺失则取 0，严格保留附录 fallback，并分别统计发生次数。
   没有强制 token 白名单，没有让模型直接生成整数分数。
5. 按附录代码将窗口概率铺到秒级曲线，包含整数右端点，交界处取平均；未覆盖秒最近邻填充。
   再对连续 2 秒求均值，得到完整 HD 分数。**不平滑、不做 Otsu 阈值化、不进行额外归一化。**
6. 全部预测完成后，单独调用固定版本的 QVHighlights 官方评估器，输出 Fair / Good / VeryGood。
   主指标是 VeryGood mAP、HIT@1。

这是一种密集视觉核验基线：每条约 150 秒视频通常产生 50 次短窗口核验。
它没有 memory 建库、事件图、agent 工具调用或按需回看。

## 论文未完全指定的实现细节

以下选择在正式运行前固定，不根据 val 指标择优：

- 使用 Decord 和原始 presentation timestamps；完整窗口在 0、0.5、1、1.5、2、2.5 秒采样。
  取不晚于目标时刻开始的真实帧，不生成插值帧。
- 残段帧数为 `max(2, round(2 * 时长))`；奇数帧由原生 processor 做时间维填充。
- 按 Qwen 工具的 per-frame 范围，`smart_resize` 的 factor 为 32，最少 131,072、最多 786,432 像素；
  bicubic antialias。显式预处理后禁用 processor 的二次缩放与采样。
- 视频片段使用本地相对时间戳。媒体结尾比标注短不足 2 秒的已审计情况，
  只读取真实视频，再按附录最近邻规则映射到完整标注时间轴；不伪造末尾视频帧。
- 本轮原始视频资源均为 300 像素高，1,499/1,519 个视频为 300×534；
  六帧窗口最多产生 513 个视觉 token。完整尺寸审计写入运行目录的 `source_resolution_audit.json`。
  这是本地可用视频的输入质量，不能假定与论文原始视频下载分辨率完全一致。
- 模型 revision 为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、未量化；
  16 个模型文件已按历史审计重新核对大小和 SHA256。
- 本机使用 RTX 4090 24GB、Python 3.10.12、torch 2.9.0、vLLM 0.12.0、Transformers 4.57.6，
  与论文 RTX 6000 Ada 48GB 及历史环境有差异。
- 每张 GPU 独立 TP=1，最大 8 并发序列、4,096 token 上下文；固定 1 GiB KV cache，eager 模式。
  跳过旧版 vLLM 的自动多模态 profiling，并设置 processor 的尺寸估计上限以约束 encoder cache。
  输入已显式预处理且 `do_resize=False`，该估计上限不改变模型输入像素。

完整声明见 `configs/reze_protocol_v2.json`。这些差异限制了“严格历史复现”的结论。

## 工程验证、运行与恢复

已通过真实视频 12 窗口批量推理，并复查其中两个窗口单独推理：P(Yes) 差值均为 0。
核心逻辑、断点完整性与已有基线共 27 项测试通过，源码通过 Ruff；另用固定 5 条完整查询检验读帧、推理、
末尾残段、落盘与官方评估。该 smoke 只用于工程验证，不按其准确率修改方法。

```bash
cd $ZEROSHOT_ROOT
PY=$SHARED_DATA_ROOT/env/inference/bin/python
$PY -m pytest -q
$PY scripts/launch_reze.py --run $ZEROSHOT_ARTIFACT_ROOT/runs/reze_smoke_v2 --smoke --gpus 6
$PY scripts/launch_reze.py --run $ZEROSHOT_ARTIFACT_ROOT/runs/reze_val_v2 --gpus 4 5 6 7
$PY scripts/summarize_reze.py --run $ZEROSHOT_ARTIFACT_ROOT/runs/reze_val_v2
```

启动前核对 GPU 空闲状态。按 manifest 顺序对 worker 数取模分片；每条查询原子保存，
重启核对源码、配置、视频字节与已完成预测，仅处理剩余样本。
单 worker 最多自动重启 2 次；15 分钟无进度判为卡住，记录后终止本任务的该 worker。
失败不静默跳过，不补零冒充模型结果。systemd 后台执行可脱离终端；机器断电后需重新启动同一命令恢复。

SSD 的 `runs/reze_val_v2/` 保存源代码快照、绑定摘要、逐查询输入帧索引/时间戳/像素哈希、
top-20 原始 logprobs、概率曲线、运行日志和官方评估。
小型归档及比较结果写入本目录 `results/reze_val_v2/`、`results/reze_comparison.json`，
完成报告为 `docs/REZE_RESULTS_2026-10-19.md`。旧基线结果保持独立。

当前全量服务为 `qvh-reze-val-20261019-v3.service`（产物版本 v2）。
systemd 使用 `Type=oneshot`，待推理和评估退出后再执行汇总脚本。
查询状态可读取 `runs/reze_val_v2/progress.json`；日志命令为：

```bash
journalctl --user -u qvh-reze-val-20261019-v3.service --no-pager
```

worker 日志位于 `workers/<编号>/attempt_<次数>.log`。
`gpu_telemetry.csv` 记录后段每 10 秒的显存、利用率与功率，采样从启动约 2 分钟后开始，
因此不能用它声称完整运行的绝对峰值或总能耗。

### 显存故障与协议修订

v1 通过 5 条 smoke 后，在连续处理约 5–10 条查询时发生 CUDA OOM；失败预测和代码快照保留在
`runs/reze_val_v1/`。本地源码表明，encoder cache 在达到容量前保留已完成请求的视觉特征；
Qwen3-VL 的默认容量估算为 151,250 tokens，远大于这里的单个短窗口需求。
该行为也有[vLLM 上游报告](https://github.com/vllm-project/vllm/issues/27992)。

v2 仅修改 `mm_processor_kwargs.size` 的估计范围：shortest_edge=131072，longest_edge=4718592，
由本机注册表核验 encoder 估计降为 51,200 tokens。实际输入帧仍按原配置提前处理，
processor 不做二次缩放；CPU tensor 逐值比对相同。
新增 `check_reze_sustained.py` 连续执行 21 条真实查询，跨越旧故障边界，记录显存和修复前后概率。
该检查已完成 1,035 个窗口：显存从 20,787 MiB 升至 21,603 MiB 后保持稳定。
修复前后 1,035 对窗口中，51 对概率差异大于 1e-5；平均绝对差 0.000713，最大 0.215123。
输入像素及 prompt token 数一致，但缓存淘汰会改变 BF16 批处理调度，不能承诺逐位复现概率。
这项对照记录数值敏感性，不用于按指标挑选版本；详细数据在 `runs/reze_sustained_v3/`。
正式 v2 从全部查询重新运行，不混合失败 v1 的局部预测；未改变打分方式以迎合验证结果。

## 比较与耗时口径

同一批 val、同一组标签和官方评价指标允许比较 HD 效果；输入模型和预算并不相同。
REZE 报告从原视频解码到预测、评估的墙钟时间，同时记录推理时间之和与输入/输出 token。
四卡并行墙钟时间不能等同于单卡 GPU 时间；VTG-GPT 已有 96.95 秒仅包含复用作者描述后的计算。
历史 MERIT 的 9 条失败仍在其全量分母中，不删去以改善比较结果。
差值区间按源视频聚类做 10,000 次配对 bootstrap（seed 42），作为探索性统计，不证明论文分数等价。

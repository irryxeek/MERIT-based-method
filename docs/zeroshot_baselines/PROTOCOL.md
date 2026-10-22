# CLIP 与 VTG-GPT：复现实验协议

> 发布说明（2026-10-22）：保留原报告的日期、指标和结论；调整引用链接与机器路径。历史命令和本地资料不表示本仓库附带完整运行环境，详见[报告发布约定](../report_publication_workflow.md)。

冻结日期：2026-10-19。主实验只跑完整 QVHighlights validation，不训练、不按验证结果调参。
执行目录为 `$ZEROSHOT_ROOT`，与 MERIT 原实验隔离。

## 1. 数据和指标

- 1,550 条查询、1,519 个视频片段；沿用此前 MERIT 完整 val 身份集合。
- 每个 2s clip 输出一个有限实数。推理清单仅包含 query、视频身份、时长和所需数据路径。
  VTG-GPT 另读取作者发布的 query 改写；不向模型提供 relevant windows、saliency scores 或相关 clip ID。
- 主指标：官方 `HL-min-VeryGood-mAP / Hit1`，另保存 Fair / Good。
  AP 按三名标注者分别计算后平均；Hit@1 对三名标注者取最大命中后平均。
- 标签 SHA256：`f668a1eaea156ec5315e14718999cea043a8cf948d3cafbd8e8d655318c3cd02`。
  官方评估器文件哈希记录于 `manifests/data_audit.json`，每次评估验证。
- 全集严格核对 qid、vid、原始 query、分数长度；不存在成功子集筛选。
  遇到解码、推理或非有限分数错误停止并保留现场，不用常数填充失败查询。

## 2. CLIP 中心帧基线

按照 [QVHighlights 论文 §5.2](https://arxiv.org/html/2107.09609) 的中心帧与 query 相似度方案：

1. 每个 2s clip 取 1、3、5… 秒时显示的原视频帧。依据实际解码时间戳定位，不以平均 FPS 假定代替。
2. 仅当末尾不完整 clip 的中心超出实际视频时，使用仍位于该 clip 的最后有效帧；逐视频记录。
   整个 clip 缺失会报错。此规则先于正式评估冻结。
3. 原版 CLIP 预处理：BICUBIC Resize、224 CenterCrop、RGB、官方归一化。
4. OpenAI ViT-B/32，图像和文本批量 64；原始英文 query，不加模板、不改写，不静默截断。
5. CUDA 原生混合 FP16 权重编码，特征转 FP32 后单位归一化，图文余弦相似度直接作为 HD 分数。
   不做平滑、生成式评分、候选重排或监督校准。

官方源码提交 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`；
权重 SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。
四个 CPU 解码线程、每视频一个 Decord 线程、一张 RTX 4090 推理。
五条 smoke 核对批量与单图特征的数值一致性（FP16 容差 2e-3）。
`trace.json` 保留源视频、帧索引、时间戳、采样 RGB 和特征哈希。

## 3. VTG-GPT 作者中间数据评分复现

使用 [官方仓库](https://github.com/YoucanBaby/VTG-GPT) 提交
`9d6606140edb36c89ea5ce96a0dc5e50361d108a` 的原始 `VTG_GPT.locate_span`：

1. 读取官方 `val.zip` 的逐 clip caption，以及官方 query 文件中的五个改写。
2. 使用 `sentence-transformers/paraphrase-distilroberta-base-v2`，revision
   `1972a507973d92e15f4f7cd9457d9fcf43ea117f`，FP32、eager attention。
3. 原 query 和五个改写各自产生一条与 caption 的余弦相似度曲线。
   按原代码逐曲线 min-max 归一化，再对六条曲线平均，形成 HD 分数。
4. 保留原实现的候选生成，参数 bins=10、top_k=8、gap=6、NMS=0.95；主实验仅评 HD。
   候选后处理不改变上述 HD 分数。

缓存仅复用完全一致的 Sentence-BERT 编码调用；五条 smoke 与未缓存原实现的 HD 分数、
MR 候选逐值一致。此次没有替换 caption 模型，也没有重新生成 caption 或 query 改写。
13 条 query 的作者 caption 比完整时间轴少末尾一格：保留 raw 输出并显式补零一格，
另核对补零前后官方 HD 结果完全一致；没有插值或重新生成。

**论文与仓库差异：**[2024 论文表 4](https://arxiv.org/html/2403.02076v1#S4.T4)
报告使用改写后 val 为 36.08 / 62.29，并描述平均五个改写的相似度；当前公开代码包含原 query、
逐曲线归一化后平均六条。本轮冻结并复现公开代码，未为追逐论文数值修改聚合方式。
公开仓库后续也已更新，不能仅凭结果差距确定具体原因。

## 4. 结果、成本与核验

每轮保存代码/配置/输入绑定哈希、模型和环境信息、逐查询预测、完成凭据、官方指标。
VTG-GPT 首次 smoke 的 NumPy 时间戳 JSON 序列化失败已单独保留；修复后重新通过 smoke，
未改动评分公式。它不构成全量集样本排除。

墙钟按各推理进程记录：CLIP 包括读视频、预处理和编码；VTG-GPT 仅包括加载已生成的文字及后续评分。
不将二者比值解释为端到端加速比。下载、软件准备和人工检查单列为未计入部分。

最终汇总对两组结果及历史 MERIT 全集逐查询配对，核对其均值与官方数值一致。
探索性区间按源 YouTube 视频聚类、seed=42、10,000 次配对 bootstrap，保持 query 加权；
区间不包含算法、标注和实现的系统误差，也未做多重比较校正。
旧 MERIT 的失败输出仍保留于完整分母，不将历史 train20/100 或 VideoARM 成功子集混排。

# M1 范围与推理效率修订（2026-10-14）

用户要求“先只做 M1 方法实验，跟论文原始方法数据做一下对比，然后尽量提升一下推理效率”。后续 smoke8/dev50、完整 val/test 只启动 MR/HD 的 M1，低预算实验暂缓。正式计划从 15,468 次变为 6,184 次样本×方法推理。原 B0/B1/A1 的源码、配置、状态、预算、预测与分数保留；被暂停的 B1 不继续执行。

模型、v2 提示、采样、图像尺寸、JPEG 编码、每 query 动作顺序及预算没有修改。执行改造为有限跨 query 并发：每条 query 使用独立 SQLite 连接、HTTP session、视频 reader 和 checkpoint；在每条 query 内仍顺序执行 Controller→工具→记忆→终答。SQLite WAL 与 30 秒 busy timeout 保护并行写入，主线程串行导出按 qid 排序的预测。SIGTERM 在请求边界暂停，正在执行的请求先落盘；异常结束仍保留已预约的预算。

像素缓存按单个 reader 的真实 PTS 保存已经编码的 JPEG，最多 256 帧。重复采样沿用相同 PTS、同样像素、相同 JPEG 字节，位置与 clip ID 按当前请求重建；不增加观察、不引入跨 query 语义缓存。缓存回归测试检查字节与 PTS 不变。

服务保持 BF16、TP=2、同样 32768 上下文和最大 24 张图。max_num_seqs=4 只增加请求调度容量。configs/execution_m1.json 单独保存执行并发；正式冻结同时保存执行配置及 benchmark 散列。批处理可能影响数值和原始输出，不宣称逐位确定性。

并发测量先按原 smoke8 标识及步骤固定 12 个真实请求：6 个 MR、6 个 HD 初始化。相同历史像素、提示、seed、temperature、输出上限按 1/4/2/2/4/1 的顺序重放，共 72 次实际请求，原始输出和用量全部保存在 qvh_data/probes/concurrency_m1_v1/。根据吞吐和接口稳定性选并发，不读取 GT 或按分数挑选。报告原始输出一致率；重放测量只能证明请求吞吐，不能代替自主 M1 的端到端质量和耗时。

checks/concurrency_benchmark_m1.json 保存实际提速。完成 dev50 后，checks/full_run_estimate.json 使用 runner 真实墙钟时间估算正式进度，并加回开发缓存命中的初始化成本。并行 query 的延迟和不能当作 GPU 小时，成本统计使用 runner 墙钟时间；服务生命周期的启动/空闲成本仍单列。

已终态的 v2 MR smoke8 作为同语义协议的开发证据保留，不修改预算或重置失败。新 dev50 和 HD smoke8 使用当前执行源码，后续正式代码一次冻结；完整 val/test 不按成绩调参。此次是用户修改范围与执行效率修订，不是第二次质量提示/预算调优。

论文数据及不可直接比较之处见 PAPER_COMPARISON.md。机器记录见 checks/m1_scope_revision.json，改造前源码与数据库快照见该文件的 prior_code_archive。

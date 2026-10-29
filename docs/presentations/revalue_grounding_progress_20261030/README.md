# ReVALUE Grounding 组会后实验进展

汇报范围：2026 年 10 月 28–30 日，实验结果截至 10 月 29 日，承接 10 月 27 日组会。

## 文件入口

- [12 页可编辑 PowerPoint](../ReVALUE_Grounding实验进展_2026-10-28至10-30.pptx)
- [12 页 PDF 预览](../ReVALUE_Grounding实验进展_2026-10-28至10-30.pdf)
- [配套工作总结](../../progress_after_meeting_2026-10-27_to_2026-10-30.md)
- [完整逐页讲稿](speaker_notes.md)，同时已嵌入 PPT。
- [可编辑 SVG 页面](svg_output/)、[原帧素材](images/)、[设计说明](design_spec.md)、[设计锁](spec_lock.md)。

## 汇报内容

Direct 整表退化、时间键与单目标诊断、Grounding 的定义与完整分母成绩、52 条反例初筛、三组原帧案例、未解决问题和下一步。

新增 100 条不是全部完成：96 次尝试、94 条合法响应，1 OOM、1 非法区间、4 条未尝试；定位指标保留完整 100 条分母。52 条分类来自助手初筛，尚无独立人工复核，也没有据此修改 GT 或证明 Agent 修复有效。

文字和图形可编辑，含 2 个原生表格及 1 个原生统计图。时间区间图使用可编辑形状；证据卡为完整原帧图片，时间标签未裁剪。

## 验证与版本

- 12/12 SVG 通过最终检查，0 错误、0 警告。
- 12/12 页面有讲稿；导出后检查通过，0 warning categories。
- PPT 包结构核验为 12 页、12 页讲稿、1 个原生 chart、2 个原生 table。
- [图表坐标核验](validation/chart_verification.md)通过；PDF 有 12 页。
- PPT SHA-256：`bb9b3f8fb14949a2bb40ff11498c07a47a787b3062510c0ac6a20c7ad9b0379e`，与研究工作区交付文件逐字节一致。

本次推送依据用户后续授权；未新运行实验。源材料与发布副本哈希见[报告发布清单](../../report_publication_manifest.json)。实时预览服务、机器状态、重复备份和运行缓存未随汇报上传。

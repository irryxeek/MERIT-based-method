<!-- ppt-master-schema: design-spec/v1 -->
# QV Grounding Counterexample qid448 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | qv_grounding_counterexample_qid448_20261030 |
| Canvas Format | ppt169, 1280 × 720 |
| Page Count | 1 |
| Primary Language | zh-CN |
| Target Audience | 导师与研究组成员 |
| Communication Intent | 用一个可追溯的定位反例解释当前 Direct 的错误，并区分已有证据与待验证的修正思路。 |
| Desired Audience Outcome | 听众能指出标注与预测的场景差异，理解因果机制仍未确定，并理解下一步条件核验实验的目标。 |
| Core Message / Ask / Action | qid448 的目标已出现在原输入帧中，但预测落到不同场景；建议检验事件条件核验能否修正定位。 |
| Delivery Context | 组会现场口述约一分钟，单页插入现有汇报。 |
| Artifact Afterlife | 可编辑的一页 PPT 与讲稿，供组会复用及会后查阅。 |
| Reading Mode | balanced |
| Content Strategy | 压缩刚才已核对的案例介绍，保留数值、证据与初筛限制。 |
| Design Style | 原帧证据与审慎判断并排，延续原汇报的配色。 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — workflow default，采用刚才整理的口述内容。 |
| Custom Animations | disabled — final Stage-2 proactive policy。 |
| Narration Audio | disabled — workflow default。 |
| Created Date | 2026-10-30 |

设计确认沿用用户此前“设计细节由你决定，直接完成”的授权；本轮明确要求一页。选择 free_design，沿用已有视觉身份，不使用模板。没有新增实验授权。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 40 px |
| Content Area | 1200 × 640 px；来源与限定说明可进入底部注释区域。 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode References**: instructional
- **Mode Behavior**: 按查询、结果对照、视觉证据、问题判断、待验证修正的顺序建立理解；说明进入讲稿。
- **Visual style**: custom
- **Visual Style References**: swiss-minimal
- **Visual Style Behavior**: 白底与开放分栏、精确对齐、直角和细线；大幅原帧对照承担证据，右侧文字保持简洁；青绿标记上下文，红色标记错误预测。
- **Theme**: 复用已有 ReVALUE 汇报的字体与语义颜色，原帧保留原色及全部时间标签。
- **Tone**: 克制、可追溯，不将单例初筛写成机制证明。

三个方案先由页面任务确定：A 是左侧完整证据、右侧诊断与待验证步骤；B 是时间区间引导、下部完整证据，强调时间错位；C 是原帧占主导、少量旁注，强调场景差异。选择 A，能在一页内兼顾证据、判断和后续实验。A 以 instructional 和 swiss-minimal 为基础；B 以 instructional 和 editorial 为基础，宋体标题与微软雅黑正文形成对比；C 以 instructional 和 photo-editorial 为基础，宋体标题与微软雅黑正文配合大图。三者的图像策略均为 custom：只采用原始证据卡，保留照片细节、真实光照和时间戳，页面标注独立可编辑；整体如实验审阅记录，不生成替代画面。

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 页面背景 |
| Secondary background | #F2F5F8 | 待验证步骤区域 |
| Primary | #17324D | 标题 |
| Accent | #237F8C | 章节提示 |
| Secondary accent | #C76128 | 待验证步骤 |
| Body text | #17324D | 正文 |
| Secondary text | #52667A | 来源与注释 |
| Divider | #D5DEE8 | 分隔 |
| Positive | #16834E | 标注区间 |
| Negative | #B74343 | 预测区间与零重叠 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 清晰无衬线，粗体 | Microsoft YaHei | Arial | sans-serif |
| Body | 清晰无衬线，常规 | Microsoft YaHei | Arial | sans-serif |

- **Title stack**: Microsoft YaHei, Arial, sans-serif
- **Body stack**: Microsoft YaHei, Arial, sans-serif

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 24 |
| Title | 42 |
| Subtitle | 30 |
| Lead | 28 |
| Annotation | 18 |
| Footnote | 16 |
| Hero | 60 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 标题点出错位，查询与设置定语境，原帧与区间对照支持判断，右侧下部承接待验证方案。
- **Composition tendency**: 左侧完整原帧证据占主导，右侧数值和解释，来源在底部。
- **Cross-page continuity**: 单页沿用已有汇报的字体及 GT 绿／预测红含义。
- **Spacing posture**: 开放分栏，留足照片与文字间距。
- **Spacing anchors**: page margin 40 px; block gap 24 px; column gutter 32 px; corner radius 0 px; body leading 34 px。

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qv_448.jpg | 1140 × 680 | 1.6764705882 | 对照原输入帧中鸡舍与预测位置的道路及花朵 | Evidence | 完整证据卡与右侧诊断并排 | no-crop | user | Existing | 已有 qid448 证据卡，顶行为标注区间，底行为预测区间，保留全部帧号与 PTS；原像素不变。 | embedded | local |

## IX. Content Outline

### Part 1: Grounding Counterexample

#### Slide 01 - qid448 定位反例

- **Audience move**: 从抽象低 IoU 结果转为看懂一个有视觉支持的事件定位错位，并分清改进假设与已验证结果。
- **Relationships**: 原输入中的鸡舍证据与输出区间的道路花景形成对照；查询约束、预测核验、必要时重新定位构成待验证的先后关系。
- **Composition**: 大幅原帧证据与区间结果并列，诊断之后接待验证步骤。
- **Title**: Grounding 反例：把鸡舍定位到了花景
- **Core message**: 目标证据已在输入中，预测却落到另一场景；错误来源仍待分辨。
- **Content**:
  - qid448；查询“鸡在鸡舍里”（The chickens are in a coop.）。
  - Direct：Qwen3-VL-8B-Instruct；整视频均匀 64 帧与真实 PTS；一次输出起止时间。
  - 标注 92–112 s；预测 130–145 s；IoU = 0。
  - 标注段采样帧可见围栏内的鸡；预测段采样帧主要为道路和花朵。
  - 助手初筛认为属于较明确的事件定位错误；识别与时间关联的具体原因尚未确定。
  - 下一步验证（未开展）：核验“鸡 + 围栏内”条件；证据不足时重新定位；检查修正率与误改率。
  - 限定：原 64 帧助手初筛，尚无独立人工复核或修正收益；原结果与 GT 未改动。
- **Images**: qv_448.jpg 完整显示，不裁掉边缘文字或时间戳。
- **Cover impact**: 以“鸡舍定位到了花景”的具体错位作为唯一钩子。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under `notes/`
- **Content**: 依据刚才口述稿与 sources 中的报告说明实验设置、原帧证据及尚未验证的方案；不声称已执行修正实验。
- **Total duration**: 约 60–90 秒
- **Notes style**: 简单、自然的中文组会讲述
- **Presentation purpose**: 用一个定位反例解释当前问题，并为后续条件核验实验提供可审查的动机。

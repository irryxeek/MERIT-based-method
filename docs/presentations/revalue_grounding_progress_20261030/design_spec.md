<!-- ppt-master-schema: design-spec/v1 -->
# ReVALUE Grounding Progress 20261030 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | revalue_grounding_progress_20261030 |
| Canvas Format | ppt169, 1280 × 720 |
| Page Count | 12 |
| Primary Language | zh-CN |
| Target Audience | 导师与研究组成员，已了解ReVALUE和10月27日组会背景 |
| Communication Intent | 汇报组会后Direct测试与grounding反例诊断，说明证据和局限，再对齐下一步最小实验方向 |
| Desired Audience Outcome | 区分已完成的基线、机制线索、标注疑点和未验证的Agent修复，并理解应先核验反例再做局部补证 |
| Core Message / Ask / Action | 已获得可追溯的定位反例；下一步以明确错误检验事件与边界补证，Agent收益尚未验证 |
| Delivery Context | 主讲人带讲的约10分钟中文组会；会后可独立查阅和编辑 |
| Artifact Afterlife | 项目归档、组会讨论与后续实验依据 |
| Reading Mode | balanced |
| Content Strategy | 依照用户已整理总结重组为结论、诊断、结果、案例、下一步；所有数据和限制保持原含义 |
| Design Style | 证据推进：深蓝标题、青绿进展、橙色未决，白底与原始帧证据交替 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — final Stage-2 proactive policy under persistent user design delegation |
| Custom Animations | disabled — final Stage-2 proactive policy |
| Narration Audio | disabled — final Stage-2 proactive policy |
| Created Date | 2026-10-30 |

Confirmation provenance: 用户此前明确“按推荐目标，设计细节由你决定，直接完成”，本次要求依据已整理内容制作PPT；沿用其授权，由助手完成两阶段决策。Free design，无模板安装，无伪造UI提交。明确交付可编辑原生图表和表格，开启导出选项。

Three solution intents, authored before catalog selection:

1. **证据推进（selected）**：先报阶段边界，再以输出诊断、grounding结果和三类案例解释下一步。Mode custom基于pyramid，标题直接表达可支持的判断。Visual style custom基于swiss-minimal，精确网格、较大留白、少量开放色面与时间区间端点意象。配色白底、深蓝#17324D、青绿#237F8C、橙#C76128；标题与正文Microsoft YaHei/Arial同族，以字重和尺寸形成层级，正文24px。Icons none。Image strategy custom：真实帧的纪录性对照，边缘清楚，保留原色、平面排列，无新增纹理或景深，像实验记录册。仅使用provided图，既有标签和像素完整保留。
2. **案例先行**：用鸡舍错位开启问题，再展示漏标疑点，最后揭示整体结果与机制诊断。Mode custom基于narrative，形成案例问题到统计解释的顺序。Visual style custom基于photo-editorial，以大图双栏、轻量题注、照片与短结论交替。配色#FAF7F1、#23313D、#237F8C、#A15C32；标题SimSun/Cambria，正文Microsoft YaHei/Arial，正文24px，形成衬线与无衬线对比。Icons none。Image strategy custom：保留纪录照片的自然颗粒和曝光，克制边框、不改场景，像学术影像随笔；使用相同provided证据，不生成画面。
3. **研究审阅手册**：按任务、执行状态、指标、失败、后续条件分类，便于会后查找。Mode custom基于briefing，事实平行呈现。Visual style custom基于data-journalism，以表格主导、多列数据、边注与细分隔线实现较高信息密度。配色#FFFFFF、#263746、#237F8C、#B74343；标题SimSun/Cambria，正文Microsoft YaHei/Arial，正文24px。Icons none。Image strategy custom：证据图按报告图版平面排布，保持细节与刻度，像研究附录；使用provided素材，无AI图。

Selection reason: 证据推进最适合10分钟汇报并延续此前深蓝、青绿的组会识别，同时能清楚分开已完成结果与下一步设想。页数12，计划范围10–12页。完整设计规范可供事后审阅，本次不中断等待新增确认。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | ppt169 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 48px |
| Content Area | x=48..1232, y=48..660；页码与来源独立保留 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode References**: pyramid
- **Mode Behavior**: 每页标题先说当前证据支持的结论，再给必要数字或案例；不把开发样本、分差或初筛解释成总体精度提升。最后把可见错误映射到待验证的修复方向。
- **Visual style**: custom
- **Visual Style References**: swiss-minimal
- **Visual Style Behavior**: 精确对齐、白底、大字重对比、较大的开放色面和细分隔线；研究证据图保留原色，不用装饰性背景干扰判断。原生文字和表格保持可编辑。
- **Theme**: 延续上次深蓝与青绿；时间区间的两端和可见证据对应关系作为跨页意象；案例页共享对照位置以便比较。
- **Tone**: 清楚、克制，明确完成与未决。

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 常规页面底色 |
| Secondary background | #F2F5F8 | 注释、轻量背景区 |
| Primary | #17324D | 标题、主结构、关键数字 |
| Accent | #237F8C | 已完成诊断与方向 |
| Secondary accent | #C76128 | 未决问题、混合案例 |
| Body text | #17324D | 正文 |
| Secondary text | #52667A | 来源、解释与边注 |
| Divider | #D5DEE8 | 分隔线 |
| Surface | #FFFFFF | 图表表面 |
| Grid | #E7EDF3 | 数值刻度线 |
| Positive | #16834E | 发布GT区间，延续证据图编码 |
| Negative | #B74343 | 模型预测区间与错误提示，延续证据图编码 |
| Unknown | #7A8793 | 未决类别 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 精确的粗体无衬线 | Microsoft YaHei | Arial | sans-serif |
| Body | 易读的中性无衬线 | Microsoft YaHei | Arial | sans-serif |
| Code | 短输出契约和数组 | Microsoft YaHei | Consolas | monospace |

- **Title stack**: Microsoft YaHei, Arial, sans-serif
- **Body stack**: Microsoft YaHei, Arial, sans-serif
- **Code stack**: Microsoft YaHei, Consolas, monospace
- **Role rationale**: Code用于数组、时间键及start/end契约，区别于解释文字；其他角色统一正文或标题家族。

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 24 |
| Title | 42 |
| Subtitle | 30 |
| Annotation | 18 |
| Lead | 28 |
| Footnote | 16 |
| Display | 112 |
| Hero | 64 |
| Number | 32 |
| Code | 22 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 判断标题、主要证据、必要限制、来源；一页一个核心判断。
- **Composition tendency**: 机制页用开放序列和并列对照；数据页用表格或条形图；案例页以原帧证据为主、旁边给判断和限制。
- **Cross-page continuity**: 案例P08–P10有意保持同一GT／预测比较逻辑，分别回答真错误、标签疑点与两者并存；统计页与结束页降低图片负担。
- **Spacing posture**: anchor页留白大，dense页用精确对齐保持可读，breathing页突出一个机制判断。
- **Spacing anchors**: page margin=48; block gap=24; column gutter=32; corner radius=4; body leading=34.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P04 | table | comparison_matrix | 三种输出条件、规模和现象 |
| P05 | chart | gantt_chart | 麦片案例的预测与GT时间范围 |
| P06 | table | metric_table | 两批开发样本的执行状态和R1结果 |
| P07 | chart | horizontal_bar_chart | 52条反例的四类初筛数量 |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qv_448.jpg | 1140×680 | 1.67647 | P08鸡舍与花景的真实输入帧对照 | Evidence | 大幅证据图配右侧中文判断，完整保留时间标识 | no-crop | user | Existing | 项目52条审核中的原输入帧拼图；没有重新采样或生成事实 | none | local |
| qv_3676.jpg | 1140×680 | 1.67647 | P09圣诞树前后重复活动 | Evidence | 同样的GT与预测两行比较，突出事件内容相关 | no-crop | user | Existing | 项目52条审核中的原输入帧拼图；标签疑点并非人工确认 | none | local |
| qv_5466.jpg | 1140×680 | 1.67647 | P10编辑软件边界与标签混合问题 | Evidence | 原图与时间范围文字并列，两项独立判断 | no-crop | user | Existing | 项目52条审核中的原输入帧拼图；保留全部标签和原像素 | none | local |

## IX. Content Outline

### Part 1: 已完成什么

#### Slide 01 - Cover

- **Audience move**: 从上次模块讨论进入本阶段Direct与grounding诊断。
- **Relationships**: Direct基线、定位测试与反例核查有推进顺序；52条来自新增100样本的指标失败。
- **Composition**: 一侧完整题目与日期，另一侧大数字52；时间区间意象连接主线。
- **Title**: 从Direct基线到可核查的定位反例
- **Core message**: 组会后完成基线与诊断，下一步检验定位补证。
- **Content**:
  - ReVALUE｜组会后实验进展；林逸飞；2026.10.28–10.30，结果截至10.29。
  - 52条扩样指标反例完成初筛；22条较明确定位错误。
  - Direct → Grounding → 反例核查。
- **Cover impact**: binding hook=52条指标反例已有逐例证据，下一步有具体问题可检验；宏观构图保持开放。

#### Slide 02 - Progress boundary

- **Audience move**: 明确本阶段完成环节和未验证环节。
- **Relationships**: 整表评分、单目标诊断、直接定位、反例核查按工作顺序推进；Agent补证是后续而非完成项。
- **Composition**: 贯穿的工作序列，上方简短状态，下方一行边界。
- **Title**: 已完成基线与诊断，Agent修复尚待验证
- **Core message**: 工作重点已落在grounding错误的可解释诊断。
- **Content**:
  - 10.28：Direct整表与时间键诊断；10.29：单目标判断、区间定位与扩样审核。
  - 先找可解释错误，再设计补证与回看。
  - 记忆、比较与回看尚无本阶段修复收益；上次跨数据集基线不重复计入。

#### Slide 03 - Constant scores

- **Audience move**: 理解合法输出不等于有效的时间判别。
- **Relationships**: 全视频64帧和query输入模型后生成2秒评分整表；25条有效响应内部全部同分。
- **Composition**: 大数字25/25与一段可见常数数组并列，下方说明范围和原因边界。
- **Title**: 整表输出把视频内所有片段打成同分
- **Core message**: 当前整表设置退化，不能直接推导模型缺乏时间理解。
- **Content**:
  - Qwen3-VL-8B-Instruct；全局64帧＋真实PTS；一次输出0–100分数数组。
  - QV20/20、TVSum5/10；25条有效响应各视频内同分。TVSum第6条OOM，后续未补跑。
  - 示例数组为80,80,80,…，仅展示已观察的常数形式。
  - 原始响应即常数，不是后处理广播；QV20 mAP10.71／Hit@1 15.00。

#### Slide 04 - Output diagnosis

- **Audience move**: 区分时间键负结果与单目标机制线索。
- **Relationships**: 三种条件共享QV20、原64帧与模型；变化集中在时间表示和输出目标范围。
- **Composition**: 三行条件对照表，表下独立强调稀疏读数的限制。
- **Title**: 时间键无效；单目标判断出现部分分差
- **Core message**: 11/20只是两个目标的区分比例，不是准确率。
- **Content**:
  - 原整表：20次，20/20视频内同分。
  - 显式时间键：20次，仍20/20同分，mAP／Hit@1不变。
  - 单目标：40次、每视频两个固定2秒目标，11/20对异值、9/20同分。
  - 只判断40个时间格，未计算新全网格精度；输出长度与注意力聚焦等也随条件变化。
- **Visualization**: output-conditions是三条件比较表。
- **Native-ready**: output-conditions=yes

### Part 2: Grounding结果与反例

#### Slide 05 - Grounding task

- **Audience move**: 明确现在要检验目标时间位置和边界。
- **Relationships**: 视频、query进入Direct；模型输出单连续区间或弃答；预测与任一GT取最大IoU。
- **Composition**: 输入到输出的主流程，下方一个真实时间范围示例。
- **Title**: 直接输出时间区间，检验目标在哪里
- **Core message**: 新主线是事件定位；R1与旧高光mAP分开。
- **Content**:
  - 输入64帧、真实PTS和query；一次Qwen调用；输出[start,end]、not_found或uncertain。
  - 不含候选重排、记忆与回看；支持弃答但本轮正查询未验证负query拒答能力。
  - qid5073准备麦片：预测52–67秒，GT48–64秒，IoU0.6316；旧整表全0。
  - 任务与输出均改变，不能当严格单变量精度提升。
- **Visualization**: cereal-intervals将GT48–64与预测52–67置于0–150秒公共时间轴。
- **Native-ready**: cereal-intervals=no

#### Slide 06 - Grounding results

- **Audience move**: 看到完整分母结果，知道扩样没有全部推理完成。
- **Relationships**: 首轮20与新增100是不同开发样本；执行状态与定位精度分别列出。
- **Composition**: 两行结果表为主要证据，三项停止信息独立成行。
- **Title**: 扩样获得52条反例，100条未全部完成
- **Core message**: 新100全分母R1为42%／28%，保留全部失败与未尝试。
- **Content**:
  - 首轮QV20：20/20合法；R1@0.5=70%，R1@0.7=50%，6条IoU<0.5。
  - 新增100：96次尝试，94合法；R1@0.5=42%，R1@0.7=28%，52条合法反例。
  - 1 OOM、1非法区间、4未尝试；技术缺失计零，分母100。
  - 新100与旧20的query及源视频不重叠；均为train开发诊断，非官方全量，差值不代表方法退化。
- **Visualization**: grounding-results为样本、完成情况、两阈值与反例数的原生表格。
- **Native-ready**: grounding-results=yes

#### Slide 07 - All52 review

- **Audience move**: 不把全部指标失败当作模型语义错误。
- **Relationships**: 52条反例属于四个互斥初筛类别；其中14条零IoU细分为9条较明确错误与5条范围疑点。
- **Composition**: 四类数量条形图配一个关键读数和限定说明。
- **Title**: 52条指标失败，不能全部归因于模型
- **Core message**: 初筛22明确、15疑点、10混合、5不足，未改GT和指标。
- **Content**:
  - 较明确定位错误22；标注或范围疑点15；两者并存10；证据不足5。
  - 14条零IoU中5条仍有相关事件。
  - 52/52原始64帧助手核查，尚无独立人工复核；不外推成数据集标签错误率。
- **Visualization**: failure-counts为四类数量的水平条形图，数值22、15、10、5。
- **Native-ready**: failure-counts=yes

#### Slide 08 - True error

- **Audience move**: 从抽象错误数量看到可见的事件错位。
- **Relationships**: GT中的鸡舍画面与预测中的花景、道路构成内容不匹配。
- **Composition**: 原帧证据为主，右侧预测、GT和判断。
- **Title**: 真正定位错误：鸡舍被定位成花景
- **Core message**: qid448选错事件位置。
- **Content**:
  - 查询：鸡在鸡舍里。GT92–112秒；预测130–145秒；IoU0。
  - GT有围栏内鸡；预测主要为道路、紫色花朵和文字牌。
  - 可检验方向：预测是否满足对象与场景条件；尚未运行修复。
- **Images**: qv_448.jpg完整对照。

#### Slide 09 - Annotation suspect

- **Audience move**: 理解零重叠也可能来自匹配实例漏标。
- **Relationships**: 早段预测与后段GT均出现圣诞树相关活动，query未指定最后一次。
- **Composition**: 延续上一页GT／预测证据位置，改为标注疑点解释。
- **Title**: 疑似漏标：模型找到了前一次布置圣诞树
- **Core message**: qid3676零IoU不能直接证明找错内容。
- **Content**:
  - GT118–144秒；预测0–24秒；IoU0。
  - 5.83、10.53、15.23秒可见戴圣诞帽女子整理小树。
  - 需人工确认setting up的动作粒度和重复实例覆盖，不能直接更改GT。
- **Images**: qv_3676.jpg完整对照。
- **Motion suggestion**: 对照两行与右侧判断区域沿用P08作为可选连续性；本次未启用自定义动画。

#### Slide 10 - Mixed boundary

- **Audience move**: 允许模型和标签问题同时存在。
- **Relationships**: 编辑软件早于GT开始出现，且晚于预测终点仍持续；两侧分别构成标签疑点和模型遗漏。
- **Composition**: 延续证据页，右侧分开起点与终点两项解释。
- **Title**: 边界与标注问题可能同时存在
- **Core message**: qid5466不能简单归责一方。
- **Content**:
  - GT0–8、48–62秒；预测38–57秒；IoU0.3750。
  - 38.64秒起已有软件，GT48秒起点可能过晚；约59秒仍有软件，预测57秒终点偏早。
  - 精确秒级边界仍需连续视频核验。
- **Images**: qv_5466.jpg完整对照。
- **Motion suggestion**: 证据两行延续P09；右侧增加两类独立判断。本次未启用自定义动画。

### Part 3: 未解决问题与下一步

#### Slide 11 - Limits

- **Audience move**: 分清工程可运行与科学问题已解决。
- **Relationships**: 已完成的运行保障与未解决的工程／证据问题并列，互不替代。
- **Composition**: 已处理与仍未解决两个开放区域，突出真实停止记录。
- **Title**: 分批执行可用，定位问题仍需验证
- **Core message**: 分批完成不等于OOM根因修复，也不等于Agent有效。
- **Content**:
  - 已处理GPU编号兼容、输入／源码冻结、分批执行、失败与退出记录。
  - OOM缓存累积仍是主要线索，尚无独立GPU修复验证。
  - 非法输出[150,17]被拒绝；扩样最后4条、TVSum Direct5条未补跑。
  - 64帧无声音；部分动作、状态与GT边界需人工；本阶段没有Agent修复结果。

#### Slide 12 - Next experiment

- **Audience move**: 知道后续最小对照应按错误类型设计并检验副作用。
- **Relationships**: 明确错误先核验，分别进入事件条件或边界补证；标注疑点进入人工审阅；修复与原命中回归共同决定后续扩展。
- **Composition**: 一个行动主张配两类定向验证与一条人工核查支路，底部明确验收读数。
- **Title**: 下一步：按错误类型验证局部补证
- **Core message**: 先验证具体定位错误能否被补证修复，再接入Agent。
- **Content**:
  - 选错事件：核验对象、动作、关系和场景是否同时满足。
  - 边界错误：检查起点之前和终点之后的连续证据。
  - 标注疑点：人工复核，保留发布GT；语音相关样本另行检查。
  - 同时报修复率、原命中误改、全分母定位指标和调用成本。
  - 计划尚未执行；LensWalk／Gemini代码仍待对接。
- **Closing impact**: binding takeaway=先用可靠反例验证补证，再扩大Agent实验；建议以简洁行动句和有区别的验证支路结束。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 每页讲稿解释可见证据、指标分母和限制；列出本地源报告，事实以归档材料为准；不引入新成果或预算授权。
- **Total duration**: 约10分钟
- **Notes style**: 中文、自然、导师组会口吻，结论后给证据，不照读表格。
- **Presentation purpose**: 汇报已完成工作，解释grounding失败，并对齐下一步验证方向。

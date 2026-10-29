<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: ppt169

## communication
- primary_language: zh-CN
- audience: 导师与研究组成员，了解ReVALUE及10月27日组会背景
- objective: 汇报Direct与grounding诊断，使听众区分已完成结果、证据局限和未验证的Agent修复，并理解下一步局部补证。
- core_message: 已有可追溯定位反例；先验证具体错误的补证，再扩大Agent实验。
- consumption_mode: balanced

## mode
- mode: custom
- mode_references: pyramid
- mode_behavior: 标题先陈述证据支持的判断，再以数字或原帧说明；区分机制线索、定位精度和标注疑点，结尾落实待验证的修复方向。

## visual_style
- visual_style: custom
- visual_style_references: swiss-minimal
- visual_style_behavior: 精确网格、白底、开放色面、较大留白和强字重对比；原帧保持原色，案例页共享GT与预测的阅读顺序；原生文字与数据可编辑。

## colors
- background: #FFFFFF
- secondary_bg: #F2F5F8
- primary: #17324D
- accent: #237F8C
- secondary_accent: #C76128
- body_text: #17324D
- secondary_text: #52667A
- divider: #D5DEE8
- surface: #FFFFFF
- grid: #E7EDF3
- positive: #16834E
- negative: #B74343
- unknown: #7A8793

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- title_family: Microsoft YaHei, Arial, sans-serif
- body_family: Microsoft YaHei, Arial, sans-serif
- code_family: Microsoft YaHei, Consolas, monospace
- body: 24
- title: 42
- subtitle: 30
- annotation: 18
- lead: 28
- footnote: 16
- display: 112
- hero: 64
- number: 32
- code: 22

## icons
- library: none
- inventory: none

## images
- p08: images/qv_448.jpg | source=user | crop=no-crop
- p09: images/qv_3676.jpg | source=user | crop=no-crop
- p10: images/qv_5466.jpg | source=user | crop=no-crop

## page_rhythm
- P01: anchor
- P02: dense
- P03: breathing
- P04: dense
- P05: dense
- P06: dense
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: anchor

## page_visualizations
- P04: table/comparison_matrix
- P05: chart/gantt_chart
- P06: table/metric_table
- P07: chart/horizontal_bar_chart

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters

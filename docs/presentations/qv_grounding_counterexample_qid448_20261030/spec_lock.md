<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: ppt169

## communication
- primary_language: zh-CN
- audience: 导师与研究组成员
- objective: 用 qid448 的原帧和预测区间解释 Direct 定位错误，使听众分清已观察证据、未确定原因与待验证的修正方案。
- core_message: 原输入含目标证据但预测落到另一场景；事件条件核验的修正收益待验证。
- consumption_mode: balanced

## mode
- mode: custom
- mode_references: instructional
- mode_behavior: 按查询、区间对照、原帧证据、诊断、待验证步骤递进，讲稿承接细节。

## visual_style
- visual_style: custom
- visual_style_references: swiss-minimal
- visual_style_behavior: 白底、开放分栏、直角细线与精确对齐；大图承载证据，红绿区分预测与标注，右侧用简短文字解释。

## colors
- background: #FFFFFF
- secondary_bg: #F2F5F8
- primary: #17324D
- accent: #237F8C
- secondary_accent: #C76128
- body_text: #17324D
- secondary_text: #52667A
- divider: #D5DEE8
- positive: #16834E
- negative: #B74343

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- title_family: Microsoft YaHei, Arial, sans-serif
- body_family: Microsoft YaHei, Arial, sans-serif
- body: 24
- title: 42
- subtitle: 30
- lead: 28
- annotation: 18
- footnote: 16
- hero: 60

## icons
- library: none
- inventory: none

## images
- p01: images/qv_448.jpg | source=user | crop=no-crop

## page_rhythm
- P01: anchor

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters

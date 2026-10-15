# 开发协议修订 1：stop 条件格式

登记于 2026-10-14，完整 val/test 尚未开始，没有正式冻结文件。修订依据 train 开发推理的格式诊断，不依据 val/test 分数。

## 根因和证据

v1 提示要求所有控制器动作都返回 `action/start/end/question`；校验器也对所有动作强制这四个字段。但 `stop` 不调用视觉工具，不需要工具子问题。模型通常返回：

```json
{"action":"stop","start":0.0,"end":0.0}
```

它是可解析且正确表达停止意图的 JSON。初次抽查 760 个完成控制器请求中 239 次 schema 失败全为这一模式；停队列归档时，全部 v1 完成请求中共有 267 次同模式失败。均为 `finish_reason=stop`，没有截断。后端只要求 `json_object`，不强制字段 schema；v1 的一次重试原样重复消息，temperature=0/seed42 下通常再次省略同一字段。

这是接口设计与校验规则的问题，v1 的 M1/A1 大量失败不能据此判定原方法无效。CPU 测试只模拟了四字段 stop，没有覆盖真实模型的自然三字段 stop，测试覆盖不足已补齐。

## v2 改动与保持的实验身份

- stop 接受 `action/start/end`，可选字符串 `question`；start/end 必须为 0。内部仅规范成空 question，原始模型回答保持三字段，不改写 trace。
- scene_snapper/clip_analyzer 仍必须四字段、有限且合法的区间；不放宽工具名称、未知字段、HD ID/长度、MR 候选约束或空候选处理。
- 控制器提示分别示范工具和 stop 格式。模型、revision、seed、图片、预算、开发名单、MR/HD 输出和合并规则不变。
- 新配置 `default_v2.json` / `dev_low_v2.json`，方法协议 `qvh_video_only_v2_stop_contract`；开发 run 为 `dev_v2_*` 和 `dev_low_v2_*`。
- 初始化输出协议、完整消息、像素、模型和采样保持 v1，使用明确的 `initial_protocol=qvh_video_only_v1`。只在完整身份匹配时复用初始化；控制器、局部工具和终答不跨版本共享。
- 同一 smoke8/dev50 的所有方法和预算点重新执行，不能只重跑 v1 失败的 query，也不改变原名单。旧 v1 未运行部分标记被修订取代，保持原计划记录。

## 保存与验证

源码、工具、配置和旧报告已保存到 `checks/stop_contract_diagnosis_v1.json` 所指的 archive。全部旧 run 数据库、图片、trace、预测、评分和预算留在原 `qvh_data/runs/*v1*`；不把重解析后的 stop 自动改成成功样本或旧成绩。

`checks/stop_contract_reparse_v2.json` 记录原始 267 个失败输出在新校验器下全部合法。这只是解析器验证，不是恢复了 267 个预测或证明了检索效果。新的 CPU 回归验证 MR/HD 停止后输出、无额外工具和重试，同时继续拒绝非法工具参数。实际自主 smoke/dev 结果须另外验证。

所有 v1 成本继续进入报告与 GPU 服务历史。v2 是完整 val/test 前登记的唯一开发协议修订，不是新建正式 run_id 绕过原尝试预算。正式冻结只接受完成的 v2 开发与成本诊断。

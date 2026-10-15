# 自动执行与恢复

原实验使用用户级 systemd 服务 `merit-qvh-supervisor.service`，并已在原机器启用开机启动和用户 linger。克隆仓库不会自动安装或启动服务；新机器须先按下节配置。重启后需数据盘、模型文件和指定 GPU 恢复可用，才能继续推理。

## 在新机器配置

先完成 README 中的数据、模型、容量及五视频工程检查，再配置监控。模板保留原实验的恢复参数，机器路径、GPU UUID 和运行标识需要填写。

```bash
cp configs/automation_v1.example.json configs/automation_v1.json
cp scripts/merit-qvh-supervisor.service.example scripts/merit-qvh-supervisor.service
python -m merit_qvh.cli where --config configs/main_gpu_v1.json
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
```

将 JSON 和 service 文件里的 `/path/to/MERIT-based-method`、`/path/to/python`、`<RUN_NAME>`、`<PROTOCOL_ID_FROM_WHERE>` 和 GPU UUID 占位符替换为真实值。`where` 返回完整协议 ID 和运行目录；运行目录应填写绝对路径。若分配的不是 GPU 4、5，同时更新 `gpu_uuids` 的键、两个 `CUDA_VISIBLE_DEVICES` 值及启动命令。JSON 不会自动展开 shell 变量。

首次接管既有进程时，`adopt_worker_command` 和 `service_command` 必须与真实启动参数一致；只接管属于本实验的进程。确认日志目录存在后安装用户服务：

```bash
mkdir -p ~/.config/systemd/user
cp scripts/merit-qvh-supervisor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now merit-qvh-supervisor.service
loginctl show-user "$USER" -p Linger
```

若需要退出登录后或开机时持续运行，按本机权限配置 `loginctl enable-linger "$USER"`。原机器的启用状态不代表新机器已启用。

## 检查与恢复规则

- 每 60 秒检查进程、模型服务、完成数量与磁盘空间。
- 推理进程退出后从未完成的查询继续；成功和失败结果均保留，不重复执行已完成查询，不重置已耗尽的模型请求重试次数。
- 模型服务连续三次健康检查失败，或推理 30 分钟没有新查询完成时，清理本实验进程并恢复服务。模型服务启动有 10 分钟宽限期。
- 自动重启间隔按 1、2、4、8、16、30 分钟递增，取得新进展后重置。监控本身退出时由 systemd 在 15 秒后重启。
- 每次恢复核验模型身份、冻结源码和查询结果；只使用记录的 GPU 4、5 及其 UUID。若 GPU 被其他进程占用则等待，绝不结束其他任务。
- 进程管理同时核对用户、开机 ID 和进程启动时间，避免 PID 复用造成误操作。
- 连续 10 条查询失败、配置/源码改变或存储异常时，记录明确状态并停止盲目推进。`attention_required` 表示需要检查具体原因；自动化不会自行改提示词或实验参数。
- 全部 1,550 条查询处理完后自动调用官方评估器，核对输出完整性，随后结束本实验进程并释放视觉服务占用的 GPU。

## 状态与日志

原实验的历史运行标识如下；`artifacts/` 为发布版示例路径，不包含远端运行数据。新运行以 `where` 的输出为准：

```text
artifacts/runs/merit_qvh_main_v1-15f5f6e4ee00/
```

- `runtime/automation_status.json`：最新检查时间、阶段、完成数量及等待原因。
- `runtime/automation_events.jsonl`：进程接管、自动重启、异常和完成记录。
- `logs/supervisor.log`：监控日志；恢复后的推理日志为 `logs/automatic_worker_*.log`。
- `runtime/active_runner.json`、`runtime/active_vlm_service.json`：当前进程身份和启动参数。
- `checks/supervisor_restart.json`：监控进程被终止后自动恢复的实际验证记录。
- `runtime/automation_install.json`：自动化脚本及配置的源码快照、校验和。

## 管理命令

```bash
systemctl --user status merit-qvh-supervisor.service
systemctl --user restart merit-qvh-supervisor.service
```

若需要关闭自动监控：

```bash
systemctl --user disable --now merit-qvh-supervisor.service
```

关闭监控会保留正在运行的实验进程；该命令不是终止实验。监控重启时也保留实验进程，并重新核验身份后接管。

恢复入口为 `scripts/resume_validation.py`，直接调用既有冻结推理实现。历史失败不会再次触发“连续三条新查询失败”保护；每轮恢复只对新处理的查询计数。上述运行管理改动不改变模型、提示词、评分网格、失败回退或评估分母。

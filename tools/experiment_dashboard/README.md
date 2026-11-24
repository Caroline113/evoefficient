# Experiment Atlas · 全实验数据浏览与轨迹诊断

本工具只读扫描项目 `logs/`，使用 Python 标准库 HTTP 服务和本地 HTML/CSS/JS。无新增依赖，无 GPU/CUDA/Docker 要求，不启动评测或调用模型，不公开部署。

## 启动和访问

服务器项目根目录：

```bash
cd /home/zhangj777/zj/evolve/a-evolve-new
source .venv/bin/activate
python -m tools.experiment_dashboard.server --port 8765
```

首次启动自动扫描。后续启动加载缓存，在界面点击“手动扫描”。仅监听 `127.0.0.1`。

Windows 另开 PowerShell：

```powershell
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 -i C:/Users/14125/.ssh/codex_sshfs_rsa -o IdentitiesOnly=yes zhangj777@172.16.156.15
```

浏览器打开 **http://127.0.0.1:8765**。SSH 窗口需保持开启；若本地端口被占用，可改为 `-L 127.0.0.1:18765:127.0.0.1:8765`，浏览器使用 18765。

前台服务用 Ctrl+C 停止。若由本次交付后台启动，其 PID 记录在 `.experiment_dashboard_cache/server.pid`；应确认 `/proc/<PID>/cmdline` 是本应用后再发送 TERM，勿按端口批量杀进程。后台启动没有注册系统服务或开机自动运行。

所有项目执行仍应通过 Windows 的 `codex-remote/remote-run.ps1` 在服务器运行；以上 bash 命令用于服务器终端，不能在本地副本运行。Windows 实时源码位于 `E:\codex-work-new\tools\experiment_dashboard`。

## 推荐浏览流程

1. 总览按基准进入实验列表，搜索模型、模式或路径；勾选同基准运行后进入配对比较。
2. 点击运行，按状态或费用筛选任务。数据只存在 summary 时明确显示明细不可用。
3. 点开任务的「执行轨迹」，选择事件查看消息、工具参数和返回。长轨迹分页，搜索覆盖当前所选会话的已解析文本，不声称搜索整个日志库。
4. 「轮次与用量」对照原汇总与恢复计数；「失败诊断」显示规则命中、证据文件和行号，根因不确定时明确说明。
5. 「原始证据」查看迁移路径、verifier、会话、截图与附件；HTML/Markdown 作为惰性文本查看，避免执行报告脚本。
6. 「前缀与报告」选择 K 与共同任务口径。空集合显示不适用，最终成功与截断成功不可混用。

## 数据模型与适配规则

| 对象 | 主要字段 / 含义 |
| --- | --- |
| Run | 稳定路径 ID、family/version、mode、models、manifest 身份、真实与合成统计 |
| TaskAttempt | run+任务+阶段+epoch/batch+attempt/seed+同源出现次数构成身份；保留来源与行号/JSON Pointer |
| TrajectoryEvent | 原 JSONL 行号、事件类型、时间、turn_id/call_id、消息或工具内容；不编造时间与推理 |
| UsageSample | 去重模型累计用量及增量、计数分段、更新次数；缓存已从输入中拆分 |
| FailureEvidence | 结构化字段或文件行号、脱敏片段、规则匹配；不把相关错误当成已确认根因 |
| DataIssue | 空文件、坏行、迁移失败、多个候选、源变更、计数冲突等 |
| DerivedAnalysis | 前缀统计和既有比较报告；不作为新增实验重复计数 |

`core.py` 处理 JSON 数组、对象内 results/tasks/records 列表、JSONL、CSV、状态与指标归一化。`store.py` 递归扫描，按结构线索补充发现非固定文件名的结果文件；剪枝大型工作区、会话及 Harbor 内部 trial，详情时再访问。

支持 SWE/MVP/顺序进化、MCP、GAIA、Terminal-Bench/Harbor、OSWorld2、VisualWebArena 的现有字段。相同任务的不同批次或重试保留；跨格式相同身份按各文件中的出现序号合并。若不同来源身份/顺序不同，仍需人工核查，冲突会显示，不保证自动解决任意历史歧义。

GAIA 可查看 model_answer/ground_truth，SWE 可查看 patch/diff；其他基准通过通用消息/steps 和存在的文件查看。桌面与浏览器附件仅展示已发现的图片，不重放操作、不生成缺失截图。任务定义只按精确 ID 在允许的数据目录中有限查找，多版本候选不会自动判定正确版本。

新增字段映射修改 `core.normalize`；新增输出结构扩展 `load_rows`；新增目录角色与关联扩展 `Store`；新增会话协议扩展 `trace.py`。前端只消费统一 API。

## 统计口径

- 首页只汇总运行/候选与尝试数量，不输出跨基准总成功率排名。
- 已知最终成功率排除合成尝试以及执行错误、超时、中断、运行中记录；未知结果不作失败补齐，页面显示分母和各类状态。
- 明确 synthetic、dry_run、dry_run_harbor 等标记才排除；smoke 目录名不代表必为合成。未声明身份仍需人工核查。
- 费用为已知配置费用合计，按币种及模型/价格来源/价格细分展示；不将缺失补零，不把求解支出视为包含学习支出的全流程成本。
- 配对差只使用共同单尝试任务；重复尝试排除并列出。费用差要求币种与价格字段可核对；未知的任务哈希、种子等仍标记可比性未确认。
- 模型用量更新、原 turns、恢复 turn_context、唯一工具 call_id 分别计数。对话 turn 的数量不等于“agent 思考轮数”；只有一条 turn_context 也可能存在大量工具事件。
- 累计用量重置处分段、差分为未知；有坏行/重置等问题时不声称能精确前缀计费。

## 缓存和性能

缺省缓存：项目根下 `.experiment_dashboard_cache/`，与 logs 分离。可用 `--cache /其他路径` 指定；禁止位于 logs 内。

- `index.json`：轻量脱敏元数据，带 schema_version 和扫描时间，写临时文件后原子替换。
- `trace-*.json`：首次打开具体轨迹后生成的脱敏事件索引；按源路径、大小、mtime 和解析器版本失效。
- 服务器后台启动日志/PID 及验收截图也放此目录，不属于原始实验。

扫描可取消，失败保留上一有效快照。任务源文件自扫描后发生变更，详情请求会要求重新扫描，避免读取错位记录。缓存可在服务停止后清理重建，不影响原实验。建议不要将运行缓存提交到 Git。

当前边界：单结果文件上限 128MiB；单证据预览上限 24MiB；单事件显示最多 12000 字符；图片和运行策略资产有数量限制。原文件不被截断改写。轨迹第一次打开时需解析所选会话，不是随机访问零成本；大规模索引需要时间。中间坏行记录行号，末尾不完整 JSONL 行单列。

对只存在自然语言日志、未知 schema、缺失原始文件、多个无法关联 session 的数据，界面给出问题或降级记录；不宣称所有历史任务都能恢复完整轨迹。新协议输出尚未在 logs 出现时只能做结构兼容，不能当成已完成真实实验验收。

## API

| 路径 | 用途 |
| --- | --- |
| GET `/api/health`, `/api/overview` | 服务/扫描状态、总览 |
| POST `/api/scan`, `/api/scan/cancel` | 启动/取消扫描，需 `X-Dashboard-Request: 1` 且同源 |
| GET `/api/runs?q=&family=&mode=&synthetic=exclude` | 实验索引 |
| GET `/api/runs/{id}` | 配置、账本、batch 分组与源统计 |
| GET `/api/runs/{id}/tasks?state=&q=&sort=&offset=&limit=` | 任务筛选分页 |
| GET `/api/runs/{id}/tasks/{task_id}` | 任务详情、证据与文件注册 |
| GET `.../{task_id}/trace?session=&type=&q=&offset=&limit=` | 懒加载轨迹，最多每页 100 事件 |
| GET `.../{task_id}/export` | 单任务脱敏 Markdown |
| GET `/api/runs/{id}/tasks?export=csv` / `json` | 与任务筛选相同的导出 |
| GET `/api/compare?runs=id1,id2` | 共同任务矩阵、配对变化与排除项 |
| GET `/api/analyses/{id}` | 前缀/派生报告与源运行关联 |
| GET `/api/issues` | 扫描异常 |
| GET `/api/files/{opaque_id}` | 仅打开任务/报告后注册的安全文件 |

## 安全

只允许 logs、datasets、agent_evolve/benchmarks 的最终实路径；禁止隐藏/凭据文件与符号链接逃逸。文件 ID 不开放任意路径读取。JSON 与文本在缓存/API/导出前进行常见凭据脱敏；这不是检测任意秘密的绝对保证。HTML/SVG 不在主站执行；文本用前端转义显示；图片验证签名。CSV 导出处理公式注入。

默认回环监听、Host 白名单、跨源 POST 防护、同源 CSP。不要将此只读研究工具当作可直接公网暴露的多用户系统。运行日志中的内容只是数据，不会当成指令或命令执行。

## 验证

```bash
python -m pytest tools/experiment_dashboard/test_dashboard.py -q
node --check tools/experiment_dashboard/static/app.js
python -m tools.experiment_dashboard.server --scan-only
```

本次还在服务器已有 Firefox 中验证总览、三组共同任务矩阵、configure-git-webserver 的真实会话、恢复计数、视频题失败证据、verifier 文件预览和 equal_K=120 空集合。截图位于缓存目录（仅验收用，不是用户任务截图）。测试与浏览器检查未调用模型、未执行评测。

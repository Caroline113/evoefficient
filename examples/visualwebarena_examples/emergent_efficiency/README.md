# VisualWebArena：自适应进化与涌现效率实验

本目录将现有 SWE、Terminal-Bench Adaptive、OSWorld2 实验中的 **分批求解 → 普通自适应进化 → 可选效率提案 → 下一批** 流程扩展到 VisualWebArena（VWA）。新增文件集中在本目录，不修改既有 benchmark 的运行代码或历史实验结果。

**先看清运行边界：**

- A-Evolve 负责实验模式、批次、工作区进化、效率提案、统计与产物。
- 正式任务由官方 VisualWebArena 的 `run.py:test()` 执行，保留浏览器、动作解析、登录更新、截图/trace 和官方 evaluator。不是 Harbor/OSWorld 容器任务，也不是把网页任务转换为 Terminal-Bench。
- 官方 VWA 运行在独立 Python 环境，主程序通过子进程桥接。无需把 VWA 的旧版依赖装进本项目 `.venv`。
- `--dry-run` 完全不调用模型、不打开浏览器、不访问网站；结果标记 synthetic，score 为 null，绝不伪造成通过。
- 本实现首次提供工程接入和离线验证。**离线通过不等于真实 VWA 网站、BLIP-2、模型服务与评测已全部部署或跑通。**正式结果应在部署完成后产生。

官方参考：[仓库与安装说明](https://github.com/web-arena-x/visualwebarena)、[网站部署与重置](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md)、[原生执行入口](https://github.com/web-arena-x/visualwebarena/blob/main/run.py)。以自己锁定的官方 commit 为准；manifest 会记录该 commit。

## 1. 与三个现有实验的关系

| 参考目录 | 本实现保留的部分 | VWA 必须调整的部分 |
|---|---|---|
| `examples/tb_examples/emergent_efficiency_adaptive` | AdaptiveSkillEngine、非末尾批次后进化、trajectory-only 默认语义 | 求解侧改为官方浏览器 runner，不能复用 terminal 工具 |
| `examples/swe_examples_new/emergent_efficiency` | JSON 提案解析、skill/fragment/memory 应用与 adoption 文件结构、类别 taxonomy | proposer 提示改为浏览器/视觉任务；拒绝不兼容的生成工具 |
| `examples/osworld2_examples/emergent_efficiency` | 视觉任务与外部执行器分离、独立任务日志、种子工作区 | VWA 有网站状态/账号、元素ID、参考图和不同评测协议，不沿用桌面任务配置 |

新增代码直接复用 `agent_evolve.algorithms.adaptive_skill.AdaptiveSkillEngine`、`AgentWorkspace` 和 SWE runner 的 `_extract_json_object` / `_write_proposals` / `_apply_proposals`。不复制整个 SWE/OSWorld runner，也不全局改写它们的提示常量。

## 2. 文件结构

```text
examples/visualwebarena_examples/emergent_efficiency/
├── README.md
├── run_emergent_efficiency_vwa.py   # 实验调度、进化、提案、输出
├── vwa_worker.py                   # 单任务官方 VWA 桥接，独立解释器
├── common.py                       # HTTP调用、usage、超时与JSON产物
├── run_one.sh                      # 单模式入口
├── run_matrix.sh                   # 五模式顺序运行
├── smoke_tasks.json                # 明确标记的合成输入
├── prices.example.json             # 需填写的价格模板，null不可直接用于计价
├── seed_workspace/                 # 仅包含通用浏览器技能，无任务答案
└── tests/                          # 服务端离线回归测试
```

## 3. 实验模式与公平比较

| mode | 普通进化 | 额外效率提案 | 固定效率提示 |
|---|---:|---:|---:|
| `normal-evolve` | 是 | 否 | 否 |
| `cost-aware` | 是 | 是 | 否 |
| `predefined-control` | 否 | 否 | 是 |
| `predefined-cost-aware` | 是 | 是 | 是 |
| `category-seeded-cost-aware` | 是 | 是，提供 M1–M7 广义类别 | 否 |
| `static-control` | 否 | 否 | 否 |

`predefined-*` 的控制是**本目录定义的固定浏览器效率提示**，强调视觉定位、页面变化后重新观察和避免重复导航。它不是 SWE/Terminal 的 observation masking、工具裁剪或模型路由的完整运行时实现。manifest 会明确标记，不能把跨 benchmark 的同名模式当成干预强度完全相同。

每次运行均创建新的输出子目录、独立复制 seed workspace；不会继承上一模式已经学到的技能。最后一个批次不执行进化，因为之后已无任务使用新知识。

初始种子含一项通用 browser-grounding 技能，所有模式相同。`max-skills` 在普通进化和效率提案后统一检查：保留已有技能，超限的新技能正文会被移除，并在 `skill_budget.jsonl` 记录。普通进化还会提示模型保护已有技能；这不是对任意模型修改的文件系统级隔离保证。

## 4. 最快验证：完全离线的冒烟运行

以下命令在服务器本项目根目录执行：

```bash
cd /home/azureuser/zj/a-evolve-new7/a-evolve-new
source .venv/bin/activate

# 4个合成任务，2批；不会访问网站或调用模型
bash examples/visualwebarena_examples/emergent_efficiency/run_one.sh \
  smoke cost-aware --dry-run --limit 4 --batch-size 2

# 五种模式均检查参数、工作区、批次和输出
bash examples/visualwebarena_examples/emergent_efficiency/run_matrix.sh \
  smoke_matrix --dry-run --limit 4 --batch-size 2

# 详细参数
python examples/visualwebarena_examples/emergent_efficiency/run_emergent_efficiency_vwa.py --help

# 回归测试；依然不需要真实模型和网站
python -m pytest examples/visualwebarena_examples/emergent_efficiency/tests -q
```

所有 shell 入口默认使用项目 `.venv/bin/python`，也可设置 `AEVOLVE_PYTHON=/绝对路径/python`。通过 `bash script.sh` 运行，无需修改可执行权限。

smoke 的 `success_rate`、`score` 和总成本为 null，`synthetic_tasks` 非零；此时没有模型进化，只有 `skipped_dry_run` 标记。这是通路验证，不是 benchmark 成绩。

## 5. 正式运行前：准备官方 VWA 与网站

### 5.1 两个 Python 环境

主程序使用现有项目 `.venv`。VWA 使用自己环境（官方文档列出 Python 3.10/3.11 支持），例如：

```bash
# 仅在尚未有官方 checkout 时克隆；已有目录请直接使用
cd /home/azureuser/zj
git clone https://github.com/web-arena-x/visualwebarena.git
cd visualwebarena

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install
python -m pip install -e .

# 部署时记录版本；正式实验建议固定已验证的commit
 git rev-parse HEAD
```

上面是准备步骤，不会由实验入口自动执行。若系统只有其他 Python 命令，请替换为实际解释器。不要把官方依赖批量安装到 A-Evolve 的已有 `.venv`，否则可能破坏其他实验。

### 5.2 网站地址、任务配置与登录

先按官方 `environment_docker/README.md` 部署网站，再填写实际地址：

```bash
export DATASET=visualwebarena
export CLASSIFIEDS='http://你的服务器:9980'
export CLASSIFIEDS_RESET_TOKEN='与你部署一致的reset-token'
export SHOPPING='http://你的服务器:7770'
export REDDIT='http://你的服务器:9999'
export WIKIPEDIA='http://你的服务器:8888'
export HOMEPAGE='http://你的服务器:4399'

cd /home/azureuser/zj/visualwebarena
source .venv/bin/activate
python scripts/generate_test_data.py
bash prepare.sh
```

地址必须能从 VWA worker 所在服务器访问。不要仅填写示例占位地址。以上变量均为官方 VWA 导入时需要的设置；即使只选一个站点，其他所需变量也不能缺失。

`generate_test_data.py` 生成的逐任务 JSON 才是 `--tasks-dir` 输入。不要把尚含替换占位符的 raw 模板当成已部署任务。官方登录流程在单任务执行时可能刷新 cookie；worker 会把对应 VWA 解释器目录加入 PATH，使其派生的 `python` 与 VWA 环境一致。

浏览器、网站、登录、GPU/CPU captioner 和可访问的模型服务需分别验证。原生 image_som 流程会初始化本地 captioner，某些 evaluator 也需要 captioner/VQA；不能因为求解使用多模态 HTTP 模型就删除这些依赖。`--eval-captioning-model-device` 只控制评测 captioner 的设备，不能保证原生 observation captioner 不使用 GPU。

### 5.3 模型与服务配置

求解侧接受 OpenAI-compatible **Chat Completions** 接口：

```bash
export VWA_REPO=/home/azureuser/zj/visualwebarena
export VWA_PYTHON="$VWA_REPO/.venv/bin/python"
export VWA_BASE_URL='https://你的多模态服务/v1'
export VWA_API_KEY='你的key'
export VWA_MODEL='服务实际支持的模型名'

# 普通进化和效率提案仍走项目已有 provider factory
export VWA_EVOLVER_MODEL='deepseek:deepseek-v4-flash'
# 再按该provider的既有方式配置凭据；这与VWA_API_KEY是独立的。
```

- `--model-id` 是实际 Chat Completions 模型名，直接发送给 VWA_BASE_URL，不自动解析 `deepseek:` 等 A-Evolve provider 前缀。
- `--evolver-model` 则遵循本项目已有 `create_default_llm` 的模型选择规则。
- `image_som` 必须使用能接收图片的模型；纯文本模型不能仅靠换名字变为视觉模型。`accessibility_tree_with_captioner` 提供 caption 辅助文本基线，不能把它与直接多模态模型视为同一实验条件。
- worker 调用真实模型名，同时修正官方 agent 的历史模型名白名单，避免现代模型别名导致截图未送入提示。未知模型 tokenizer 回退到 `cl100k_base` 仅用于文本 observation 截断；usage 依然采用服务返回值，不拿该 tokenizer 估算账单。
- 官方部分语义评分会请求其指定的 judge 模型。可通过 `VWA_JUDGE_BASE_URL`、`VWA_JUDGE_API_KEY` 为 judge 配置单独服务；默认使用求解侧服务。**不擅自将官方 judge 改为求解模型**；服务必须支持 evaluator 请求的模型，否则应修正部署，不能把请求失败算成功。

## 6. 选择任务、单任务试跑与正常实验

### 6.1 查看任务ID

```bash
cd /home/azureuser/zj/a-evolve-new7/a-evolve-new
source .venv/bin/activate

python examples/visualwebarena_examples/emergent_efficiency/run_emergent_efficiency_vwa.py \
  --tasks-dir "$VWA_REPO/config_files/vwa/test_classifieds" --list-tasks
```

从单站点目录读取时，ID 通常是 `0`、`1` 等；从包含多个站点的父目录读取时，ID 带相对路径，例如 `test_classifieds/0`，避免不同网站的 `0.json` 冲突。ID 以 `--list-tasks` 实际输出为准。读取顺序为相对路径字典序（例如10可能排在2前）；要严格固定配对任务顺序，请显式传 `--task-ids`。

### 6.2 真实单任务验证：先关闭进化

```bash
bash examples/visualwebarena_examples/emergent_efficiency/run_one.sh \
  real_one static-control \
  --vwa-repo "$VWA_REPO" --vwa-python "$VWA_PYTHON" \
  --tasks-dir "$VWA_REPO/config_files/vwa/test_classifieds" \
  --task-ids 0 --limit 1 --batch-size 1 \
  --model-id "$VWA_MODEL" \
  --allow-unreset-sites
```

这会真正调用求解模型和官方评测，可能产生费用及网站状态修改。`--allow-unreset-sites` 是**显式接受共享状态**的单任务调试选项，不适合直接用于声称公平的矩阵比较。

任务成功与否取决于官方 evaluator 的数值。worker进程退出0不代表成功；评分0是正常失败，未产生评分则是执行/评测错误。

### 6.3 普通进化与 cost-aware 对照

准备一个你已验证的站点重置入口，例如 `/绝对路径/reset_vwa_state.sh`。这不是本目录提供的脚本：必须按自己的部署重置相关站点数据与账号状态，并保证脚本失败时非零退出。runner 会在**每个任务前**运行它，工作目录为 VWA checkout；此命令按 argv 解析，不通过 shell 执行，不能直接在参数里堆 `&&` 或管道。

```bash
COMMON=(
  --vwa-repo "$VWA_REPO"
  --vwa-python "$VWA_PYTHON"
  --tasks-dir "$VWA_REPO/config_files/vwa/test_classifieds"
  --task-ids 0,1,2,3,4,5,6,7,8,9
  --limit 10 --batch-size 2 --parallel 1
  --model-id "$VWA_MODEL"
  --evolver-model "$VWA_EVOLVER_MODEL"
  --max-steps 30 --max-skills 10
  --reset-command 'bash /绝对路径/reset_vwa_state.sh'
)

bash examples/visualwebarena_examples/emergent_efficiency/run_one.sh \
  vwa10 normal-evolve "${COMMON[@]}"

bash examples/visualwebarena_examples/emergent_efficiency/run_one.sh \
  vwa10 cost-aware "${COMMON[@]}"
```

不要在多个进程或机器上同时复用同一组网站数据库。当前 `--parallel` 只接受1。实验默认串行，代价是速度较慢，但避免在未提供独立网站副本时伪装任务隔离。重置失败会中止实验，不会继续使用污染状态。

### 6.4 五模式矩阵

```bash
bash examples/visualwebarena_examples/emergent_efficiency/run_matrix.sh \
  vwa_matrix "${COMMON[@]}"

# 或仅选择两种模式
VWA_MODES='normal-evolve,cost-aware' \
  bash examples/visualwebarena_examples/emergent_efficiency/run_matrix.sh \
  vwa_pair "${COMMON[@]}"
```

默认顺序固定，不代表统计设计中的随机化。正式研究建议重复整个在线序列、交错模式运行顺序、使用相同task-ids和seed workspace，并留出未参与进化的任务；本目录没有自动恢复既有workspace的resume功能。重跑会创建新的时间戳目录，避免误覆盖或混入历史结果。

## 7. 技能如何真正参与求解

每个任务启动前，会读取当前 workspace 的 system prompt、fragment、完整 SKILL.md 正文和 memory JSONL，写入任务专用官方 instruction JSON 的 `intro`。这样技能内容随请求传递，不依赖浏览器环境访问主机路径，也不要求存在 `read_skill` 工具。

产物 `context_manifest.json` 列出注入文件及合并提示哈希，`workspace_context.md` 保存正文，`instruction.json` 保存实际官方提示配置。

`--workspace-context-chars` 默认30000，超限会显式失败，**不会悄悄截断技能再声称已注入**。可先降低技能数、合并重复技能或提高预算；预算越大，求解输入也可能越贵。

仅支持效率提案类型：`skill`、`prompt_fragment`、`memory`。`tool` 类型会写入 rejected 记录，因为官方浏览器agent没有可调用的自定义Python工具注册接口。不要把生成工具文件的数量当成工具已经执行的证据。

普通进化默认trajectory-only，复用现有引擎对轨迹的推断；加 `--include-feedback` 才向普通进化提供官方结果。**效率proposer沿用既有cost-aware设计，可以看到成功/分数和成本字段**；trajectory-only不适用于它。进化器获得文字轨迹和截图文件路径，不会自动把所有截图转成多模态输入；求解器本身仍使用官方截图/目标图。

## 8. token、成本与评分口径

- 每次求解或judge HTTP请求都写 `usage.jsonl`，按真实响应记录usage。
- OpenAI格式的 `prompt_tokens` 已包含缓存读取；非缓存输入=`prompt_tokens−cached_tokens`。推理token是输出token的子集，不重复相加。
- 缺少usage不会估算成“0成本成功”；标记missing，未计价总成本为null。
- 默认**不假定任何模型价格**。可复制 `prices.example.json` 并填写与实际服务一致的每百万token USD单价：

```bash
cp examples/visualwebarena_examples/emergent_efficiency/prices.example.json /tmp/vwa_prices.json
# 编辑所有null为实际非负单价，solver与judge可不同；按实际使用的模型核对
# 然后在run_one.sh或矩阵参数中加入：
# --price-file /tmp/vwa_prices.json
```

- `results.json`/`summary.json` 的成本是**求解侧配置成本**；judge另记。
- 普通进化和效率proposer复用现有provider adapter，记录其原始usage（包括converse_loop聚合及trajectory judge）；不同adapter的缓存定义不统一，暂不强行按solver价格计费。`accounting.json` 因而明确保留端到端金额未知状态，不能直接宣称完整实验节省某百分比。
- BLIP-2等本地captioner的GPU/CPU消耗不计入token账单；墙钟时间仍包含这部分。需要硬件成本时应额外采集GPU/CPU指标。
- 两种观测模式、不同viewport、captioner版本、judge配置、重置方式和文本截断预算都影响比较，必须固定。

## 9. 输出与排错

输出根目录：

```text
logs/visualwebarena_emergent_efficiency/<run-name>_<mode>_<UTC timestamp>/
├── experiment_manifest.json   # 参数、代码commit、seed哈希、价格/状态隔离说明
├── task-list.json              # 实际任务顺序与官方config哈希
├── results.json                # 标准化逐任务结果
├── summary.json / summary.csv # batch / overall
├── accounting.json            # 求解、judge、普通进化、效率提案分账
├── usage.jsonl                 # 进化侧usage；请求或loop聚合范围有标记
├── run_status.json             # 是否完整结束；计划/完成任务数
├── runner_error.json           # 调度/重置/进化失败时
├── workspace/                  # 本次运行独立工作区
├── tasks/<slug+hash>/
│   ├── job.json / instruction.json
│   ├── context_manifest.json / workspace_context.md
│   ├── worker_result.json / usage.jsonl / steps.json
│   ├── screenshot_*.png / traces/ / render_*.html
│   └── stdout.log / stderr.log / reset.*.log
├── emergence/
│   ├── normal_evolution.jsonl / observations/
│   ├── batch_*_proposal_report.json / proposals.jsonl
│   ├── proposal_adoption.json / batch_*_rejected.json
│   ├── skill_budget.jsonl / workspace_snapshots/
└── figures/batch_overview.png
```

中间结果在每个任务后保存。超时会终止worker进程组，保留已写入的日志。部分会话开销若来不及完整返回，不能假定全部usage已被采集。

| 现象 | 优先检查 |
|---|---|
|启动提示缺少站点变量|补齐官方VWA所需变量；不要填不可访问占位地址|
|找不到run.py或prompt JSON|确认VWA_REPO和官方checkout结构；先按官方流程生成提示JSON|
|worker_failed / evaluation_missing|查任务stderr.log、官方error.txt、登录和captioner依赖；不会把缺失评分当0分正常结果|
|judge模型找不到|配置VWA_JUDGE_BASE_URL/KEY，使官方指定judge可用|
|模型拒绝图片|换支持视觉输入的服务，或明确改用captioner文本基线|
|成本为空|检查price-file、usage完整性；进化adapter缓存不明时这是预期行为|
|任务互相影响|确认真实站点重置有效，保证没有其他并行实验共享站点|
|workspace context超预算|合并重复技能、降低max-skills或提高context预算|
|第四个参数被shell吞掉|用Bash数组传COMMON，避免字符串拼接和未引用路径|

## 10. 从Windows挂载工作区运行

在 `E:\codex-work\examples\visualwebarena_examples\emergent_efficiency` 修改文件会直接写入服务器。**不要在Windows本地执行浏览器/模型/测试。**

通过既有包装脚本运行离线验证：

```powershell
& 'E:\evoefficient\codex-remote\remote-run.ps1' -Command @(
  'bash',
  'examples/visualwebarena_examples/emergent_efficiency/run_matrix.sh',
  'smoke_windows', '--dry-run', '--limit', '4', '--batch-size', '2'
)
```

正式实验推荐进入已导出站点和模型环境变量的服务器shell，执行第6节命令。不要把API密钥拼进remote-run命令字符串或保存进README。

## 11. 验证范围与维护

离线测试覆盖：缓存token不重复计数、未知价格不伪造成本、task ID冲突与重复选择、技能正文/预算、官方评测0分和缺失评分的区分、workspace提示注入，以及矩阵冒烟入口。官方VWA依赖通过worker延迟导入，因此离线主程序不要求安装Playwright、captioner或网站。

若升级官方VWA，首先核对 `run.config/test`、`agent.agent.call_llm`、`Tokenizer`、`construct_agent`、`evaluator_router` 及helper judge调用接口。本桥接会在这些接口变化时显式失败，需要重新跑集成测试与真实单任务检查；manifest中的官方commit用于定位对应版本。

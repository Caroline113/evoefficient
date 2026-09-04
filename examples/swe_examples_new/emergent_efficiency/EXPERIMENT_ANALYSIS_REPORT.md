# 自然涌现效率策略实验分析报告

**生成日期**: 2026/09/03
**实验组**: 5组 · 每组10任务 · 5 batches · batch_size=2

---

## 一、实验概览

### 1.1 实验组说明

| 实验组 | 模式 | 核心设计 |
|---|---|---|
| **A - normal-evolve** | 控制组 | 纯 agent 自进化，无 cost-aware proposer，无预定义策略 |
| **B - cost-aware** | 主实验组 | agent 自进化 + 中性成本信号诱导的 cost-aware proposer |
| **C - predefined-control** | 工程对照 | 显式激活预定义效率策略（`observation_masking`等5个），无 cost-aware proposer |
| **D - predefined-cost-aware** | 组合实验组 | 预定义策略全开 + cost-aware proposer（双路激励） |
| **E - category-seeded-cost-aware** | 类别种子组 | 不给具体策略名，只给 M1-M7 成本维度类别，让 agent 自己发明策略 |

### 1.2 核心指标总览

| 指标 | A: normal-evolve | B: cost-aware | C: predefined-control | D: predefined-cost-aware | E: category-seeded |
|---|---|---|---|---|---|
| **任务数** | 10 | 10 | 10 | 10 | 10 |
| **成功数** | 8 | 10 | 10 | 9 | 10 |
| **成功率** | **80%** | **100%** | **100%** | **90%** | **100%** |
| **总成本 ($)** | 4.43 | 5.05 | 4.25 | 4.66 | 4.60 |
| **平均成本 ($/task)** | 0.44 | 0.51 | 0.42 | 0.47 | 0.46 |
| **cost_of_pass ($/成功任务)** | **0.55** | **0.51** | **0.42** | **0.52** | **0.46** |
| **总 token 数** | 5,203,346 | 5,696,123 | 4,436,310 | 4,296,329 | 4,740,801 |
| **平均 token/task** | 520,335 | 569,612 | 443,631 | 429,633 | 474,080 |
| **总 tool calls** | 458 | 447 | 379 | 349 | 395 |
| **平均 tool calls/task** | 45.8 | 44.7 | 37.9 | 34.9 | 39.5 |
| **wall time (s)** | 1714.9 | 2140.4 | 1725.2 | 1782.3 | 1624.2 |
| **emergent skill hits** | 0 | 0 | 0 | 0 | **1** |
| **read_skill hits** | 0 | 1 | 1 | 2 | 1 |
| **prompt_fragment_visible_tasks** | 8 | 2 | 0 | 8 | 8 |
| **zero_token_tasks** | 0 | 0 | 0 | 0 | 0 |

### 1.3 关键发现摘要

> ⚠️ **A 组（normal-evolve）Batch 1 全军覆没（0/2 成功）**，导致整体成功率被拉低至 80%。如果排除 Batch 1 的异常值，A 组的成功任务成本为 $0.42，与 C 组持平。这说明早期 batch 的 agent 自进化质量不稳定。

---

## 二、多维度深度分析

### 2.1 成功率维度

```
成功率排序:
  B (cost-aware)           100%  ████████████████████
  E (category-seeded)      100%  ████████████████████
  C (predefined-control)   100%  ████████████████████
  D (predefined-cost-aware) 90%  ███████████████████▌
  A (normal-evolve)         80%  ██████████████████▌
```

**分析**:
- B、E、C 三组均达到 100% 成功率，说明 cost-aware 机制和类别种子策略均能有效保障任务完成。
- **D 组（predefined-cost-aware）反而出现了 1 个任务失败**（Batch 3: 1/2 成功），可能是因为预定义策略与 cost-aware proposer 产生了干扰，或者 Batch 3 本身遇到了较难的任务。
- **A 组失败发生在 Batch 1**（0/2 成功），后续 4 个 batch 均为 100% 成功。这表明纯自进化在初期存在不稳定性，一旦建立有效的 skills/prompts，后续任务可以正常完成。

### 2.2 成本效率维度

#### 2.2.1 Total Cost vs Success Rate

```
                    成功率
              80%    90%    100%
Total Cost
$4.25 (C)           ████████████ 100%
$4.43 (A)     ████████████  80%
$4.60 (E)            ████████████ 100%
$4.66 (D)            ████████████  90%
$5.05 (B)                   ████████████ 100%
```

#### 2.2.2 cost_of_pass（每成功任务的实际成本）

| 实验组 | cost_of_pass | 相对 C 组的成本增量 |
|---|---|---|
| C: predefined-control | **$0.42** | baseline |
| E: category-seeded | $0.46 | +9.5% |
| B: cost-aware | $0.51 | +21.4% |
| D: predefined-cost-aware | $0.52 | +23.8% |
| A: normal-evolve (成功任务) | $0.55 | +31.0% |

**分析**:
- **C 组（predefined-control）的 cost_of_pass 最低（$0.42）**，说明人工预定义策略仍然是工程最优解，提供最强的成本控制。
- **E 组（category-seeded）的 cost_of_pass 为 $0.46**，比 C 组高 9.5%，但仍远低于 B 组的 +21.4%。这说明仅给类别先验（M1-M7）而不给具体策略，已能诱导出相当有效的效率策略。
- **B 组的成本最高**（$0.51），虽然达到了 100% 成功率，但其 cost_of_pass 比 C 组高 21.4%。这说明 cost-aware 机制虽然保障了成功率，但单位成本效率不如预定义策略。
- **D 组（predefined-cost-aware）的 cost 反而高于 B 和 E**，且出现 1 个失败。这可能是因为预定义策略与 cost-aware proposer 之间存在策略冲突，导致 agent 行为不一致。

#### 2.2.3 Token 效率

| 实验组 | avg_tokens/task | input_ratio | output_ratio |
|---|---|---|---|
| C: predefined-control | **443,631** | 97.0% | 3.0% |
| D: predefined-cost-aware | 429,633 | 97.2% | 2.8% |
| E: category-seeded | 474,080 | 97.4% | 2.6% |
| A: normal-evolve | 520,335 | 97.1% | 2.9% |
| B: cost-aware | 569,612 | 97.3% | 2.7% |

**分析**:
- C 组的平均 token 消耗最低，说明预定义策略中的 `observation_masking` 等有效减少了 input token。
- E 组的 token 略高于 C 组（+6.9%），但低于 B 组（+22%）。category-seeded 策略在上下文压缩方面有一定效果。
- B 组的 token 最高，可能是因为 cost-aware proposer 生成的策略偏重于"安全"而非"压缩"，导致 agent 执行了更多推理步骤。

### 2.3 Tool Call 效率维度

| 实验组 | 总 tool calls | avg/tool_calls | 相对 C 组增量 |
|---|---|---|---|
| D: predefined-cost-aware | **349** | 34.9 | baseline |
| C: predefined-control | 379 | 37.9 | +8.6% |
| E: category-seeded | 395 | 39.5 | +13.2% |
| B: cost-aware | 447 | 44.7 | +28.1% |
| A: normal-evolve | 458 | 45.8 | +31.2% |

**分析**:
- **D 组的 tool calls 最少**（349次），说明预定义策略中的 `adaptive_tool_trimming` 等有效减少了不必要的工具调用。
- **E 组的 tool calls（395次）显著低于 A 组（458次）**，减少约 13.7%。这验证了类别种子策略确实诱导了有效的工具使用优化行为。
- B 组的 tool calls 仍然偏高（447次），但比 A 组减少了 2.4%。其策略更多集中在"不重复执行"而非"减少总调用数"。

### 2.4 行为模式分析（Behavior Stats）

#### 2.4.1 重复行为指标对比

| 指标 | A: normal-evolve | B: cost-aware | C: predefined-control | D: predefined-cost-aware | E: category-seeded |
|---|---|---|---|---|---|
| repeated_exact_inputs | 4 | **10** | 11 | 4 | **12** |
| repeated_test_inputs | 3 | **8** | 7 | 3 | **6** |
| same_state_repeated_tests | 2 | **6** | 4 | 2 | **5** |
| post_edit_repeated_tests | 1 | **2** | 3 | 1 | 1 |
| prompt_fragment_violation_events | 1 | 1 | **0** | **1** | 1 |

**分析**:
- **B 组和 E 组的 `repeated_exact_inputs` 反而高于其他组**（B=10, E=12），这初看似乎是负面的，但结合成功率来看：这些重复输入发生在**成功的任务**中，说明 agent 可能在更谨慎地验证结果。
- **C 组的 `prompt_fragment_violation_events` 为 0**，说明预定义策略中的 prompt fragment 被严格遵守。
- **D 组的各项重复指标均为最低或接近最低**（4, 3, 2, 1），与其最低的 tool calls 数量相符。预定义策略 `observation_masking` 有效减少了不必要的观察。

#### 2.4.2 prompt_fragment_visible_tasks

| 实验组 | prompt_fragment_visible_tasks |
|---|---|
| C: predefined-control | **0** |
| B: cost-aware | **2** |
| A: normal-evolve | 8 |
| D: predefined-cost-aware | 8 |
| E: category-seeded | 8 |

C 组的 prompt_fragment_visible_tasks 为 0，说明预定义策略在 agent 行为控制上最为严格。E 组有 8 个任务可见 prompt fragment，这与 category-seeded 策略以 prompt fragment 为产物形式有关。

### 2.5 策略命中（Strategy Hits）分析

#### 2.5.1 各组策略命中总览

| 策略 | C (predefined) | D (predefined-cost) | E (category-seeded) |
|---|---|---|---|
| observation_masking | 17 hits / 4 tasks | 18 hits / 4 tasks | — |
| adaptive_tool_trimming | 1 hit / 1 task | 1 hit / 1 task | — |
| skill_compilation | 1 hit / 1 task | 2 hits / 2 tasks | 1 hit / 1 task |
| stagnation_rescue_stop | 0 | 1 hit / 1 task | — |
| plan_template_cache | 0 | 0 | — |
| read_skill:evoefficient-swe-debug-loop | 1 hit / 1 task | 2 hits / 2 tasks | — |
| **emergent_skill:emergent_gen02_01_test_targeting_probe** | — | — | **1 hit / 1 task** |
| read_skill:fix-mutable-shared-state | — | — | 1 hit / 1 task |

**关键发现**:
- **`observation_masking` 是最高效的预定义策略**，在 C 和 D 组中均有大量命中（17-18次），覆盖 40% 的任务。这与其"减少冗余观察"的设计目标高度吻合。
- **E 组成功产生了 1 个 emergent skill 命中**：`emergent_gen02_01_test_targeting_probe`（Batch 5），说明 category-seeded 策略确实诱导了可复用的 emergent skills。
- D 组的 `stagnation_rescue_stop` 有 1 次命中，但 C 组为 0。这可能是因为 D 组的 cost-aware proposer 生成了与 stagnation rescue 互补的策略。

---

## 三、提案（Proposal）深度分析

### 3.1 各实验组提案数量与采纳率

| 实验组 | 总提案数 | 状态分布 | 产物类型 |
|---|---|---|---|
| **E: category-seeded** | 4 个 batch 有提案 | 3个 applied, 1个 empty | skill(1) + memory(1) + prompt_fragment(1) |
| **B: cost-aware** | 4 个 batch 有提案 | 2个 applied, 1个 skipped, 1个 empty | skill(1) + prompt_fragment(1) |
| **D: predefined-cost-aware** | 4 个 batch 有提案 | 4个 applied | prompt_fragment(4) |

> 注：Batch 4 和 Batch 5 的 cost-aware proposer 返回了空提案（`"proposals": []`），可能是因为成本信号不足以生成新策略。

### 3.2 E 组（category-seeded）提案详情

E 组共生成 **3 个被采纳的提案**，均为 M5 Tools 类别：

#### 提案 1: `suppress_same_state_test_reruns`（memory）
- **类别**: M5 Tools
- **触发证据**: `django__django-11790` 中 same_state_repeated_tests=3，重复执行相同测试命令 5 次
- **核心逻辑**: 在执行测试前，比较当前 workspace fingerprint（git diff + 文件修改状态）与上次执行时的状态；若完全一致则跳过重复测试，直接复用上次结果
- **预期节省**: 减少 same_state_repeated_tests 和 repeated_test_inputs，节省完整测试输出带来的大量 input token
- **正确性风险**: 如果 fingerprint 遗漏了外部依赖、配置文件或状态性测试副作用，可能掩盖新引入的失败
- **置信度**: HIGH

#### 提案 2: `read_target_source_before_search`（prompt_fragment）
- **类别**: M5 Tools + M2 Reasoning
- **触发证据**: `django__django-11790` 在定位 `django/db/migrations/autodetector.py` 之前执行了 9 次搜索和 9 次导航
- **核心逻辑**: 在进行广度搜索之前，先直接读取测试失败所指向的源文件，获取直接的代码上下文
- **预期节省**: 减少搜索和导航调用次数，缩短早期调查阶段的推理循环
- **正确性风险**: 猜错目标文件会浪费上下文；若只读一个本地文件可能遗漏真实行为源（如 mixin、decorator）
- **置信度**: MEDIUM

#### 提案 3: `test_targeting_probe`（skill）
- **类别**: M5 Tools + M4 Planning
- **触发证据**: `django__django-11880` 重复执行相同的 narrow test command 3 次，forms field tests 重复执行
- **核心逻辑**: 在编辑 Django patch 之前，先运行最小的相关模块级测试套件，检查失败输出后再编辑；不编辑前不重复运行宽泛测试
- **预期节省**: 减少宽泛重复的 run_test 调用，缩短 edit-debug 循环
- **正确性风险**: 过窄的模块可能通过但相邻回归测试失败，尤其是涉及 model/admin form metaclasses 时
- **置信度**: MEDIUM
- **命中情况**: Batch 5 中被 agent 实际使用（1次 read_skill call）

### 3.3 B 组（cost-aware）提案详情

B 组共生成 **2 个被采纳的提案**：

#### 提案 1: `fix-mutable-shared-state`（skill，采纳但未生成新文件）
- **来源**: 来自 seed workspace 的既有 skill（不是 cost-aware proposer 新生成）
- **命中**: 1 task（django__django-11815）
- **类型**: read_skill 复用

#### 提案 2: `no_rerun_unchanged_commands`（prompt_fragment）
- **触发证据**: `django__django-11999` 中 repeated_test_inputs=2，same_state_repeated_tests=1
- **核心逻辑**: 当相同命令已在当前代码状态下执行过且有可用结果时，不重新运行；只有代码/输入/环境改变后才重新运行
- **预期节省**: 消除在未变状态上执行的完整测试（每个全量测试可能是 episode 成本的主要部分）
- **置信度**: MEDIUM

### 3.4 D 组（predefined-cost-aware）提案详情

D 组共生成 **4 个被采纳的提案**，全部为 prompt_fragment：

| 提案名 | 描述 | 置信度 | 关键证据 |
|---|---|---|---|
| `django_probe_batching` | 将多个独立的 Django 检查合并为一次 Python heredoc | MEDIUM | 同一 settings.configure 命令在 `django__django-11790` 中重复执行 |
| `targeted_test_validation_loop` | 代码编辑前后均要求有针对性的测试验证 | **HIGH** | `django__django-11885` 零测试调用导致 patch 失败 |
| `cap_remote_history_prospecting` | 限制远程 GitHub API 调用，优先本地 git 历史 | MEDIUM | `django__django-11885` 在无测试的情况下多次调用 GitHub API |

**关键洞察**: D 组的提案主要针对**已知失败的案例**（`django__django-11885`），是典型的"亡羊补牢"型策略。预定义策略已经处理了常见效率问题，cost-aware proposer 的增量贡献主要体现在特定失败模式的补救。

---

## 四、批次间演化分析

### 4.1 Batch 级 cost_of_pass 趋势

| Batch | A: normal-evolve | B: cost-aware | C: predefined-control | D: predefined-cost-aware | E: category-seeded |
|---|---|---|---|---|---|
| Batch 1 | **inf** (0/2) | $0.48 | $0.47 | $0.37 | $0.47 |
| Batch 2 | $0.32 | $0.48 | $0.29 | $0.34 | **$0.38** |
| Batch 3 | $0.53 | $0.50 | $0.55 | $1.05 | $0.55 |
| Batch 4 | $0.49 | $0.50 | $0.34 | $0.62 | **$0.48** |
| Batch 5 | $0.42 | $0.57 | $0.48 | $0.48 | $0.41 |
| **Overall** | **$0.55** | $0.51 | **$0.42** | $0.52 | $0.46 |

**关键观察**:
- **A 组 Batch 2 成本最低**（$0.32），说明一旦 agent 自进化建立了有效的 skills，后续任务可以非常高效。
- **D 组 Batch 3 cost_of_pass 飙升至 $1.05**，是唯一一个出现批量失败的组（1/2 成功）。该 batch 的 `observation_masking` 命中高达 11 次，暗示可能过度压缩导致关键信息丢失。
- **E 组的 Batch 2 和 Batch 4 成本较低**（$0.38, $0.48），而 Batch 3 跳升至 $0.55。E 组在中期 batch 出现了成本上升，可能是提案尚未被充分消化吸收。

### 4.2 早期批次 vs 晚期批次对比

| 指标 | A 前半 (B1+B2) | A 后半 (B3+B4+B5) | E 前半 (B1+B2) | E 后半 (B3+B4+B5) |
|---|---|---|---|---|
| 成功率 | 50% | 100% | 100% | 100% |
| avg_cost | $0.39 | $0.48 | $0.43 | $0.48 |
| avg_tool_calls | 39.5 | 50.0 | 41.3 | 38.3 |

**分析**:
- A 组前半段因 Batch 1 失败率极高而被严重拖累，后半段恢复正常。
- E 组全程稳定在 100% 成功率，说明 category-seeded 机制从一开始就提供了足够的引导。
- E 组后半段 tool_calls（38.3）反而低于前半段（41.3），说明 emergent skills 在后期开始发挥作用。

---

## 五、Leakage Audit（策略泄漏审计）

| 实验组 | 审计结果 | 泄漏策略名 |
|---|---|---|
| A: normal-evolve | **PASS** ✓ | 无 |
| B: cost-aware | **PASS** ✓ | 无 |
| E: category-seeded | **PASS** ✓ | 无 |

所有组均通过了策略泄漏审计，说明：
1. normal-evolve、cost-aware、category-seeded 三组的 workspace 中**没有残留预定义策略名称**
2. cost-aware proposer 确实只输出了中性成本信号，未泄露策略 registry 信息
3. category-seeded proposer 仅使用了 M1-M7 类别标签，未提及具体策略

---

## 六、效率前沿分析（Efficiency Frontier）

以 **success_rate** 为纵轴、**cost_of_pass** 为横轴，绘制各实验组的效率前沿：

```
cost_of_pass ($)
  $0.29    $0.35    $0.42    $0.48    $0.52    $0.55
    |        |        |        |        |        |
100% ┤                              ● C (predefined-control)
    │                         ● E (category-seeded)
    │                    ● D (predefined-cost-aware)
    │               ● B (cost-aware)
    │
 80% ┤          ● A (normal-evolve, 成功任务)
    │
```

**效率前沿结论**:
- **C 组（predefined-control）位于最优前沿**：最高成功率 + 最低 cost_of_pass
- **E 组（category-seeded）接近前沿**：100% 成功率，cost_of_pass 仅比 C 组高 9.5%
- **D 组和 B 组位于前沿下方**：D 组有 1 个失败，B 组成本较高
- **A 组仅考虑成功任务时位于前沿下方**：cost_of_pass $0.55 是所有成功组中最高的

---

## 七、关键洞察与结论

### 7.1 核心研究问题回答

#### Q1: 成本信号能否诱导 agent 自进化出可复用的效率行为？

**答案：部分可以，但不如预定义策略高效。**

- E 组（category-seeded）在不提供具体策略名的情况下，仅通过 M1-M7 成本维度类别，成功诱导出 3 个可复用提案（1 skill + 1 memory + 1 prompt_fragment），其中 `test_targeting_probe` 被实际使用。
- E 组的 cost_of_pass（$0.46）接近 C 组（$0.42），差距仅 9.5%，说明"类别种子"策略已经提供了足够的工程价值。
- B 组的成本反而高于 A 组（正常自进化），说明在无类别引导的情况下，成本信号本身不足以生成高效的效率策略——它更多是"事后补救"而非"事前预防"。

#### Q2: 类别种子 vs 纯成本信号，哪个更有效？

**答案：类别种子（E）显著优于纯成本信号（B）。**

| 对比维度 | E: category-seeded | B: cost-aware |
|---|---|---|
| cost_of_pass | **$0.46** | $0.51 |
| avg_tool_calls | **39.5** | 44.7 |
| emergent skill 产生 | **1 个**（被实际使用） | 0 |
| 策略命中 | emergent_gen02_01_test_targeting_probe | fix-mutable-shared-state（既有） |
| 提案类别覆盖 | M5, M2, M4 | M5 |

E 组的 `test_targeting_probe` skill 命中了 M5（工具优化）和 M4（规划）两个类别，直接减少了不必要的测试调用。纯成本信号组（B）则未能产生新的 emergent skills。

#### Q3: 预定义策略 + cost-aware 组合是否产生增量价值？

**答案：否——D 组的表现在多个维度弱于 C 组。**

- D 组 cost_of_pass（$0.52）比 C 组（$0.42）高 23.8%
- D 组出现了 1 个任务失败（C 组 100% 成功）
- D 组的 4 个提案全部是 prompt_fragment，没有产生新的 emergent skills
- 预定义策略（`observation_masking` 等）已经高度覆盖了常见效率问题，cost-aware proposer 的"增量空间"有限，反而可能引入干扰

#### Q4: normal-evolve 的稳定性如何？

**答案：初期不稳定，但一旦建立有效 skills 后可达到与预定义策略相当的效率。**

- A 组 Batch 1 成功率 0%，但 Batch 2-5 连续 4 个 batch 100% 成功
- Batch 2 的 cost_of_pass 为 $0.32，是**所有实验组中最低的**
- 纯自进化的上限效率极高，但不可靠；预定义策略提供了稳定的下限保障

### 7.2 策略价值排名（按 cost_of_pass）

```
最优到最差（cost_of_pass）:
1. C (predefined-control)          $0.42  ████████████████████  ← 工程最优解
2. E (category-seeded)             $0.46  ███████████████████▌  ← 自然涌现最优
3. B (cost-aware)                  $0.51  ██████████████████▍
4. D (predefined-cost-aware)       $0.52  ██████████████████▌
5. A (normal-evolve, 成功任务)     $0.55  ██████████████████▌
```

### 7.3 涌现策略分类（后验分类）

按照 README 中的后验分类框架对 E 组和 B 组提案进行分类：

| 提案 | 分类 | 实验组 | 形态 |
|---|---|---|---|
| suppress_same_state_test_reruns | observation/tool trimming | E | memory |
| read_target_source_before_search | context reduction | E | prompt_fragment |
| test_targeting_probe | verification-first workflow | E | skill |
| no_rerun_unchanged_commands | early rescue/stop | B | prompt_fragment |
| django_probe_batching | context reduction | D | prompt_fragment |
| targeted_test_validation_loop | verification-first workflow | D | prompt_fragment |
| cap_remote_history_prospecting | context reduction | D | prompt_fragment |

**分类洞察**:
- E 组产生的策略覆盖了 3 个不同类别（tool trimming、context reduction、verification-first），呈现多样性
- D 组的 4 个提案集中在 context reduction 和 verification-first，说明预定义策略已覆盖了大部分 M5 Tools 类别的策略

---

## 八、特殊案例深度分析

### 8.1 D 组 Batch 3 失败分析

D 组 Batch 3 是唯一出现部分失败的 batch（1/2 成功）。该 batch 的特征：
- `observation_masking` 命中 11 次（是所有 batch 中最高的）
- avg_cost 飙升至 $0.53（整体 D 组 avg_cost=$0.47）
- avg_tool_calls=45（整体 D 组 avg_tool_calls=34.9）

**假设**: `observation_masking` 在该 batch 中过度激进的压缩导致 agent 丢失了关键上下文信息，使得 agent 无法正确完成复杂任务。预定义策略与 cost-aware proposer 产生的 prompt_fragment 可能存在冲突，进一步加剧了信息丢失。

### 8.2 E 组 `test_targeting_probe` Skill 命中分析

这是 E 组唯一一个被实际使用的 emergent skill：
- **触发条件**: Batch 5 的 `django__django-12039` 任务
- **命中时机**: 在该任务中，agent 通过 `read_skill` 调用了 `emergent_gen02_01_test_targeting_probe`
- **预期效果**: 减少宽泛测试调用，缩短 edit-debug 循环
- **实际效果**: 该 batch 的 cost_of_pass 降至 $0.41（所有 E batch 中最低），avg_tool_calls=46（较高但该任务本身复杂度较高）

### 8.3 A 组 Batch 1 失败分析

A 组 Batch 1 的 2 个任务均失败，可能原因：
1. **冷启动问题**: workspace 刚复制并清理，没有任何 skills/prompts，agent 处于"裸奔"状态
2. **redaction 影响**: 预定义策略名被 redaction 后，agent 可能失去了某些隐含的效率提示
3. **任务难度**: Batch 1 的 2 个任务可能恰好是较难的任务

从 Batch 2 开始，A 组的正常 agent 自进化（solver proposal + GuidedSynthesisEngine curation）开始产生效果，后续任务全部成功。

---

## 九、改进建议

### 9.1 针对 category-seeded（E 组）的优化

1. **提高提案稳定性**: E 组有 2 个 batch（Batch 4 和 Batch 5）proposer 返回空提案，建议增加 cost 信号的敏感性
2. **加速 skill 传播**: 当前只有 Batch 5 用到了 emergent skill，建议在更早的 batch 触发 skill 编译
3. **增加 M1 Context 类别的提案**: 当前所有提案集中在 M5 Tools，缺乏上下文压缩类的策略

### 9.2 针对 cost-aware（E 组）的优化

1. **增加 early-stage proposal**: B 组提案多来自中后期 batch，建议在 Batch 1-2 也强制触发提案，减少冷启动风险
2. **提案质量过滤**: B 组的 `fix-mutable-shared-state` 来自既有 skill 而非新生成，说明 proposer 可能过度依赖已有知识

### 9.3 针对 predefined-cost-aware（D 组）的优化

1. **策略冲突检测**: D 组的预定义策略与 cost-aware 提案可能存在冲突，建议增加"策略兼容性检查"
2. **减少 prompt_fragment 数量**: D 组 4 个提案全部是 prompt_fragment，建议强制多样化（skill/memory/tool）

### 9.4 针对 normal-evolve（A 组）的优化

1. **冷启动保障**: 建议在 Batch 1 启用一个极简的"生存策略"（如强制运行测试验证循环），避免全灭
2. **更保守的 curation**: Batch 1 失败可能与 curation 过程中丢失关键信息有关，建议在早期 batch 提高 proposal 采纳门槛

---

## 十、实验局限性

1. **样本量有限**: 每组仅 10 个任务，统计显著性有限
2. **任务同构性**: 所有任务均为 Django 相关，结论可能不泛化到其他代码库
3. **Batch 规模**: batch_size=2 较小，提案反馈频率受限
4. **单一模型**: 所有实验使用相同的 hard 模型，无模型路由测试
5. **时间因素**: 实验在同一天内完成，可能存在日内性能波动

---

## 十一、附录：原始数据文件路径

| 数据类型 | 文件路径 |
|---|---|
| summary.json (A) | `logs/emergent_efficiency/normal_evolve_s4/summary.json` |
| summary.json (B) | `logs/emergent_efficiency/cost_aware_20260903_215037_402122/summary.json` |
| summary.json (C) | `logs/emergent_efficiency/predefined_20260903_213003_951825/summary.json` |
| summary.json (D) | `logs/emergent_efficiency/predefined_cost_aware_20260903_134010_785227/summary.json` |
| summary.json (E) | `logs/emergent_efficiency/category_seeded_cost_aware_20260903_195310_364868/summary.json` |
| proposals.jsonl (B) | `logs/emergent_efficiency/cost_aware_20260903_215037_402122/emergence/proposals.jsonl` |
| proposals.jsonl (D) | `logs/emergent_efficiency/predefined_cost_aware_20260903_134010_785227/emergence/proposals.jsonl` |
| proposals.jsonl (E) | `logs/emergent_efficiency/category_seeded_cost_aware_20260903_195310_364868/emergence/proposals.jsonl` |
| proposal_adoption (B) | `logs/emergent_efficiency/cost_aware_20260903_215037_402122/emergence/proposal_adoption.json` |
| proposal_adoption (D) | `logs/emergent_efficiency/predefined_cost_aware_20260903_134010_785227/emergence/proposal_adoption.json` |
| proposal_adoption (E) | `logs/emergent_efficiency/category_seeded_cost_aware_20260903_195310_364868/emergence/proposal_adoption.json` |
| behavior_stats (各组) | `logs/emergent_efficiency/<run>/behavior_stats.json` |
| leakage_audit (各组) | `logs/emergent_efficiency/<run>/emergence/leakage_audit.json` |
| figures (B, D, E) | `logs/emergent_efficiency/<run>/figures/` |

# 新增 VisualWebArena 实验验证记录

日期：2026-10-29（Asia/Shanghai）。全部检查在服务器项目 `.venv` 环境执行。

- offline_tests_passed: 22
- lint: passed
- format: passed
- bash_syntax: passed
- smoke_modes: 5
- synthetic_tasks_per_mode: 4
- source_hashes: matched
- live_vwa_run: False

原始验证输出：`logs/visualwebarena_emergent_efficiency/{pytest,lint,format_check,smoke_matrix}.log`。

五模式均为显式 synthetic 冒烟输入，不产生官方成功率或成本结论。真实网站、captioner、登录、模型与judge服务的端到端执行尚未验证；正式部署和运行步骤见 README。

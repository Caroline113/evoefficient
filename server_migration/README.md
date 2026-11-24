# GitHub 备份与服务器迁移

当前项目路径：`/home/azureuser/zj/a-evolve-new7/a-evolve-new`。

建议保存两类产物：GitHub 分支保存代码、环境依赖清单、少量报告；私有 Release 附件保存完整轨迹、源码迁移包和 Conda 环境包。普通 Git 单文件不能超过 100 MiB，Release 单个附件必须小于 2 GiB。不要把整个 `.venv` 作为 Git 文件提交。

官方说明：
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- https://docs.python.org/3/library/venv.html
- https://conda.github.io/conda-pack/

## 当前实际环境

| 环境 | Python | 目录大小（未压缩） | 用途 |
|---|---|---:|---|
| 项目 `.venv` | 3.11.15 | 约 0.99 GB | 实验控制、进化、统计、测试 |
| `aevolve` Conda | 3.11.15 | 约 1.02 GB | Conda 基础实验环境 |
| Harbor `.venv` | 3.13.17 | 约 8.64 GB | Harbor 与实际任务执行 |

只迁移 `aevolve` 不足以恢复当前实验；Harbor 有单独的 Python 和依赖。源系统是 Linux x86_64 / glibc 2.35。Conda 环境包适合兼容 Linux 系统，不能保证跨操作系统或任意旧 glibc 使用；GPU 驱动和 Docker 也不在 Python 包里。

## 环境快照

```bash
source .venv/bin/activate
python server_migration/export_environments.py --name 20261107
```

快照在 `server_migration/exports/20261107/`，可以随代码版本管理：

- 每个环境的 Python、平台、完整包版本。
- `requirements.lock.txt`：普通 pip 依赖固定版本。
- `local_sources.json`：本地/可编辑安装的包，单独记录来源，恢复时重新安装。
- `aevolve/environment.yml`：无构建号 Conda 依赖。
- `aevolve/environment-history.yml`：较宽松、用于换平台重建的直接依赖。
- `aevolve/conda-linux-64-explicit.txt`：同平台更严格的 Conda 包地址。
- `SHA256.json`：快照完整性校验。

Conda YAML 不包含原环境变量，也不混入带旧机器路径的 pip 子段；pip 包从单独 lock 恢复。本地源依赖不会假装成 PyPI 上的同名包。原始导出保存在 `private/`，该目录被本目录 `.gitignore` 排除；不应提交到 GitHub。

## 生成完整迁移包

```bash
python server_migration/create_backup.py --name 20261107-final \
  --data logs --data agent_evolve/benchmarks --pack-conda
```

输出在 `server_migration/archives/20261107-final/`，该目录不会进入普通 Git：

1. `source-repositories.tar.gz`：项目与 Harbor 两个仓库的历史 bundle、当前 tracked 修改/删除 patch、未跟踪文件，以及补充的源码快照。普通 `git bundle` 不包含未提交改动，所以这些部分必须一起保存。
2. `experiment-data.tar.gz`：完整实验日志与 benchmark 目录。
3. `aevolve-conda.tar.gz`：通过 conda-pack 打包的实际环境；未忽略 editable/missing-file 错误。若打包失败，会记录失败，不能当作环境已保存。
4. `manifest.json` 与 `SHA256SUMS`：文件大小、校验和、数据在打包过程中是否变化。

备份脚本不停止实验。若 `changed_during_backup` 非空，需在相关实验完成后重新备份；它不是文件系统原子快照。未显式选择的训练数据、模型、VM 镜像、Docker 镜像和外部链接目标不包含在这个数据包中，使用额外 `--data 项目内相对路径` 添加。特别大的独立模型/VM 应另存对象存储或磁盘副本。

源码补充包覆盖常见文本源文件且单文件小于 20 MiB；排除虚拟环境、日志、数据集和私有备份目录。主代码历史、tracked patch 和 untracked 包一起用于恢复，不能只依赖补充包。

## GitHub 账号与仓库

本次已切换为 `Caroline113`，确认对私有仓库 `DeepSoftwareAnalytics/EvolveEfficient` 有管理员权限。旧账号 `gnohgnailoug` 无访问权限。新服务器上需要重新登录；当前服务器的登录不会随代码迁移。

```bash
gh auth login --hostname github.com --web --scopes repo
gh api user --jq .login
gh repo view DeepSoftwareAnalytics/EvolveEfficient --json nameWithOwner,isPrivate,viewerPermission
```

登录成功并确认仓库权限后，可以使用 HTTPS 配合 GitHub CLI 凭据：

```bash
git config --local --replace-all credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper '!gh auth git-credential'
git remote set-url origin https://github.com/DeepSoftwareAnalytics/EvolveEfficient.git
```

若 `GH_TOKEN` / `GITHUB_TOKEN` 环境变量覆盖了登录账号，需要在当前 shell 中移除旧变量后重新检查账号，不要把 Token 写进仓库 URL。

先为当前服务器状态建立备份分支，确认暂存内容后提交，避免直接覆盖 main：

```bash
git switch -c backup/server-20261107
git status --short
git add -A
git diff --cached --stat
git commit -m 'Back up experiment code and migration environment manifests'
git push -u origin HEAD
```

现有 `.gitignore` 忽略 `/logs/`、虚拟环境和部分旧代码。大数据使用迁移包；被忽略但确实需要版本管理的源码请检查后按具体路径添加，不要整仓库 `git add -f .`。运行中的真实 `.env`、SSH 私钥和 GitHub 凭据需要单独通过安全方式迁移，不提交到仓库。

## 私有 Release 保存大文件

先确认 `isPrivate=true`，且当前账号有写权限。仓库如果公开，不应上传包含原始实验输入/输出和环境内容的迁移包。

```bash
gh release create migration-20261107 --draft \
  --target backup/server-20261107 \
  --title 'Server migration snapshot 2026-11-07' \
  --notes 'Code snapshot, experiment data, and aevolve environment backup.'

gh release upload migration-20261107 \
  server_migration/archives/20261107-final/source-repositories.tar.gz \
  server_migration/archives/20261107-final/experiment-data.tar.gz \
  server_migration/archives/20261107-final/aevolve-conda.tar.gz \
  server_migration/archives/20261107-final/SHA256SUMS \
  server_migration/archives/20261107-final/manifest.json
```

先保存为草稿，检查后再发布。若文件超过 2 GiB，需要分卷或改用对象存储；不要强塞进 Git。Conda 环境内可能含运行配置，上传前同样只使用你有权限的私有备份目的地。

`publish_backup.py` 可先生成审查计划，再按计划上传：

```bash
python server_migration/publish_backup.py --snapshot 20261107-final
python server_migration/publish_backup.py --snapshot 20261107-final --execute
```

该脚本使用独立临时 Git index 生成备份提交并推送新分支，不切换当前分支，不改动真实暂存区，不更新 main。它检查私有性、写权限、附件哈希、文件大小和常见凭据特征；审查后源码变化会要求重新生成计划。成功后核对 Release 中附件大小并记录分支/Release URL。检测不等于所有内容都已人工脱敏，原始数据只放私有目的地。

**服务器上的本地包不算异地备份。**必须确认 GitHub 上传成功，或另下载到本地/其他存储，才能删除旧服务器。

## 在新服务器恢复

### 1. 取回代码与数据

```bash
git clone --branch backup/server-20261107 \
  https://github.com/DeepSoftwareAnalytics/EvolveEfficient.git a-evolve-new
gh release download migration-20261107 \
  --repo DeepSoftwareAnalytics/EvolveEfficient --dir migration-download
cd migration-download
sha256sum -c SHA256SUMS
tar -xzf source-repositories.tar.gz
```

若要恢复包含未提交改动的原始仓库状态，用 bundle 克隆到新目录，应用 patch，再解包未跟踪/补充源码。以下对项目和 Harbor 各执行一次：

```bash
git clone sources/project/history.bundle restored-project
cd restored-project
git apply --binary ../sources/project/tracked-worktree.patch
tar -xzf ../sources/project/untracked.tar.gz
tar -xzf ../sources/project/working-tree-sources.tar.gz
```

patch 为空时不必执行 `git apply`。tracked 文件的删除由 patch 表达，不能省略。恢复 Harbor 时将 `project` 换为 `harbor`。确保在空的新目录恢复，不覆盖已有工作区。

### 2. 快速恢复 aevolve Conda

```bash
mkdir -p "$HOME/envs/aevolve"
tar -xzf aevolve-conda.tar.gz -C "$HOME/envs/aevolve"
source "$HOME/envs/aevolve/bin/activate"
conda-unpack
python --version
```

本地可编辑依赖仍需按 `local_sources.json` 重新指向新源码。若不使用环境包，在安装 Conda 后可由 `environment.yml` 重建，再安装 aevolve 的 pip lock 和本地源码；同平台也可使用 explicit 文件。`--from-history` 更适合跨平台重解，但不会精确固定所有依赖。

### 3. 重建两个 venv

Python venv 包含绝对路径和解释器引用，不保证搬走后可用。项目环境用恢复的 Python 3.11.15 创建，Harbor 另外准备 Python 3.13.17：

```bash
python server_migration/restore_venv.py 
  --snapshot server_migration/exports/20261107 --environment project_venv 
  --python "$HOME/envs/aevolve/bin/python" 
  --destination "$PWD/.venv" --project-root "$PWD" 
  --harbor-root "$HOME/zj/harbor" --execute
```

对 Harbor 再执行一次，将 environment 改为 `harbor_venv`，python 改为新服务器实际的 3.13.17 路径，destination 改为新的 Harbor `.venv` 路径。程序默认只打印计划，加 `--execute` 才安装；目标环境必须尚不存在。本地额外依赖用 `--source 包名=/新路径` 映射。

有网重建需要包源仍提供这些版本，CUDA 版 PyTorch 等可能需要原来的额外 wheel 索引。为进一步加速，可以提前用相同 Python/平台下载 wheelhouse：

```bash
.venv/bin/python -m pip download \
  -r server_migration/exports/20261107/project_venv/requirements.lock.txt \
  -d /path/to/wheelhouse-project
```

Harbor 的 wheelhouse 必须用 Python 3.13 环境单独准备，不能和 3.11 共用；也需收齐本地源码的构建依赖（如 hatchling/uv_build）。恢复时加 `--wheelhouse /path/to/wheels` 可关闭 PyPI 依赖下载。本次未自动下载数 GB 的 wheelhouse。

### 4. 恢复实验数据并验证

在新项目根目录解包 `experiment-data.tar.gz`。部分旧结果记录了旧服务器绝对路径；文件搬迁后，需要将这些字段映射到新目录后再做原始轨迹分析，不能把路径失效当作零用量。模型端点配置需单独重新设置。

```bash
.venv/bin/python -m pip check
/path/to/harbor/.venv/bin/python -m pip check
.venv/bin/python -m pytest examples/tb_examples/emergent_efficiency_train_eval/tests -q
docker --version
nvidia-smi
```

Python 包恢复不等于 Docker 镜像、GPU 驱动、模型服务也已恢复。验证通过后先跑小规模任务，再迁移长实验。

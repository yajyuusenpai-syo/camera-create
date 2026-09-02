# 三模型隔离环境安装

本文面向 Linux + NVIDIA GPU 服务器。Pi3X、MoGe-3 和 VIPE 分别安装到独立
Python 3.10 环境；`camera-create` 主控程序安装在 Pi3X 环境中。公司服务器推荐
标准 `venv`，只能使用 Conda 的测试服务器使用 Conda prefix 备选方案。两种脚本
生成相同的 `.envs` 布局，CLI 调用方式不变。上游源码、环境和权重均被
`.gitignore` 排除，不会上传到 camera-create 仓库。

## 1. 系统准备

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential ninja-build \
  python3.10 python3.10-venv python3.10-dev
nvidia-smi
nvcc --version
```

VIPE 会编译 CUDA 扩展，只有驱动而没有 CUDA toolkit/nvcc 的服务器不能完成其安装。
先根据 `nvidia-smi` 和 `nvcc --version` 确定可用 CUDA，再选择 PyTorch wheel。

## 2. 获取 camera-create

服务器不保存 GitHub token 时，可在有身份的电脑下载压缩包并用 `scp` 上传；或者给
服务器配置只读 deploy key。仓库公开后也可直接使用 HTTPS：

```bash
git clone https://github.com/yajyuusenpai-syo/camera-create.git
cd camera-create
```

## 3. 分别克隆三个上游仓库

```bash
bash scripts/clone_models.sh
cat third_party/SOURCE_VERSIONS.txt
```

结果如下：

```text
camera-create/third_party/
├── Pi3/
├── MoGe/
├── vipe/
└── SOURCE_VERSIONS.txt
```

脚本默认 Pi3=`main`、MoGe=`main`（包含 MoGe-3）、VIPE=`v1.2.0`。为了复现测试
结果，可用 tag 或 commit 覆盖版本：

```bash
PI3_REF=<tested-commit> \
MOGE_REF=<tested-moge3-commit> \
VIPE_REF=v1.2.0 \
bash scripts/clone_models.sh
```

脚本采用 detached checkout，避免误把上游修改提交进本项目。每次部署都应保存
`SOURCE_VERSIONS.txt` 到实验记录中。

## 4. 公司服务器：标准 venv 方案

默认 wheel 组合为 Pi3X/cu124、MoGe-3/cu130、VIPE/cu128：

```bash
bash scripts/setup_three_envs.sh
```

生成：

```text
camera-create/.envs/
├── pi3x/   # torch 2.5.1 + torchvision 0.20.1 + numpy 1.26.4
├── moge3/  # numpy >=2 + MoGe-3 + Triton/FlexGEMM
└── vipe/   # VIPE v1.2.0 + CUDA Torch + 编译后的 CUDA 扩展
```

默认 CUDA wheel 不一定适合所有服务器。安装前可独立覆盖：

```bash
PI3_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
MOGE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
VIPE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
bash scripts/setup_three_envs.sh
```

MoGe-3 官方当前默认 CUDA 13.0，但也说明可重装为 cu128。选择 cu128 时仍保持它与
Pi3X 环境隔离，解决 NumPy 2.x 与 1.26.4 冲突。不要把三个环境的 `site-packages`
加入同一个 `PYTHONPATH`。

如需将环境放到高速数据盘：

```bash
ENV_ROOT=/data/envs/camera-create \
SOURCE_ROOT=/data/src/camera-create-models \
bash scripts/setup_three_envs.sh
```

`SOURCE_ROOT` 在克隆与安装两个脚本中必须保持一致。

## 5. 测试服务器：Conda 备选方案

不要先运行 `setup_three_envs.sh`；直接使用 Conda 脚本创建同样的三个隔离路径：

```bash
bash scripts/clone_models.sh
bash scripts/setup_three_conda_envs.sh
```

脚本使用 `conda create --prefix`，不会污染 Conda 的 base 环境：

```text
camera-create/.envs/
├── pi3x/   # 独立 Conda prefix
├── moge3/  # 独立 Conda prefix
└── vipe/   # 独立 Conda prefix
```

如果 `conda` 不在 `PATH`：

```bash
CONDA_COMMAND=/opt/conda/bin/conda \
bash scripts/setup_three_conda_envs.sh
```

CUDA wheel 同样可以分别覆盖。例如测试服务器统一使用 cu128 时：

```bash
MOGE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
VIPE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
bash scripts/setup_three_conda_envs.sh
```

Pi3X 仍建议保留 Torch 2.5.1/cu124，因为这是其官方固定版本。只有确认服务器驱动不
支持 cu124 时，才修改 `PI3_TORCH_INDEX_URL`，并重新执行真实模型验证。

不需要 `conda activate`，直接使用 prefix 中的解释器最稳定：

```bash
.envs/pi3x/bin/python cli.py --help
.envs/moge3/bin/python -c "from moge.model.v3 import MoGeModel; print('OK')"
.envs/vipe/bin/vipe --help
```

如需手动进入环境：

```bash
conda activate "$(pwd)/.envs/pi3x"
conda deactivate
```

注意：`setup_three_envs.sh` 与 `setup_three_conda_envs.sh` 不能对同一个已存在的
`.envs` 混用。如果需要切换方案，应使用新的 `ENV_ROOT`，例如：

```bash
ENV_ROOT=/data/camera-conda-envs bash scripts/setup_three_conda_envs.sh
```

此时运行 CLI 必须显式传入三个绝对路径。

## 6. 放置 checkpoint

```text
camera-create/ckpt/
├── pi3x/   # Pi3X snapshot/checkpoint
├── moge3/  # Ruicheng/moge-3-vitl snapshot/checkpoint
└── vipe/   # VIPE、GeoCalib 和 Torch/Hugging Face cache
```

权重不得提交到 Git。MoGe-3 不像旧版默认模型那样可省略权重参数，worker 必须显式
指向 `ckpt/moge3` 中的 checkpoint。若模型页面需要接受许可，先在个人电脑完成授权，
再下载并 `scp -r` 到服务器，服务器无需保存 Hugging Face token。

## 7. 验证隔离状态

权重放置前只检查软件环境：

```bash
.envs/pi3x/bin/python scripts/check_three_envs.py \
  --env-root .envs --project-root . --skip-checkpoints
```

权重放置后完整检查：

```bash
.envs/pi3x/bin/python scripts/check_three_envs.py \
  --env-root .envs --project-root .
```

该脚本分别启动三个解释器，不会在同一进程导入三套 Torch/NumPy。三项均应显示
`cuda: true`；Pi3X 应显示 NumPy 1.26.4，MoGe-3 应显示 NumPy 2.x。

## 8. 激活和排错

手动进入某个环境：

```bash
source .envs/pi3x/bin/activate
deactivate
source .envs/moge3/bin/activate
deactivate
source .envs/vipe/bin/activate
```

常见错误：

- `No module named venv`：安装 `python3.10-venv`。
- `torch.cuda.is_available() == False`：wheel、驱动或 GPU 容器映射不匹配。
- VIPE 安装时报 `torch.version.cuda is None`：误装了 CPU Torch。
- VIPE 编译找不到 `nvcc`：安装匹配的 CUDA toolkit，并设置正确 `CUDA_HOME`。
- MoGe-3 无法加载 FlexGEMM/Triton：确认 Linux、NVIDIA GPU 和所选 Torch CUDA
  wheel 兼容；MoGe-3 不支持 macOS。
- `numpy` 冲突：确认命令使用的是 `.envs/<model>/bin/python`，不要使用裸 `pip`。
- Conda 求解很慢：脚本只用 Conda 创建 Python 3.10 prefix，模型依赖仍由各 prefix
  的 `python -m pip` 安装；可安装 `mamba`，但不是必需。
- 安装停在 `Requirement already satisfied: packaging...` 后长时间没有输出：旧版
  `conda run` 正在捕获后续 PyTorch 下载输出，看起来像卡住。新版脚本使用
  `--no-capture-output` 实时显示进度。可在另一个终端用
  `pgrep -af 'conda|pip|python'` 确认进程；若确实无进程，再重新运行脚本。
- PyTorch 安装报 `No matching distribution found for numpy`：旧脚本只配置了
  PyTorch CUDA index，而该索引不托管 NumPy。新版同时增加官方 PyPI 作为依赖
  回退源。无需删除环境；更新代码后重新运行同一个安装脚本，pip 会复用缓存。
- `CondaError: Run 'conda init'`：脚本使用 `conda run --prefix`，正常情况下无需
  `conda init`；检查 `CONDA_COMMAND` 是否指向真实的 Conda 可执行文件。
- 安装日志出现与本项目无关的 `deepfilternet`、`evo` 或系统 `matplotlib` 冲突：
  表示 base/user-site 包通过 `PYTHONPATH` 或 `~/.local` 泄漏。新版脚本会自动清除
  `PYTHONPATH/PYTHONHOME/PIP_USER` 并设置 `PYTHONNOUSERSITE=1`。旧终端可先执行：

  ```bash
  unset PYTHONPATH PYTHONHOME PIP_USER
  export PYTHONNOUSERSITE=1
  ENV_ROOT="$PWD/.conda-envs-clean" bash scripts/setup_three_conda_envs.sh
  ```

  不要为了消除提示而把 `deepfilternet`、`evo` 安装进模型环境。

## 9. 当前端到端状态

主 CLI 已分别调用以下程序：

```text
.envs/pi3x/bin/python  scripts/run_pi3x_worker.py
.envs/moge3/bin/python scripts/run_moge3_worker.py
.envs/vipe/bin/vipe    infer ...
```

调用示例：

```bash
.envs/pi3x/bin/python cli.py \
  --input /data/input.mp4 \
  --output /data/camera-result \
  --pi3x-python .envs/pi3x/bin/python \
  --moge3-python .envs/moge3/bin/python \
  --vipe-command .envs/vipe/bin/vipe
```

主进程依次运行两个深度 worker，确保同一时刻只有一个深度模型占用 GPU；worker
退出后才加载 NPZ 并融合。隔离逻辑和缓存契约已由无模型测试覆盖，但真实模型、
显存峰值、VIPE CUDA 扩展以及最终 metric 精度仍需要在目标服务器验证。

# 三模型隔离环境安装

本文面向 Linux + NVIDIA GPU 服务器。Pi3X、MoGe-3 和 VIPE 分别安装到独立
Python 3.10 `venv`；`camera-create` 主控程序安装在 Pi3X 环境中。上游源码、虚拟
环境和权重均被 `.gitignore` 排除，不会上传到 camera-create 仓库。

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

## 4. 创建三个 Python 环境

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

## 5. 放置 checkpoint

```text
camera-create/ckpt/
├── pi3x/   # Pi3X snapshot/checkpoint
├── moge3/  # Ruicheng/moge-3-vitl snapshot/checkpoint
└── vipe/   # VIPE、GeoCalib 和 Torch/Hugging Face cache
```

权重不得提交到 Git。MoGe-3 不像旧版默认模型那样可省略权重参数，worker 必须显式
指向 `ckpt/moge3` 中的 checkpoint。若模型页面需要接受许可，先在个人电脑完成授权，
再下载并 `scp -r` 到服务器，服务器无需保存 Hugging Face token。

## 6. 验证隔离状态

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

## 7. 激活和排错

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

## 8. 当前端到端状态

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

# 部署与调用

完整的源码克隆、三套 `venv` 创建、CUDA wheel 选择、checkpoint 放置和环境排错见
[`THREE_ENV_SETUP.md`](THREE_ENV_SETUP.md)。本页说明架构选择和调用约定。

- 公司服务器：`bash scripts/setup_three_envs.sh`（标准 Python `venv`）。
- 测试服务器：`bash scripts/setup_three_conda_envs.sh`（三个 Conda prefix）。
- 两者都生成 `.envs/pi3x`、`.envs/moge3`、`.envs/vipe`，不要混用安装脚本。
- VIPE v1.2.0 固定使用官方 Torch 2.9.0/cu128 组合；Conda 脚本会把匹配的 CUDA
  Toolkit 12.8 安装在 VIPE prefix 内，公司 venv 方案则需系统提供 nvcc 12.8。

## 1. 部署结论

- Linux 推荐；VIPE 含 CUDA 扩展，Windows 原生环境通常不适合作为生产环境。
- 三套环境统一使用 Python 3.10，但分别安装依赖，不共享 site-packages。
- 推荐显存 48 GB 以上；显存较小时降低 `--pi3x-chunk` 和
  `--max-inference-side`。

采用三环境是合理且必要的：Pi3X 固定 NumPy 1.26.4，而 MoGe-3 要求 NumPy 2.x；
VIPE 还需要独立编译 CUDA 扩展。三个阶段通过 NPZ/JSON 文件衔接，因此不存在
跨环境共享 Torch tensor 的需求。

推荐目录：

```text
/opt/camera-create/envs/pi3x
/opt/camera-create/envs/moge3
/opt/camera-create/envs/vipe
```

优先使用标准 `python -m venv`。如果 CUDA/PyTorch 只能通过 Conda 管理，可以为
三个模型分别创建 Python 3.10 Conda 环境；Conda 并不是代码本身的强制要求。

## 2. 安装三个模型

自动安装入口：

```bash
bash scripts/clone_models.sh
# 二选一：
bash scripts/setup_three_envs.sh
# bash scripts/setup_three_conda_envs.sh
```

将模型放在以下固定目录：

```text
camera_create/ckpt/pi3x/   # Pi3X Hugging Face snapshot或本地权重目录
camera_create/ckpt/moge3/  # MoGe-3 ViT-L checkpoint（必须显式提供）
camera_create/ckpt/vipe/   # VIPE/GeoCalib 的 Hugging Face 与 Torch 缓存
```

源码可放在 `third_party/Pi3`、`third_party/MoGe`、`third_party/vipe`，但这些上游
仓库不提交到 camera-create Git 仓库。分别在对应环境中安装并验证：

```bash
/opt/camera-create/envs/pi3x/bin/python -c "from pi3 import Pi3X; print('Pi3X OK')"
/opt/camera-create/envs/moge3/bin/python -c "from moge.model.v3 import MoGeModel; print('MoGe-3 OK')"
/opt/camera-create/envs/vipe/bin/vipe --help
```

模型具有各自许可证，尤其 Pi3X 权重可能限制商用。

VIPE 不只有自身 Python 包。第一次创建 SLAM 网络时，上游还会尝试从 Google Drive
下载 DROID-SLAM `droid.pth`；GeoCalib 则会从 GitHub Release 下载
`pinhole.tar`。生产 CLI 默认禁止这种运行中下载，并在进入 VIPE 前检查：

```text
ckpt/vipe/torch/hub/droid_slam/droid.pth
ckpt/vipe/torch/hub/geocalib/pinhole.tar
```

测试服务器能访问 Hugging Face 和 GitHub 时，一次性准备：

```bash
.envs/vipe/bin/python scripts/prepare_vipe_assets.py \
  --cache-root ckpt/vipe --download-missing
```

若只能访问国内 HF 反向代理，可覆盖 DROID URL（代理域名由部署方自行确认）：

```bash
.envs/vipe/bin/python scripts/prepare_vipe_assets.py \
  --cache-root ckpt/vipe --download-missing \
  --droid-url https://hf-mirror.com/vslamlab/droidslam/resolve/main/droid.pth
```

这里的 DROID-SLAM 下载地址是社区 Hugging Face 镜像
`vslamlab/droidslam`，不是 NVIDIA 官方模型仓库；GeoCalib 使用官方 v1.0 Release。
如果不希望使用镜像，可在能访问 Google Drive 的电脑下载官方文件，然后执行：

```bash
.envs/vipe/bin/python scripts/prepare_vipe_assets.py \
  --cache-root ckpt/vipe \
  --droid-source /downloads/droid.pth \
  --geocalib-source /downloads/pinhole.tar
```

公司服务器离线时，直接从已经成功准备的同版本测试机复制整个目录即可：

```bash
scp -r test-server:/path/camera-create/ckpt/vipe/torch ./ckpt/vipe/
.envs/vipe/bin/python scripts/prepare_vipe_assets.py --cache-root ckpt/vipe
```

这两个文件是模型数据，不含 Python/CUDA 二进制，可以跨两台 Linux 服务器迁移；
VIPE 源码仍须固定 v1.2.0，三套环境仍按各自服务器的 CUDA 方案安装。若 shell 已设置
`TORCH_HOME`，CLI 会遵循该变量并在那里检查，部署时建议先 `unset TORCH_HOME`，使缓存
稳定落在 `ckpt/vipe/torch`。只有确认运行机能访问 Google Drive/GitHub 时，才可显式
传 `--allow-vipe-downloads` 恢复 VIPE 上游的自动下载行为。

MoGe-3 推荐 `Ruicheng/moge-3-vitl`，不默认采用 1.25B 的 ViT-G。ViT-L 已明显改善
局部几何且部署成本较低。默认 `refine_steps=3`；A100 论文数据约 121 ms/帧，
而 MoGe-2 ViT-L 约 39 ms/帧，应为视频处理预留约三倍的 MoGe 推理时间。

## 3. 安装 patched VIPE

克隆脚本会生成 `third_party/vipe`，执行：

```bash
.envs/vipe/bin/python scripts/setup_vipe.py --vipe-source third_party/vipe
```

该脚本会把 cached metric-depth backend 与 `vipe_cached_depth` 配置复制到
VIPE checkout，并为 v1.2.0 回填 cached backend 所需的原始帧索引，然后使用当前
Python 环境进行 editable install。运行前检查：

```bash
.envs/pi3x/bin/python scripts/check_three_envs.py --env-root .envs --project-root .
```

完整检查也会验证上述两个 VIPE 资产；`--skip-checkpoints` 会同时跳过模型权重和
VIPE 资产，只适合刚创建完环境时使用。

如果旧安装报 `DepthEstimationInput has no attribute frame_idx`，拉取最新代码后只需
重新应用源码补丁，不需要重新编译 CUDA 扩展：

```bash
.envs/vipe/bin/python scripts/setup_vipe.py \
  --vipe-source third_party/vipe --skip-install
```

## 4. 端到端调用

```bash
.envs/pi3x/bin/python cli.py \
  --input /data/input.mp4 \
  --target-fps 24 \
  --max-frames 241 \
  --max-video-seconds 10.06 \
  --device cuda:0 \
  --pi3x-python .envs/pi3x/bin/python \
  --moge3-python .envs/moge3/bin/python \
  --vipe-command .envs/vipe/bin/vipe
```

或安装项目后：

```bash
camera-create --input /data/input.mp4 --target-fps 24 --max-frames 241
```

输出固定写在原视频旁边：`cam_input.mp4.json` 和 `input.mp4.camera/`。旧版
`--output` 仅为命令兼容而保留，传入后会给出弃用提示并被忽略。

单视频运行会在输入目录的 `.camera_create_ckpt/` 原子保存 Pi3X、MoGe-3、融合 metric
depth 和 VIPE 阶段状态。如果 VIPE 或导出报错，使用完全相同的输入与关键参数重新执行
原命令，会显示 `[resume]` 并跳过已经验证的阶段。输入文件、模型路径或推理参数变化
会自动使旧缓存失效。成功后默认清理大型缓存；需要保留时使用：

```bash
.envs/pi3x/bin/python cli.py --input input.mp4 --keep-stage-cache
# 或将失败恢复点放到高速大容量磁盘
.envs/pi3x/bin/python cli.py --input input.mp4 \
  --stage-cache-dir /fast-disk/camera-resume/input-001
```

显存不足时：

```bash
.envs/pi3x/bin/python cli.py --input input.mp4 \
  --pi3x-chunk 8 --pi3x-stride 4 --max-inference-side 448
```

已知相机水平 FOV 时应传入 `--fov-x-deg`，可提高 MoGe-3 的尺度/内参一致性。

CLI 已实现 Pi3X、MoGe-3 和 VIPE 的独立子进程调用。默认路径就是上面三个
`.envs` 可执行文件，也可以省略显式参数。隔离调度已有单元测试，但真实 CUDA 模型
端到端仍需服务器 smoke test，不能仅凭无 GPU 测试宣称 metric 精度已经验证。

## 5. Metric 含义与限制

MoGe-3 提供米制深度锚点，Pi3X 提供时序一致深度，融合后的米制深度进入
VIPE bundle adjustment，因此输出平移单位标记为 metre。单目 metric 尺度仍受
MoGe-3 域偏差影响；生产使用必须查看 `camera_report.json`，最好再用已知长度、
IMU、LiDAR 或标定场景做外部尺度验证。代码不会把“数组形状正确”等同于
“真实世界尺度绝对准确”。

## 6. 采用 MoGe-3 的依据

MoGe-3 ViT-L 相对 MoGe-2 的论文平均结果包括：metric-depth 阈值准确率
`77.3 -> 82.7`、局部点图严格阈值准确率 `46.6 -> 55.9`、全局点图相对误差
`8.73 -> 7.91`。提升集中在细结构和深度不连续区域，符合相机/几何缓存用途。

- 官方仓库：https://github.com/microsoft/MoGe
- MoGe-3 论文：https://arxiv.org/abs/2607.17967
- 官方评测表：https://arxiv.org/html/2607.17967v2#S4

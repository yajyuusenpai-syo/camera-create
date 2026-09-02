# 部署与调用

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
若用户没有预先设置 `HF_HOME`/`TORCH_HOME`，CLI 会让 VIPE 将自动下载的
GeoCalib 等权重写入 `ckpt/vipe/`；已有环境变量始终优先，不会被覆盖。

MoGe-3 推荐 `Ruicheng/moge-3-vitl`，不默认采用 1.25B 的 ViT-G。ViT-L 已明显改善
局部几何且部署成本较低。默认 `refine_steps=3`；A100 论文数据约 121 ms/帧，
而 MoGe-2 ViT-L 约 39 ms/帧，应为视频处理预留约三倍的 MoGe 推理时间。

## 3. 安装 patched VIPE

仓库已包含 `third_party/vipe`，执行：

```bash
/opt/camera-create/envs/vipe/bin/python scripts/setup_vipe.py
```

该脚本会把 cached metric-depth backend 与 `vipe_cached_depth` 配置复制到
VIPE checkout，然后使用当前 Python 环境进行 editable install。运行前检查：

```bash
/opt/camera-create/envs/vipe/bin/python scripts/check_environment.py
```

## 4. 端到端调用

```bash
python cli.py \
  --input /data/input.mp4 \
  --output /data/camera_result \
  --device cuda:0
```

或安装项目后：

```bash
camera-create --input /data/input.mp4 --output /data/camera_result
```

如需重复写入同一输出目录，普通结果文件会安全覆盖。使用 `--keep-work` 时，
若已有 `output/work`，程序会停止并要求先由操作者确认如何处理旧中间结果。

显存不足时：

```bash
python cli.py --input input.mp4 --output result \
  --pi3x-chunk 8 --pi3x-stride 4 --max-inference-side 448
```

已知相机水平 FOV 时应传入 `--fov-x-deg`，可提高 MoGe-3 的尺度/内参一致性。

> 当前实现状态：现有 CLI 尚未实现 Pi3X、MoGe-3 两个独立 worker，仍直接加载
> MoGe-2。完成 worker 改造前，上述三环境命令是部署规范，不代表当前提交已经能
> 以三环境完成真实端到端推理。

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

# 部署与调用

## 1. 环境要求

- Linux 推荐；VIPE 含 CUDA 扩展，Windows 原生环境通常不适合作为生产环境。
- Python 3.10 或 3.11、CUDA GPU、与 CUDA 匹配的 PyTorch。
- 推荐显存 48 GB 以上；显存较小时降低 `--pi3x-chunk` 和
  `--max-inference-side`。

项目优先使用标准 Python `venv`，不强制 Conda：

```bash
cd camera_create
bash scripts/setup_venv.sh
source .venv/bin/activate
```

如果服务器的 CUDA/PyTorch 只能通过 Conda 管理，可以创建 Python 3.10
环境后继续执行相同的 `pip install -e .`。Conda 不是代码运行的必要条件。

## 2. 安装三个模型

将模型放在以下固定目录：

```text
camera_create/ckpt/pi3x/   # Pi3X Hugging Face snapshot或本地权重目录
camera_create/ckpt/moge2/  # MoGe-2 snapshot；允许其中包含 model.pt
camera_create/ckpt/vipe/   # VIPE/GeoCalib 的 Hugging Face 与 Torch 缓存
```

Pi3X 与 MoGe-2 的 Python 源码包也必须可导入。根据其上游仓库安装后，用：

```bash
python -c "from pi3 import Pi3X; from moge.model.v2 import MoGeModel; print('OK')"
```

验证。模型具有各自许可证，尤其 Pi3X 权重可能限制商用。
若用户没有预先设置 `HF_HOME`/`TORCH_HOME`，CLI 会让 VIPE 将自动下载的
GeoCalib 等权重写入 `ckpt/vipe/`；已有环境变量始终优先，不会被覆盖。

## 3. 安装 patched VIPE

仓库已包含 `third_party/vipe`，执行：

```bash
python scripts/setup_vipe.py
```

该脚本会把 cached metric-depth backend 与 `vipe_cached_depth` 配置复制到
VIPE checkout，然后使用当前 Python 环境进行 editable install。运行前检查：

```bash
python scripts/check_environment.py
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

已知相机水平 FOV 时应传入 `--fov-x-deg`，可提高 MoGe-2 的尺度/内参一致性。

## 5. Metric 含义与限制

MoGe-2 提供米制深度锚点，Pi3X 提供时序一致深度，融合后的米制深度进入
VIPE bundle adjustment，因此输出平移单位标记为 metre。单目 metric 尺度仍受
MoGe-2 域偏差影响；生产使用必须查看 `camera_report.json`，最好再用已知长度、
IMU、LiDAR 或标定场景做外部尺度验证。代码不会把“数组形状正确”等同于
“真实世界尺度绝对准确”。

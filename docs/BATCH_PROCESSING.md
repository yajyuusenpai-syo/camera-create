# 多 GPU 目录批处理

目录模式递归使用 `os.walk` 查找视频，将每个结果原子写入原视频所在目录：

```text
dataset/a/clip.mp4
dataset/a/cam_clip.mp4.json
dataset/a/clip.mp4.camera/          # NPY 与 camera_report.json

dataset/b/movie.mkv
dataset/b/cam_movie.mkv.json
dataset/b/movie.mkv.camera/         # NPY 与 camera_report.json
```

默认扩展名包括 `mp4,mkv,mov,avi,webm,m4v,mpg,mpeg,ts`。每段视频先由 ffmpeg
转换成最高 10.06 秒、24 FPS、241 帧的处理副本，原文件不会被修改。目录名保留
完整视频文件名再加 `.camera`，所以同目录的 `clip.mp4` 与 `clip.mkv` 不会冲突。

## 调用

单卡验证：

```bash
.envs/pi3x/bin/python cli.py \
  --input /data/videos \
  --gpu-ids 0 \
  --workers-per-gpu 1
```

8 卡、每卡 4 个 worker：

```bash
.envs/pi3x/bin/python cli.py \
  --input /data/videos \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4 \
  --target-fps 24 \
  --max-frames 241 \
  --max-video-seconds 10.06
```

也可以使用包装脚本：

```bash
bash scripts/run_batch.sh /data/videos \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4
```

每个调度 worker 通过 `CUDA_VISIBLE_DEVICES` 绑定一张物理 GPU，在该 worker 内依次
调用隔离的 Pi3X、MoGe-3、VIPE 进程。32 个 worker 表示最多有32条完整流水线并行，
不是32个轻量线程。Pi3X 显存需求较大，首次部署应从 `--workers-per-gpu 1` 开始，
根据 `nvidia-smi` 峰值逐步提高；盲目设置为4可能直接 OOM，并不一定更快。

## Checkpoint 与 resume

扫描结果排序后，通过轮询方式预先固定分配：

```text
worker 0 -> videos[0], videos[worker_count], ...
worker 1 -> videos[1], videos[worker_count + 1], ...
```

checkpoint 默认保存在：

```text
/data/videos/.camera_create_ckpt/run_<配置和任务哈希>/
├── worker_000.json
├── worker_001.json
├── ...
├── stage_cache/
│   └── worker_<编号>_<视频哈希>/
│       ├── *.normalized.mp4 / normalized.json
│       └── pipeline/
│           ├── pi3x_depth.npz
│           ├── moge3_depth.npz
│           ├── metric_depth_cache.npz
│           ├── stage_state.json
│           └── vipe/
└── summary.json
```

每个 worker 在任务开始、成功或失败后原子更新自己的 JSON。重新执行相同命令时，
完整且满足当前 FPS/帧数/时长配置的 `cam_<原文件名>.json` 会跳过；失败或不完整的结果
会从最后一个有效阶段继续：Pi3X、MoGe-3 和融合 metric depth 均不会重复推理，
VIPE 失败时只重跑 VIPE。worker JSON 的失败条目会记录 `stage_cache` 路径与
`completed_stages`；NPZ 和 JSON 均采用临时文件加原子改名，半写文件不会被复用。

输入文件大小/修改时间、任务列表或关键推理参数发生变化时会创建新 run 哈希，
pipeline 内部还有独立配置指纹，避免错误复用旧深度。成功发布最终 JSON 后默认删除
该视频的大型阶段缓存；失败缓存会保留。调试或希望成功后也保留时传：

```bash
bash scripts/run_batch.sh /data/videos --gpu-ids 0,1 --keep-stage-cache
```

强制重算最终结果使用：

```bash
bash scripts/run_batch.sh /data/videos --gpu-ids 0,1 --overwrite
```

自定义 checkpoint 位置：

```bash
bash scripts/run_batch.sh /data/videos \
  --checkpoint-dir /fast-disk/camera-checkpoints \
  --gpu-ids 0,1,2,3
```

## 输出 JSON

每个成功视频写出 `format_version: 2`、`is_metric: true`、源/目标 FPS、帧数和
OpenCV 坐标系下的逐帧 `c2w` 与像素单位 3x3 `intrinsics`。时间戳严格使用
`frame_index / target_fps`。只有现有 metric camera 验证通过后才会发布最终 JSON；
先写 `.tmp` 再原子改名，进程中断不会留下被误判为成功的半文件。

JSON 写在视频旁边，包含逐帧 `c2w` 和 3x3 `intrinsics`；相同视频对应的 NPY、
`camera_report.json` 写入 `<原视频完整文件名>.camera/`。`fps` 等于实际
`target_fps`，`frame_count` 等于 `frames` 长度且不超过 `max_frames`。

启动时会打印全部批处理参数、扫描到的视频数、GPU/worker 数和 checkpoint 路径；
运行中由 tqdm 汇总 `completed/skipped/failed`。单个视频失败不会终止该 worker 的
后续静态任务，完整 traceback 保存在对应 worker checkpoint 中。命令在存在失败
或 worker 崩溃时返回非零退出码。

## 常用参数

```text
--video-extensions .mp4,.mkv,.mov
--target-fps 24
--max-frames 241
--max-video-seconds 10.06
--workers-per-gpu 1
--checkpoint-dir PATH
--overwrite
--keep-stage-cache
--ffmpeg-command /path/to/ffmpeg
```

目录模式不能传 `--output`、`--work-dir`、`--stage-cache-dir` 或 `--keep-work`。
单视频模式也采用同样的旁路输出结构，不再要求 `--output`。为兼容旧命令，该参数仍
可解析，但会提示已弃用并被忽略。失败恢复点默认放在视频所在目录的
`.camera_create_ckpt/`；成功后自动清理。可用 `--stage-cache-dir PATH` 指定位置，或用
`--keep-stage-cache` 在成功后保留。

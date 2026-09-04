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

## 8 机 × 8 卡 × 4 worker

多机模式要求所有机器看到完全相同的输入目录快照，并共享同一个 checkpoint 目录。
8 台机器分别使用唯一的 `--node-rank 0` 到 `--node-rank 7`，其余参数以及
`--run-id` 必须完全一致。例如第 0 台机器：

```bash
.envs/pi3x/bin/python cli.py \
  --input /shared/videos \
  --checkpoint-dir /shared/camera-checkpoints \
  --run-id camera-50k-v1 \
  --num-nodes 8 \
  --node-rank 0 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4 \
  --target-fps 24 \
  --max-frames 241 \
  --max-video-seconds 10.06
```

第 1～7 台只把 `--node-rank` 改成对应编号。不要为不同节点修改 `--run-id`，
也不要让两台正常工作的机器使用相同 node rank。多机模式不使用
`torch.distributed`：50k 个视频是相互独立的任务，程序把本机 worker 映射为：

```text
local_workers = 8 GPU × 4 = 32
global_workers = 8 node × 32 = 256
global_worker_id = node_rank × 32 + local_worker_id
tasks = sorted_videos[global_worker_id::256]
```

### DLC 自动机器识别

DLC 每台机器只启动一次本命令，并提供 `WORLD_SIZE`、`RANK`、`MASTER_ADDR`、
`MASTER_PORT` 时，CLI 会自动把 `WORLD_SIZE` 解释为机器数、`RANK` 解释为机器
rank，并记录 coordinator 地址。它同时兼容 Accelerate 风格的横线和下划线参数。
推荐的 DLC 启动命令是：

```bash
.envs/pi3x/bin/python cli.py \
  --input /shared/videos \
  --checkpoint-dir /shared/camera-checkpoints \
  --run-id "${DLC_JOB_ID:-camera-50k-v1}" \
  --num_machines "$WORLD_SIZE" \
  --num_processes "$((WORLD_SIZE * 8))" \
  --machine_rank "$RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4 \
  --disable-cudnn \
  --disable-sdp
```

仓库提供了等价且默认强制禁用cuDNN/融合SDP的包装脚本；DLC每个入口调用：

```bash
bash scripts/run_dlc_batch.sh \
  /shared/videos \
  /shared/camera-checkpoints \
  camera-50k-v1
```

该脚本要求DLC提供 `WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT`，默认调用
`.envs/pi3x/bin/python`。需要使用公司开发机的其他Python环境路径时设置
`CAMERA_CREATE_PYTHON=/path/to/pi3x-env/bin/python`。

也可以省略 `--num_machines`、`--machine_rank`、`--main_process_ip` 和
`--main_process_port`，直接使用对应环境变量。`--run-id` 的自动候选依次是
`CAMERA_CREATE_RUN_ID`、`DLC_JOB_ID`、`PAI_JOB_ID`；如果DLC没有提供其中任何一个，
多机模式仍要求显式指定。`--num_processes` 表示DLC的全局GPU槽位数，因此这里必须是
`8机器 × 8 GPU = 64`；camera worker 总数还会乘 `--workers-per-gpu 4`，最终为256。

如果平台实际使用 `accelerate launch` 或 `torchrun`，因而在每台机器上拉起8个入口
进程，CLI 会从 `WORLD_SIZE=64`、`LOCAL_WORLD_SIZE=8` 和全局 `RANK` 还原为8台
机器；只有每台机器的 `LOCAL_RANK=0` 会创建本机32个camera worker，其余入口进程
打印 `idle_launcher_process` 后正常退出。若存在 `GROUP_WORLD_SIZE/GROUP_RANK`，优先
使用这两个更明确的机器级变量。这样不会发生“8个launcher进程各自再创建32个worker”
的重复和显存争抢。能控制DLC入口命令时，仍优先选择每机只执行一次上述CLI，避免
创建无用的launcher进程。

本项目的视频任务互相独立，不需要梯度或张量跨机通信，因此
`MASTER_ADDR/MASTER_PORT` 只作为DLC拓扑审计信息写入manifest和summary，不会建立
PyTorch process group。实际跨机一致性由共享manifest、全局静态分片和文件系统lease
保证。

公司环境要求禁用cuDNN和SDP优化时，必须加入：

```text
--disable-cudnn --disable-sdp
```

该设置会传播到Pi3X、MoGe-3及VIPE三个隔离Python环境。`--disable-sdp` 关闭Flash、
memory-efficient和cuDNN SDP内核，但保留math SDP fallback；若把所有SDP后端一起关闭，
使用 `scaled_dot_product_attention` 的模型会直接无可用后端。也可以通过环境变量统一
设置：

```bash
export CAMERA_CREATE_DISABLE_CUDNN=1
export CAMERA_CREATE_DISABLE_SDP=1
```

`manifest.json` 由第一个启动的节点以独占方式创建。后续节点会核对完整视频清单
（相对路径、大小、修改时间）、拓扑和关键推理参数；任何差异都会在启动推理前报错，
而不是静默地产生重复或缺失任务。输入集合或处理参数需要改变时，使用新的
`--run-id`。

共享 checkpoint 下还会为每个正在处理的视频创建原子 lease，并持续写 heartbeat。
即使误启动重复 node rank，同一视频也只有一个进程能进入推理。进程死亡并超过
`--lease-timeout-seconds`（默认 900 秒）后，下一次运行可以接管该 lease；正常长推理
会持续刷新 heartbeat，不会因为超过 900 秒而被抢占。多机安全依赖 checkpoint 路径
位于所有节点都支持原子目录创建/改名的同一共享文件系统（常规 NFS/CephFS 均可）；
节点各自使用本地 checkpoint 目录时无法提供跨机 lease 保护。

多机 checkpoint 结构：

```text
/shared/camera-checkpoints/run_camera-50k-v1/
├── manifest.json
├── worker_000.json ... worker_255.json
├── leases/                         # 仅含当前正在运行的任务
├── stage_cache/
│   └── worker_<global-id>_<video-hash>/
└── summary_node_000.json ... summary_node_007.json
```

每个节点只汇报自己的 `videos_assigned` 和进度，避免8台机器竞争覆盖一个
`summary.json`。判断全局完成时，应确认8个 node summary 都存在，并验证输入目录下
每个视频都有有效的 `cam_<视频名>.json`。节点故障后使用相同 node rank、run id 和
参数重新启动；对应全局 worker checkpoint 与失败视频的 stage cache 会继续复用。

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

多机写 `summary_node_<node-rank>.json`，避免节点之间相互覆盖。单机同时保留原有
`summary.json`，并写一份相同内容的 `summary_node_000.json`，兼容已有调用。

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
--num-nodes 1
--node-rank 0
--run-id NAME
--lease-timeout-seconds 900
--num-processes 64
--main-process-ip HOST
--main-process-port PORT
--disable-cudnn
--disable-sdp
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

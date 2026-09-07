# 视频清单分片批处理

`camera-create` 的 `--input` 只接受 `.txt` 或 `.json` 视频路径清单，不再直接接受
单个视频或目录。推荐先将完整数据集确定性地切成若干份，再为每一份启动一个互相
独立的单机器任务。每个任务内部仍支持多 GPU、多进程、checkpoint/resume 和 tqdm。

默认情况下，每张GPU启动一个常驻Pi3X服务和一个常驻MoGe-3服务；同卡所有camera
worker共享这两个服务，因此两个深度模型在整份清单中各只加载一次。每个服务串行
处理本卡请求，避免常驻多份模型导致显存爆炸。VIPE仍然为每个视频单独启动和退出。
调试旧行为时可传 `--no-persistent-depth-services`，恢复每个视频重新加载深度模型。
启动阶段会先在所有选定GPU上并行加载Pi3X并等待全部就绪，再并行加载MoGe-3；不会
把Pi3X和MoGe-3两组同时加载。VIPE启动和执行逻辑不受该优化影响。

## 生成8份清单

以下命令使用 `os.walk` 递归扫描视频，稳定排序后轮询均分，确保各分片的视频数最多
相差1：

```bash
.envs/pi3x/bin/python scripts/create_video_shards.py \
  --input-dir /mnt/vlm-ks3/chenkaijin/datasets/10kExport/sources \
  --output-dir /mnt/vlm-ks3/chenkaijin/datasets/10kExport/shards \
  --shards 8 \
  --prefix clip \
  --format txt
```

生成 `clip_1.txt` 到 `clip_8.txt`。每行是一个规范化绝对视频路径。需要JSON时使用
`--format json`，生成文件结构为：

```json
{
  "format_version": 1,
  "shard_index": 1,
  "shard_count": 8,
  "videos": ["/data/a.mp4", "/data/b.mkv"]
}
```

CLI也接受纯JSON字符串数组、对象中的 `paths` 数组，以及数组内的
`{"path": "..."}` / `{"video_path": "..."}` 条目。相对路径按清单所在目录解析。
空行和TXT中以 `#` 开头的注释会被忽略。不存在、重复或扩展名不受支持的路径会在
启动模型前报错。

## 每个单机器任务的启动命令

为8个任务分别选择 `clip_1.txt` 到 `clip_8.txt`；无需 `WORLD_SIZE`、`RANK`、
`MASTER_ADDR` 或 `torchrun`。例如第一个任务：

```bash
cd /mnt/vlm-ks3/chenkaijin/data_pipeline/new_camera_infer/camera-create

exec "$PWD/.envs/pi3x/bin/python" cli.py \
  --input /mnt/vlm-ks3/chenkaijin/datasets/10kExport/shards/clip_1.txt \
  --target-fps 24 \
  --max-frames 241 \
  --max-video-seconds 10.06 \
  --lease-timeout-seconds 900 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 6 \
  --disable-cudnn \
  --disable-sdp \
  --pi3x-python "$PWD/.envs/pi3x/bin/python" \
  --moge3-python "$PWD/.envs/moge3/bin/python" \
  --vipe-command "$PWD/.envs/vipe/bin/vipe"
```

包装脚本等价调用：

```bash
bash scripts/run_batch.sh /path/to/clip_1.txt \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 6 \
  --disable-cudnn \
  --disable-sdp
```

清单模式默认固定为 `num_nodes=1,node_rank=0`，不会读取 DLC/torchrun 的分布式环境
变量，因此已经切好的清单不会被二次分片。旧的显式 `--num-nodes/--node-rank` 参数
仍保留用于兼容，但8个独立任务不应传入它们。

## 输出结构

输出始终写回每个源视频所在目录，而不是清单目录：

```text
/dataset/group/a.mp4
/dataset/group/cam_a.json
/dataset/group/a.mp4.camera/
├── camera_report.json
├── intrinsics_K.npy
├── intrinsics.npy
├── poses_c2w_metric.npy
├── extrinsics_w2c_metric.npy
└── scale_per_frame.npy
```

最终JSON为 `cam_<原视频主名>.json`，包含 `format_version: 2`、`is_metric: true`、
逐帧 `c2w` 和像素单位 `intrinsics`。同一目录下若同时存在 `a.mp4` 和 `a.mkv`，
二者都会映射到 `cam_a.json`，因此同一清单内会在推理前拒绝该冲突。

## Checkpoint与防重复

默认checkpoint位置由清单名隔离：

```text
<清单目录>/.camera_create_ckpt/clip_1/run_<任务哈希>/
├── manifest.json
├── worker_000.json ...
├── stage_cache/
└── summary.json
```

也可以传 `--checkpoint-dir PATH`。重新执行相同清单和参数时，有效的现有
`cam_<stem>.json` 会跳过；失败任务会复用已完成的 Pi3X、MoGe-3、metric-depth缓存。
最终JSON采用原子发布。

每个视频目录使用：

```text
<视频目录>/.camera_create_ckpt/video_leases/
```

lease以最终JSON绝对路径为身份，与清单位置无关。即使清单意外重叠，同一输出也只能被一个进程持有；
正常推理持续刷新heartbeat，进程死亡并超过 `--lease-timeout-seconds` 后可被恢复。

常用选项：

```text
--video-extensions .mp4,.mkv,.mov
--target-fps 24
--max-frames 241
--max-video-seconds 10.06
--gpu-ids 0,1,2,3,4,5,6,7
--workers-per-gpu 6
--lease-timeout-seconds 900
--checkpoint-dir PATH
--overwrite
--keep-stage-cache
--disable-cudnn
--disable-sdp
--no-persistent-depth-services
```

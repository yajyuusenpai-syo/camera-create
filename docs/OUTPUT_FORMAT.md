# 输出格式

对于输入 `/data/videos/clip.mp4`，主结果与中间数组结构为：

```text
/data/videos/cam_clip.json
/data/videos/clip.mp4.camera/
```

`cam_clip.json` 使用 format v2，包含源/目标 FPS、帧数、metric 标志、原视频
`source_resolution`、内参推理所用的 `intrinsics_inference_resolution`，以及每帧
的 `frame_index`、`timestamp_seconds`、4x4 `c2w` 和 3x3 `intrinsics`。
`clip.mp4.camera/` 包含：

| 文件 | 形状 | 含义 |
|---|---:|---|
| `poses_c2w_metric.npy` | `(T,4,4)` | OpenCV camera-to-world，平移单位米，首帧为单位阵 |
| `extrinsics_w2c_metric.npy` | `(T,4,4)` | 上述矩阵的逆，传统 world-to-camera extrinsics |
| `intrinsics.npy` | `(T,1,4)` | `[fx,fy,cx,cy]`，像素单位 |
| `intrinsics_K.npy` | `(T,3,3)` | 3×3 内参矩阵，像素单位 |
| `intrinsics_normalized.npy` | `(T,1,4)` | 分辨率无关的 `[fx/W,fy/H,cx/W,cy/H]` |
| `intrinsics_normalized_K.npy` | `(T,3,3)` | 上述归一化内参的3×3矩阵形式 |
| `scale_per_frame.npy` | `(T,)` | Pi3X 深度到 MoGe-3 米制深度的 EMA 尺度 |
| `camera_report.json` | JSON | 坐标约定、视频信息和数值验证结果 |

坐标采用 OpenCV 约定：相机局部坐标 `+x` 向右、`+y` 向下、`+z` 向前。
JSON逐帧同时保留像素单位的 `intrinsics` 和分辨率无关的
`intrinsics_normalized`。归一化规则是K矩阵第0行除以图像宽度W、第1行除以图像
高度H，最后一行保持 `[0,0,1]`。

默认先在现有FPS/时长归一化转码中保持宽高比缩放到480像素高，再把该视频交给
VIPE估计内参。16:9视频通常得到854×480；可用 `--processing-height` 修改。
像素内参 `intrinsics` 对应 `intrinsics_inference_resolution`，而不是
`source_resolution`。兼容字段 `image_width`/`image_height` 也表示内参推理分辨率。

`intrinsics` 不是“米制”量；它以像素表达。`is_metric: true` 表示外参平移继承
MoGe-3单目metric深度并以米为目标单位，但不是传感器或GT尺度认证。JSON同时写入
`metric_scale_provenance` 和 `metric_scale_validated_against_ground_truth: false`。

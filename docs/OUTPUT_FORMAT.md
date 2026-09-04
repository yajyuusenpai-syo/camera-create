# 输出格式

对于输入 `/data/videos/clip.mp4`，主结果与中间数组结构为：

```text
/data/videos/cam_clip.mp4.json
/data/videos/clip.mp4.camera/
```

`cam_clip.mp4.json` 使用 format v2，包含源/目标 FPS、帧数、metric 标志，以及每帧
的 `frame_index`、`timestamp_seconds`、4x4 `c2w` 和 3x3 `intrinsics`。
`clip.mp4.camera/` 包含：

| 文件 | 形状 | 含义 |
|---|---:|---|
| `poses_c2w_metric.npy` | `(T,4,4)` | OpenCV camera-to-world，平移单位米，首帧为单位阵 |
| `extrinsics_w2c_metric.npy` | `(T,4,4)` | 上述矩阵的逆，传统 world-to-camera extrinsics |
| `intrinsics.npy` | `(T,1,4)` | `[fx,fy,cx,cy]`，像素单位 |
| `intrinsics_K.npy` | `(T,3,3)` | 3×3 内参矩阵，像素单位 |
| `scale_per_frame.npy` | `(T,)` | Pi3X 深度到 MoGe-3 米制深度的 EMA 尺度 |
| `camera_report.json` | JSON | 坐标约定、视频信息和数值验证结果 |

坐标采用 OpenCV 约定：相机局部坐标 `+x` 向右、`+y` 向下、`+z` 向前。
`intrinsics` 不是“米制”量；它以像素表达。Metric 指深度和外参平移的单位为米。

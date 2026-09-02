# 处理链设计

`cli.py` 是唯一推荐入口，内部调用顺序如下：

1. `video.py` 解码视频；保持原视频给 VIPE，只生成受控尺寸的深度推理帧。
2. `depth.py` 用重叠时间窗运行 Pi3X，获得时序一致的相对深度。
3. MoGe-3 worker 逐帧推理，获得米制深度、点图、有效掩码和归一化内参；默认采用
   ViT-L 与 3 次 SSR refinement。
4. 使用论文指定的 inverse-depth weighted scale 与 momentum=0.99 EMA，生成
   VIPE `CachedDepthModel` 消费的米制深度缓存。
5. `vipe_runner.py` 临时设置缓存路径并执行 `vipe_cached_depth` bundle adjustment。
6. `artifacts.py` 读取 VIPE 稀疏结果；平移线性插值、旋转使用 quaternion
   SLERP，并将数值漂移投影回 SO(3)。
7. 同时导出 c2w、w2c、两种内参表达和尺度历史，再写数值验证报告。

中间深度默认缩到最长边 560，避免原 Stage 2 将 961 帧 720p tensor 和多份
全分辨率深度同时驻留 GPU。Pi3X 只有当前时间窗进入 GPU，MoGe-3 每次只有
一帧进入 GPU；CPU 内存仍需容纳推理帧及约三份低分辨率深度数组。

## 三环境边界

三个模型使用三个独立的 Python 3.10 环境，通过磁盘缓存和子进程传递数据：

```text
主 CLI
  -> Pi3X worker（RGB 帧 -> relative_depth.npz）
  -> MoGe-3 worker（RGB 帧 -> metric_depth.npz）
  -> 主 CLI 融合深度（-> vipe_depth_cache.npz）
  -> VIPE worker（视频 + metric cache -> sparse camera）
  -> 主 CLI 插值、验证并导出 metric camera
```

环境之间不得传递 Python/Torch 对象，只传递有版本标记的 NPZ、JSON 和视频路径。
这样 Pi3X 可保留 NumPy 1.26.4，MoGe-3 可使用 NumPy 2.x 与 FlexGEMM/Triton，
VIPE 则使用其 CUDA 扩展所需的独立 PyTorch/CUDA 组合。

## MoGe-3 兼容范围

MoGe-3 与 MoGe-2 的输出契约基本一致，均包含 `points`、`depth`、`intrinsics`、
`mask`，可继续作为 VIPE metric-depth 融合的锚点。但它不是零代码改动替换：

```python
from moge.model.v3 import MoGeModel

model = MoGeModel.from_pretrained(ckpt_path).to(device)
output = model.infer(image, refine_steps=3, use_fp16=True)
```

MoGe-3 checkpoint 必须显式提供。最终缓存只消费最后一次 refinement 的 `depth`、
`intrinsics` 和 `mask`；`*_per_step` 仅用于调试，不进入 VIPE。

截至本文档更新时，仓库现有 `depth.py` 仍直接导入 MoGe-2 并与 Pi3X 同进程运行。
因此三环境内容是已确定的迁移目标；在 MoGe-3 worker 和 Pi3X worker 落地之前，
不得把当前 CLI 标记为三环境端到端已验证。

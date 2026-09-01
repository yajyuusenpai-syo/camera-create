# 处理链设计

`cli.py` 是唯一推荐入口，内部调用顺序如下：

1. `video.py` 解码视频；保持原视频给 VIPE，只生成受控尺寸的深度推理帧。
2. `depth.py` 用重叠时间窗运行 Pi3X，获得时序一致的相对深度。
3. `depth.py` 逐帧运行 MoGe-2，获得米制深度锚点。
4. 使用论文指定的 inverse-depth weighted scale 与 momentum=0.99 EMA，生成
   VIPE `CachedDepthModel` 消费的米制深度缓存。
5. `vipe_runner.py` 临时设置缓存路径并执行 `vipe_cached_depth` bundle adjustment。
6. `artifacts.py` 读取 VIPE 稀疏结果；平移线性插值、旋转使用 quaternion
   SLERP，并将数值漂移投影回 SO(3)。
7. 同时导出 c2w、w2c、两种内参表达和尺度历史，再写数值验证报告。

中间深度默认缩到最长边 560，避免原 Stage 2 将 961 帧 720p tensor 和多份
全分辨率深度同时驻留 GPU。Pi3X 只有当前时间窗进入 GPU，MoGe-2 每次只有
一帧进入 GPU；CPU 内存仍需容纳推理帧及约三份低分辨率深度数组。


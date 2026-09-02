# 本机 Python 环境验证记录

验证日期：2026-09-01

## 环境

- Conda prefix：`camera_create/.conda-env`
- Python：3.10.21
- PyTorch：2.13.0+cpu
- NumPy：2.2.6
- SciPy：1.15.3
- OpenCV：5.0.0

## 已通过

```text
python -m compileall: PASS
editable install/import: PASS
installed camera-create --help: PASS (exit 0)
pytest: 2 passed
ruff: All checks passed
pip check: No broken requirements found
```

## 尚不能在本机验证

环境探针确认本机缺少以下真实推理条件：

- CUDA GPU / CUDA PyTorch；
- 可导入的 `pi3`；
- 可导入的 `moge`；
- 安装后的 `vipe` CLI；
- `ckpt/pi3x` 与 `ckpt/moge3` 模型文件。

因此本次结论是 Python 项目、CLI、数学处理和输出插值的标准验证通过，不能将其
表述为 Pi3X + MoGe-3 + VIPE 真实模型端到端验证通过。三环境 worker 调度和缓存
校验已经实现并可在无 GPU 条件下测试；部署 GPU 环境后仍须分别检查三个环境，
再执行真实视频 smoke test。

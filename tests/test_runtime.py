"""Test that the cross-environment PyTorch backend policy disables fused kernels."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from camera_create.runtime import configure_torch_backends


class Toggle:
    """Act like one PyTorch backend getter/setter while recording its state."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, value: bool | None = None) -> bool | None:
        if value is None:
            return self.enabled
        self.enabled = value
        return None


def test_disable_cudnn_and_fused_sdp_keeps_math_fallback(monkeypatch) -> None:
    flash = Toggle()
    memory = Toggle()
    cudnn_sdp = Toggle()
    math = Toggle(False)
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(enabled=True),
            cuda=SimpleNamespace(
                enable_flash_sdp=flash,
                enable_mem_efficient_sdp=memory,
                enable_cudnn_sdp=cudnn_sdp,
                enable_math_sdp=math,
                flash_sdp_enabled=flash,
                mem_efficient_sdp_enabled=memory,
                cudnn_sdp_enabled=cudnn_sdp,
                math_sdp_enabled=math,
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    state = configure_torch_backends(disable_cudnn=True, disable_sdp=True)

    assert state == {
        "cudnn_enabled": False,
        "flash_sdp_enabled": False,
        "mem_efficient_sdp_enabled": False,
        "cudnn_sdp_enabled": False,
        "math_sdp_enabled": True,
    }

"""Apply reproducible PyTorch backend policy inside each isolated model process."""

from __future__ import annotations

from typing import Any


def configure_torch_backends(
    disable_cudnn: bool = False, disable_sdp: bool = False
) -> dict[str, Any]:
    """Disable cuDNN and fused CUDA SDP while retaining the safe math SDP fallback."""
    import torch

    if disable_cudnn:
        torch.backends.cudnn.enabled = False
    if disable_sdp:
        cuda = torch.backends.cuda
        for function_name in (
            "enable_flash_sdp",
            "enable_mem_efficient_sdp",
            "enable_cudnn_sdp",
        ):
            function = getattr(cuda, function_name, None)
            if function is not None:
                function(False)
        enable_math = getattr(cuda, "enable_math_sdp", None)
        if enable_math is not None:
            enable_math(True)
    return {
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "flash_sdp_enabled": _backend_enabled(torch, "flash_sdp_enabled"),
        "mem_efficient_sdp_enabled": _backend_enabled(
            torch, "mem_efficient_sdp_enabled"
        ),
        "cudnn_sdp_enabled": _backend_enabled(torch, "cudnn_sdp_enabled"),
        "math_sdp_enabled": _backend_enabled(torch, "math_sdp_enabled"),
    }


def _backend_enabled(torch_module: Any, function_name: str) -> bool | None:
    """Read an SDP backend flag across PyTorch versions without assuming it exists."""
    function = getattr(torch_module.backends.cuda, function_name, None)
    return bool(function()) if function is not None else None

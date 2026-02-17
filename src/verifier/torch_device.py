from __future__ import annotations

import os


def resolve_torch_device(torch_module):
    """
    Resolve torch device with optional override.

    Supported env values:
    - TORCH_DEVICE=auto (default)
    - TORCH_DEVICE=cpu
    - TORCH_DEVICE=cuda
    """
    requested = os.getenv("TORCH_DEVICE", "auto").strip().lower()

    if requested == "cpu":
        return torch_module.device("cpu"), "cpu-forced"

    if requested == "cuda":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda"), "cuda-forced"
        return torch_module.device("cpu"), "cuda-requested-but-unavailable"

    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), "cuda-auto"
    return torch_module.device("cpu"), "cpu-auto"

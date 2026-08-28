"""Device selection and synchronisation, shared by the quant_transformers_*.py scripts.

The HQQ scripts hardcode `mps` because they only ever ran on the Mac. The AWQ/GPTQ work runs the same benchmarks on a rented NVIDIA box, so the three `quant_transformers_*.py` scripts take a `--device` instead.

`sync` matters more than it looks. Both MPS and CUDA queue work asynchronously, so a timed region that doesn't synchronise before stopping the clock measures how fast Python queued the work, not how long the GPU took. `torch.mps.synchronize()` and `torch.cuda.synchronize()` are separate calls and neither exists on the other backend, hence the dispatch.
"""

import torch


def resolve(name: str) -> str:
    """'auto' -> the best available backend; anything else passes through."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def describe(device: str) -> str:
    if device.startswith("cuda"):
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return device

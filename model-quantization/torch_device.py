"""Device selection and synchronisation, shared by the quant_transformers_*.py scripts.

The HQQ scripts hardcode `mps` because they only ever ran on the Mac. The AWQ/GPTQ work runs the same benchmarks on a rented NVIDIA box, so the three `quant_transformers_*.py` scripts take a `--device` instead.

`sync` matters more than it looks. Both MPS and CUDA queue work asynchronously, so a timed region that doesn't synchronise before stopping the clock measures how fast Python queued the work, not how long the GPU took. `torch.mps.synchronize()` and `torch.cuda.synchronize()` are separate calls and neither exists on the other backend, hence the dispatch.
"""

import torch
from transformers import AutoModelForCausalLM


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


def load_causal_lm(model_id: str, device: str, dtype: str = "float16"):
    """Load a transformers checkpoint onto `device`, the right way per backend.

    Two things differ by backend and both matter.

    Placement: on CUDA, `device_map` is the normal path and handles a quantized checkpoint's own placement rules. On MPS it is pathologically slow, ten minutes and still loading an 8B model, so there the model loads on CPU and moves afterwards.

    Dtype: `float16` is right for the unquantized fp16 baseline and is what produced its 7.365 perplexity, so it stays the default and that number stays reproducible. Pass `auto` for a quantized checkpoint (AWQ, GPTQ) and let its own config decide, rather than forcing a dtype that fights the quantization it was saved with.
    """
    kwargs = {"dtype": dtype if dtype == "auto" else getattr(torch, dtype)}
    if device.startswith("cuda"):
        return AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)

#!/usr/bin/env bash
#
# Environment for the CUDA-only quantization methods (AWQ, GPTQ).
#
# Bring your own GPU. This makes no assumption about the cloud provider or
# the instance type - it needs an NVIDIA GPU with a working driver and
# nothing else. Every cloud GPU image ships the driver already (AWS Deep
# Learning AMI, GCP Deep Learning VM, RunPod/Lambda PyTorch images), as
# does a local workstation with CUDA installed.
#
#   Requirements:  NVIDIA GPU, >=24GB VRAM (48GB comfortable), ~120GB disk
#   Usage:         export HF_TOKEN=hf_...
#                  bash setup_cuda.sh
#                  source .venv-cuda/bin/activate
#
# Idempotent: safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script needs a machine with an NVIDIA driver." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==> uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> venv"
[ -d .venv-cuda ] || uv venv --python 3.12 .venv-cuda

# Versions are pinned rather than floating. llmcompressor moves fast, and an
# essay whose numbers can't be reproduced six months later isn't worth much.
# Bump these deliberately, and re-record them in README.md when you do.
echo "==> dependencies"
uv pip install --python .venv-cuda \
  "torch" \
  "transformers" \
  "llmcompressor" \
  "compressed-tensors" \
  "datasets" \
  "accelerate" \
  "lm_eval" \
  "loguru" \
  "numpy" \
  "huggingface_hub[cli]"

echo "==> smoke test"
.venv-cuda/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "->", torch.cuda.get_device_name(0))
import llmcompressor, transformers, lm_eval
print("llmcompressor", llmcompressor.__version__)
print("transformers", transformers.__version__)
PY

echo "==> installed versions (record these in README.md)"
uv pip freeze --python .venv-cuda | grep -Ei '^(torch|transformers|llmcompressor|compressed-tensors|lm[-_]eval|datasets|accelerate)=' || true

if [ -z "${HF_TOKEN:-}" ]; then
  echo
  echo "HF_TOKEN is not set. Llama 3.1 is a gated repo - export it before downloading:"
  echo "  export HF_TOKEN=hf_..."
fi

echo
echo "Done. Next:"
echo "  source .venv-cuda/bin/activate"
echo "  python fetch_wikitext.py"
echo "  hf download meta-llama/Llama-3.1-8B-Instruct"

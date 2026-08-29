#!/usr/bin/env bash
#
# The whole AWQ/GPTQ run, unattended, on a rented NVIDIA box.
#
# This exists for one reason: GPU time is metered and interactive debugging is
# the expensive way to find a bug. Every phase runs fail-fast, every phase is
# timed, and the running cost is printed as it accrues.
#
#   Usage:  export HF_TOKEN=hf_...
#           bash run_cuda_all.sh                 # everything
#           bash run_cuda_all.sh --dry-run       # print the plan, run nothing
#           HOURLY_USD=0.80 bash run_cuda_all.sh # spot pricing in the cost line
#
# RESUMABLE. Every phase checks results.json first and skips what is already
# measured, because results_io.update() merges per label and per metric. A spot
# reclaim, a dropped SSH session or an OOM costs you one phase, not the run.
# Re-run the same command to continue. Use --force to remeasure regardless.
#
# THE fp16 GATE. Phase 2 stops the run if the fp16-cuda perplexity does not
# match the 7.3648 the same loop produced on the M4 Pro. If the CUDA path
# differs, every number measured afterwards inherits that difference, and it is
# much cheaper to find out in phase 2 than in the essay.

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv-cuda/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct
FP16_REFERENCE=7.3648        # fp16-torch, measured on MPS by this same loop
FP16_TOLERANCE=0.01          # 1%; fp16 kernel differences across backends are ~0.4%
HOURLY_USD="${HOURLY_USD:-1.86}"   # g6e.xlarge on-demand, us-east-1
THROUGHPUT_REPEATS="${THROUGHPUT_REPEATS:-5}"

# results.json bookkeeping needs nothing but the stdlib, so it must not depend on
# the CUDA venv existing. If it did, a missing venv would make have() fail for
# every phase, which reads as "nothing measured yet" and silently reruns the lot.
HELPER_PY=$([ -x "$PY" ] && echo "$PY" || echo python3)

DRY_RUN=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

START=$(date +%s)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

elapsed_cost() {
  local now secs
  now=$(date +%s); secs=$((now - START))
  printf '   [%dm %02ds elapsed, ~$%.2f]\n' \
    $((secs / 60)) $((secs % 60)) \
    "$(awk -v s="$secs" -v h="$HOURLY_USD" 'BEGIN{print s/3600*h}')"
}

# Has this label already got this metric in results.json?
have() {
  [ "$FORCE" = 1 ] && return 1
  $HELPER_PY - "$1" "$2" <<'PY'
import json, sys, pathlib
f = pathlib.Path("results/results.json")
if not f.exists(): sys.exit(1)
d = json.loads(f.read_text())
sys.exit(0 if sys.argv[2] in d.get(sys.argv[1], {}) else 1)
PY
}

# phase <label> <metric> <dir> <command...>
phase() {
  local label="$1" metric="$2" dir="$3"; shift 3
  if have "$label" "$metric"; then
    echo "   skip  $label/$metric (already in results.json)"
    return 0
  fi
  say "$label / $metric"
  if [ "$DRY_RUN" = 1 ]; then
    echo "   would run (in $dir): $*"
    return 0
  fi
  ( cd "$dir" && "$@" )
  elapsed_cost
}

# ---------------------------------------------------------------- preflight
say "preflight"
if [ "$DRY_RUN" = 0 ]; then
  [ -x "$PY" ] || { echo "no $PY - run: bash setup_cuda.sh" >&2; exit 1; }
  [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set; $MODEL is gated" >&2; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  [ -f data/wikitext-2-raw-test.txt ] || $PY fetch_wikitext.py
  $PY -c "import llmcompressor, lm_eval, transformers; print('llmcompressor', llmcompressor.__version__)"
fi
echo "   pricing: \$$HOURLY_USD/hr, throughput repeats: $THROUGHPUT_REPEATS"

# ------------------------------------------------------- 1. fp16 control
# Runs first and on the same GPU. Without it the AWQ and GPTQ tokens/sec
# numbers have no baseline they can legitimately be read against.
phase fp16-cuda perplexity fp16 \
  ../$PY perplexity_torch.py --model "$MODEL" --label fp16-cuda --device cuda --dtype float16

if [ "$DRY_RUN" = 0 ]; then
  say "fp16 gate"
  $HELPER_PY - "$FP16_REFERENCE" "$FP16_TOLERANCE" <<'PY'
import json, sys, pathlib
ref, tol = float(sys.argv[1]), float(sys.argv[2])
got = json.loads(pathlib.Path("results/results.json").read_text())["fp16-cuda"]["perplexity"]["ppl"]
drift = abs(got - ref) / ref
print(f"   fp16-cuda ppl = {got:.4f}, reference {ref:.4f}, drift {drift * 100:.2f}%")
if drift > tol:
    print(f"   STOP: drift exceeds {tol * 100:.0f}%. The CUDA path differs from the "
          f"loop that produced the reference; every later number inherits it.")
    sys.exit(1)
print("   gate passed")
PY
fi

phase fp16-cuda mmlu fp16 \
  ../$PY mmlu_torch.py --model "$MODEL" --label fp16-cuda --device cuda --dtype float16
phase fp16-cuda throughput fp16 \
  ../$PY throughput_torch.py --model "$MODEL" --label fp16-cuda --device cuda --dtype float16 \
  --repeats "$THROUGHPUT_REPEATS"

# ------------------------------------------------------------- 2. AWQ
# Smoke test first. 8 samples at 512 tokens exercises every line of the real
# run - the modifiers, oneshot, save_compressed - in about two minutes. An
# llmcompressor API change is the likeliest failure here and this is the cheap
# place to hit it. Its output is thrown away and never reaches results.json.
if [ "$FORCE" = 1 ] || ! have awq-q4 perplexity; then
  say "awq / smoke test"
  if [ "$DRY_RUN" = 1 ]; then
    echo "   would run the 8-sample AWQ smoke test"
  else
    ( cd awq && ../$PY quantize.py --hf-dir "$MODEL" --num-samples 8 --max-seq-len 512 \
        --out-dir /tmp/awq-smoke --label awq-smoke )
    $HELPER_PY -c "
import json, pathlib
f = pathlib.Path('results/results.json'); d = json.loads(f.read_text())
d.pop('awq-smoke', None); f.write_text(json.dumps(d, indent=2))
print('   smoke passed, awq-smoke dropped from results.json')"
    rm -rf /tmp/awq-smoke
    elapsed_cost
  fi
fi

AWQ_DIR=../models/llama-3.1-8b-instruct-awq-q4
if [ "$FORCE" = 1 ] || [ ! -d "models/llama-3.1-8b-instruct-awq-q4" ]; then
  say "awq / quantize (256 samples, 2048 tokens)"
  [ "$DRY_RUN" = 1 ] && echo "   would quantize with AWQ" || { ( cd awq && ../$PY quantize.py --hf-dir "$MODEL" ); elapsed_cost; }
fi

phase awq-q4 perplexity awq ../$PY perplexity.py --model "$AWQ_DIR" --label awq-q4 --device cuda
phase awq-q4 mmlu       awq ../$PY mmlu.py       --model "$AWQ_DIR" --label awq-q4 --device cuda
phase awq-q4 throughput awq ../$PY throughput.py --model "$AWQ_DIR" --label awq-q4 --device cuda \
  --repeats "$THROUGHPUT_REPEATS"

# ------------------------------------------------------------- 3. GPTQ
GPTQ_DIR=../models/llama-3.1-8b-instruct-gptq-q4
if [ "$FORCE" = 1 ] || [ ! -d "models/llama-3.1-8b-instruct-gptq-q4" ]; then
  say "gptq / quantize (256 samples, 2048 tokens)"
  [ "$DRY_RUN" = 1 ] && echo "   would quantize with GPTQ" || { ( cd gptq && ../$PY quantize.py --hf-dir "$MODEL" ); elapsed_cost; }
fi

phase gptq-q4 perplexity gptq ../$PY perplexity.py --model "$GPTQ_DIR" --label gptq-q4 --device cuda
phase gptq-q4 mmlu       gptq ../$PY mmlu.py       --model "$GPTQ_DIR" --label gptq-q4 --device cuda
phase gptq-q4 throughput gptq ../$PY throughput.py --model "$GPTQ_DIR" --label gptq-q4 --device cuda \
  --repeats "$THROUGHPUT_REPEATS"

# ----------------------------------------------------------------- done
say "done"
[ "$DRY_RUN" = 0 ] && $HELPER_PY -c "
import json
d = json.load(open('results/results.json'))
for label in ['fp16-cuda', 'awq-q4', 'gptq-q4']:
    row = d.get(label, {})
    ppl = row.get('perplexity', {}).get('ppl')
    acc = row.get('mmlu', {}).get('accuracy')
    tg  = row.get('throughput', {}).get('tg_tokens_per_sec')
    size = row.get('size_gb')
    print(f\"  {label:10s} ppl={ppl if ppl is None else round(ppl, 4)!s:8s} \"
          f\"mmlu={acc!s:6s} tg={tg if tg is None else round(tg, 1)!s:7s} \"
          f\"size={size if size is None else round(size, 2)!s} GB\")
"
elapsed_cost
cat <<'EOF'

   Copy the results back, then TERMINATE the instance (not stop):
     scp <host>:~/llm-inference-experiments/model-quantization/results/results.json .
EOF

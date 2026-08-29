#!/usr/bin/env bash
#
# 200-question MMLU for the three checkpoints that live on the CUDA box.
#
# WHY ONLY THESE THREE. awq-q4 and gptq-q4 exist nowhere else: they are built on
# the rented GPU and destroyed with it, so this is the only chance to score them
# at any question count without paying to rebuild them. fp16-cuda comes along
# because a baseline measured by different code than the thing it baselines is
# not a baseline. Every other checkpoint in this project already exists on the
# Mac and can be re-scored there for free, on the same engines that produced the
# published numbers, which is the better trade than shipping 28GB up here.
#
# WHY A SEPARATE KEY. Results land under `mmlu_200`, never `mmlu`. The 50-question
# table stays intact and internally comparable until every checkpoint has a
# 200-question number. A table mixing the two would be the same class of error as
# the perplexity-convention bug this project already shipped once.
#
# The subset nests: lm_eval's --limit is a positional slice with no shuffling, so
# 20-per-task contains the same 5-per-task questions plus 15 more. The 50-question
# result is recoverable from this run's raw output rather than being a different
# sample of the same benchmark.
#
#   Usage:  bash run_mmlu_200.sh

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv-cuda/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct
LIMIT=20            # 20 per task x 10 tasks = 200 questions
KEY=mmlu_200
HOURLY_USD="${HOURLY_USD:-2.25}"
HELPER_PY=$([ -x "$PY" ] && echo "$PY" || echo python3)

START=$(date +%s)
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
elapsed_cost() {
  local secs=$(( $(date +%s) - START ))
  printf '   [%dm %02ds elapsed, ~$%.2f]\n' $((secs/60)) $((secs%60)) \
    "$(awk -v s="$secs" -v h="$HOURLY_USD" 'BEGIN{print s/3600*h}')"
}
have() {
  $HELPER_PY - "$1" "$2" <<'PY'
import json, sys, pathlib
f = pathlib.Path("results/results.json")
if not f.exists(): sys.exit(1)
sys.exit(0 if sys.argv[2] in json.loads(f.read_text()).get(sys.argv[1], {}) else 1)
PY
}

# phase <label> <dir> <script> <model-arg> [extra...]
phase() {
  local label="$1" dir="$2" script="$3" model="$4"; shift 4
  if have "$label" "$KEY"; then echo "   skip  $label/$KEY (already measured)"; return 0; fi
  say "$label / $KEY (200 questions)"
  ( cd "$dir" && ../$PY "$script" --model "$model" --label "$label" --device cuda \
      --limit-per-task "$LIMIT" --metric-key "$KEY" "$@" )
  elapsed_cost
}

[ -x "$PY" ] || { echo "no $PY - run setup_cuda.sh first" >&2; exit 1; }

phase fp16-cuda fp16 mmlu_torch.py "$MODEL" --dtype float16
phase awq-q4    awq  mmlu.py       ../models/llama-3.1-8b-instruct-awq-q4
phase gptq-q4   gptq mmlu.py       ../models/llama-3.1-8b-instruct-gptq-q4

say "200-question results"
$HELPER_PY -c "
import json
d = json.load(open('results/results.json'))
print(f\"  {'label':10s} {'50q':>7s} {'200q':>7s}  {'n':>4s}\")
for k in ['fp16-cuda', 'awq-q4', 'gptq-q4']:
    r = d.get(k, {})
    a = r.get('mmlu', {}).get('accuracy')
    b = r.get('mmlu_200', {})
    print(f\"  {k:10s} {'' if a is None else format(a,'.3f'):>7s} \"
          f\"{'' if not b else format(b.get('accuracy',0),'.3f'):>7s}  {b.get('n_questions','')!s:>4s}\")
"
elapsed_cost

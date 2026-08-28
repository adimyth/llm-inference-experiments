"""The fixed MMLU subset, defined exactly once.

Every checkpoint in this project is scored on the *same* 50 questions, so an accuracy difference between two checkpoints is attributable to quantization rather than to which questions got asked. That guarantee only holds if there is one definition of the subset, which is why this list lives at the root and every method's mmlu.py imports it rather than restating it.

10 subtasks, 5 questions each: 3 STEM, 3 humanities, 2 social sciences, 2 other, following MMLU's own category grouping.

`lm_eval`'s `--limit N` takes a plain positional slice of each task's dataset - verified against its source, there is no shuffling anywhere in the pipeline - so this really is the same 50 questions every run, not a random sample that happens to differ each time.

Cut down from an original 500-question, 20-task design after measuring the real per-request cost. Each question needs 4 requests, one per answer choice, and at roughly 5s per request (dominated by lm_eval reprocessing the few-shot prompt, not by model size) 500 questions would have taken multiple hours per checkpoint, times seven checkpoints. 50 questions trades statistical robustness for something that finishes. Treat single-question differences (2 percentage points) as noise.
"""

TASKS = [
    # STEM
    "mmlu_high_school_physics",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_biology",
    # humanities
    "mmlu_high_school_us_history",
    "mmlu_philosophy",
    "mmlu_world_religions",
    # social sciences
    "mmlu_high_school_geography",
    "mmlu_high_school_psychology",
    # other
    "mmlu_professional_medicine",
    "mmlu_marketing",
]
LIMIT_PER_TASK = 5  # 10 tasks x 5 = 50 questions

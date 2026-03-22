#!/bin/bash
# Submit all no-title inference jobs.
# Run from: experiments/data_leakage/slurm/
# Requires OPENAI_API_KEY (and optionally OPENAI_ORG_ID) in the environment for evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in \
    no_title_qwen3-0.6b.sh \
    no_title_qwen3-1.7b.sh \
    no_title_qwen3-4b.sh \
    no_title_qwen3-8b.sh \
    no_title_gemma-3-270m.sh \
    no_title_gemma-3-1b.sh \
    no_title_gemma-3-4b.sh \
    no_title_gemma-3-12b.sh \
    no_title_gemma-3-27b.sh \
; do
    echo "Submitting $script"
    sbatch "$SCRIPT_DIR/$script"
done

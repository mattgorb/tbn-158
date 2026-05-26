#!/usr/bin/env bash
# End-to-end smoke test on the 200M model. Runs exactly 1 optimizer step
# (= 16 micro-batches under the default grad_accum=16), saves a checkpoint,
# and runs one full eval round against the SlimPajama validation split.
#
# Expected wall-clock on 1× 48 GB GPU: ~45-60 seconds total.
#
# Outputs land in outputs/smoke_test/ (separate from real runs).
#
# Usage: bash scripts/smoke_test.sh [variant]
#   variant ∈ {plain, bitnet158, tbn158}, default tbn158
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${1:-tbn158}"

python pretrain.py \
    --size 200M --variant "${VARIANT}" \
    --steps 1 --warmup_steps 0 \
    --save_steps 1 --eval_steps 1 \
    --logging_steps 1 \
    --output_dir outputs/smoke_test \
    --run_name "smoke_test_${VARIANT}"

echo
echo "✓ Smoke test finished."
echo "  Checkpoint: outputs/smoke_test/checkpoint-1/"
echo "  Inspect in WandB run: smoke_test_${VARIANT}"
echo
echo "  Verify in WandB:"
echo "   - train/loss logged at step 1"
echo "   - train/num_input_tokens_seen ≈ 262K"
echo "   - eval/loss and eval/perplexity logged"

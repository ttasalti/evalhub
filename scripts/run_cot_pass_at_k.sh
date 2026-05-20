#!/usr/bin/env bash
# ============================================================================
# DEPRECATED — kept for backward compatibility.
#
# The end-to-end CoT-Pass@K pipeline has been renamed for parity with the
# split scripts `run_eval_only.sh` and `run_judge_only.sh`. This shim
# delegates to `run_end_to_end.sh` so existing callers keep working.
# ============================================================================
set -euo pipefail
echo "[DEPRECATED] scripts/run_cot_pass_at_k.sh — use scripts/run_end_to_end.sh" >&2
exec "$(dirname "$0")/run_end_to_end.sh" "$@"

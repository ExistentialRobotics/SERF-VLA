#!/usr/bin/env bash
set -euo pipefail

# Example scripts:
#   bash scripts/train/train_2d_image.sh
#   bash scripts/train/train_2d_image.sh --task-id 0026 --batch-size 8 --num-train-steps 40000

CONFIG_NAME="pi_behavior_b1k_fast--50t_lora"
TASK_ID="0021"
BATCH_SIZE=16
NUM_TRAIN_STEPS=20000
EXTRA_ARGS=()

usage() {
    sed -n '4,7p' "$0"
    cat <<'EOF'

Options:
  --task-id ID              Single BEHAVIOR task id, e.g. 0021.
  --batch-size N            Global train batch size.
  --num-train-steps N       Number of training steps.
  --                         Pass remaining args to scripts/train_b1k.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-train-steps)
            NUM_TRAIN_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TASK_ID" ]]; then
    echo "Task id is required." >&2
    exit 1
fi

EXP_NAME="${CONFIG_NAME}/task_${TASK_ID}--bs_${BATCH_SIZE}--iter_${NUM_TRAIN_STEPS}/$(date +%Y.%m.%d_%H:%M:%S)"

uv run python scripts/train_b1k.py "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --task_ids "$TASK_ID" \
    "${EXTRA_ARGS[@]}"

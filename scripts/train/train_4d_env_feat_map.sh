#!/usr/bin/env bash
set -euo pipefail

# Example scripts:
#   bash scripts/train/train_4d_env_feat_map.sh
#   bash scripts/train/train_4d_env_feat_map.sh --task-id task-0026 --batch-size 8 --num-train-steps 40000

CONFIG_PREFIX="pi_serf_behavior_b1k_fast--4d_env_feat_map--50t_lora"
TASK_ID="task-0021"
BATCH_SIZE=16
NUM_TRAIN_STEPS=20000
EXTRA_ARGS=()

usage() {
    sed -n '4,7p' "$0"
    cat <<'EOF'

Options:
  --task-id ID              Task id in task-XXXX or XXXX form.
  --batch-size N            Global train batch size.
  --num-train-steps N       Number of training steps.
  --                         Pass remaining args to scripts/train_serf_b1k.py.
EOF
}

normalize_task_id() {
    local raw="${1#task-}"
    printf "task-%04d" "$((10#$raw))"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-id)
            TASK_ID="$(normalize_task_id "$2")"
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

CONFIG_NAME="${CONFIG_PREFIX}--${TASK_ID}"
EXP_NAME="${CONFIG_NAME}/bs_${BATCH_SIZE}--iter_${NUM_TRAIN_STEPS}/$(date +%Y.%m.%d_%H:%M:%S)"

uv run python scripts/train_serf_b1k.py "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    "${EXTRA_ARGS[@]}"

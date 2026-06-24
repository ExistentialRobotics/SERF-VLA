#!/usr/bin/env bash
set -euo pipefail

# Example scripts:
#   bash scripts/test/test_2d_image_pre.sh
#   bash scripts/test/test_2d_image_pre.sh --task-id 0026
#   bash scripts/test/test_2d_image_pre.sh --logging-tag pretrained_eval --write-video false -- eval_instance_ids='[0,1,2]'

CONDA_ENV_PYTHON_PATH="/mnt/hdd/miniconda3/envs/behavior/bin/python"

TASK_ID="0021"
CONFIG_NAME="pi_behavior_b1k_fast"
LOGGING_TAG="pretrained_eval"
CHECKPOINT_PATH="checkpoints/behavior-1k-solution/behavior_50t_checkpoint"

RECORD_STEP_Q_SCORE=true
WRITE_VIDEO=true
WRITE_THIRD_PERSON_VIDEO=true
SERVER_LOG="logs/_server/serve_b1k_pretrained.log"
EVAL_ARGS=()

usage() {
    sed -n '4,8p' "$0"
    cat <<'EOF'

Options:
  --task-id ID                    Task id used in log names, e.g. 0021.
  --checkpoint-path PATH          Checkpoint directory served by the policy server.
  --logging-tag TAG               Suffix used in the evaluation log path.
  --conda-env-python-path PATH    Python executable for OmniGibson eval.py.
  --record-step-q-score BOOL      Enable or disable step Q-score recording.
  --write-video BOOL              Enable or disable first-person video writing.
  --write-third-person-video BOOL Enable or disable third-person video writing.
  --server-log PATH               Policy server log file.
  --                              Pass remaining args to OmniGibson eval.py.
EOF
}

task_name_from_id() {
    local raw="${1#task-}"
    local numeric
    numeric="$(printf "%d" "$((10#$raw))")"

    case "$numeric" in
        21) echo "collecting_childrens_toys" ;;
        22) echo "putting_shoes_on_rack" ;;
        26) echo "assembling_gift_baskets" ;;
        *)
            echo "Unsupported task id: $1. Expected one of 0021, 0022, 0026." >&2
            return 1
            ;;
    esac
}

patch_task_0021_bddl_goal() {
    local raw="${1#task-}"
    local numeric
    numeric="$(printf "%d" "$((10#$raw))")"
    if [[ "$numeric" == "21" ]]; then
        python3 scripts/setup/patch_collecting_childrens_toys_bddl.py
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --logging-tag)
            LOGGING_TAG="$2"
            shift 2
            ;;
        --conda-env-python-path)
            CONDA_ENV_PYTHON_PATH="$2"
            shift 2
            ;;
        --record-step-q-score)
            RECORD_STEP_Q_SCORE="$2"
            shift 2
            ;;
        --write-video)
            WRITE_VIDEO="$2"
            shift 2
            ;;
        --write-third-person-video)
            WRITE_THIRD_PERSON_VIDEO="$2"
            shift 2
            ;;
        --server-log)
            SERVER_LOG="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EVAL_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

TASK_NAME="$(task_name_from_id "$TASK_ID")"
patch_task_0021_bddl_goal "$TASK_ID"
LOGGING_NAME="${CONFIG_NAME}/task_${TASK_ID}--${LOGGING_TAG}"
mkdir -p "$(dirname "$SERVER_LOG")"

SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[INFO] Stopping policy server (PID=${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        sleep 2

        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[WARN] Force killing policy server (PID=${SERVER_PID})..."
            kill -9 "${SERVER_PID}" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting policy server..."
uv run python scripts/serve_b1k.py policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CHECKPOINT_PATH}" \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "[INFO] Policy server started in background. PID=${SERVER_PID}"
echo "[INFO] Server log: ${SERVER_LOG}"

sleep 10

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Policy server exited early. Check server log:"
    tail -n 100 "${SERVER_LOG}" || true
    exit 1
fi

echo "[INFO] Running evaluation..."
set +e
"${CONDA_ENV_PYTHON_PATH}" BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py \
    log_path="./logs/${LOGGING_NAME}" \
    policy=websocket \
    model.host=localhost \
    task.name="${TASK_NAME}" \
    write_video="${WRITE_VIDEO}" \
    write_third_person_video="${WRITE_THIRD_PERSON_VIDEO}" \
    record_step_q_score="${RECORD_STEP_Q_SCORE}" \
    "${EVAL_ARGS[@]}"

EVAL_EXIT_CODE=$?
set -e
echo "[INFO] Evaluation finished with exit code ${EVAL_EXIT_CODE}"
exit "${EVAL_EXIT_CODE}"

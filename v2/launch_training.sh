#!/usr/bin/env bash
# Unified launcher for both single-node GPU and multi-worker distributed training
# Fixed: worker logs not appearing due to SSH quoting issue
# Author: Kamal Giri (updated for reliability and clarity)

set -euo pipefail
cd "$(dirname "$0")"
source cluster.env

# Ensure directories exist
mkdir -p logs models plots

echo "Select training mode:"
echo "  1) Single-node GPU training"
echo "  2) Multi-worker distributed training"
read -p "Enter choice [1/2]: " MODE_CHOICE

# Common input prompts (with defaults)
read -p "Dataset path [default: $DATA_FILE]: " DATA_FILE_IN
read -p "Epochs [default: $EPOCHS]: " EPOCHS_IN
read -p "Batch size [default: $BATCH_SIZE]: " BATCH_IN
read -p "Learning rate [default: $LR]: " LR_IN
read -p "Input size [default: $INPUT_SIZE]: " INPUT_IN

DATA_FILE="${DATA_FILE_IN:-$DATA_FILE}"
EPOCHS="${EPOCHS_IN:-$EPOCHS}"
BATCH_SIZE="${BATCH_IN:-$BATCH_SIZE}"
LR="${LR_IN:-$LR}"
INPUT_SIZE="${INPUT_IN:-$INPUT_SIZE}"
DATASET_NAME=$(basename "$DATA_FILE" .csv)
RUN_ID=$(date +"%Y%m%d_%H%M%S")


# SINGLE-NODE GPU TRAINING
if [[ "$MODE_CHOICE" == "1" ]]; then
    echo "[INFO] Launching SINGLE-NODE GPU training locally..."
    source ../.venv/bin/activate || { echo "venv not found"; exit 1; }

    LOG_FILE="logs/local_gpu_${DATASET_NAME}_${RUN_ID}.log"
    python3 main.py --mode local-gpu \
        --data "$DATA_FILE" \
        --input_size "$INPUT_SIZE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" 2>&1 | tee "$LOG_FILE"

    echo "[DONE] Local GPU training complete. Logs saved to $LOG_FILE"
    exit 0
fi

#MULTI-WORKER DISTRIBUTED TRAINING
if [[ "$MODE_CHOICE" == "2" ]]; then
    echo "[INFO] Launching DISTRIBUTED training..."
    echo "Chief : $CHIEF_HOST"
    echo "Workers: ${WORKERS[*]}"

    # Clean previous processes
    for HOST in "$CHIEF_HOST" "${WORKERS[@]}"; do
        echo "[CLEANUP] $HOST"
        ssh -o BatchMode=yes "$SSH_USER@$HOST" "pkill -f 'python3 main.py' || true" || echo "Skipped cleanup on $HOST"
    done

    #Assign random available port for coordination service
    PORT=$((12000 + RANDOM % 5000))
    echo "[INFO] Using port: $PORT"

    #Sync source code to all hosts
    for HOST in "$CHIEF_HOST" "${WORKERS[@]}"; do
        echo "[SYNC] Syncing code to $HOST..."
        ssh "$SSH_USER@$HOST" "mkdir -p '$REMOTE_CODE_DIR/logs' '$REMOTE_CODE_DIR/models' '$REMOTE_CODE_DIR/plots'"
        scp -r src "$SSH_USER@$HOST:$REMOTE_CODE_DIR/" >/dev/null
        scp main.py cluster.env "$SSH_USER@$HOST:$REMOTE_CODE_DIR/" >/dev/null
    done

    #Build TF_CONFIG for cluster definition
    WORKER_ENTRIES=""
    for w in "${WORKERS[@]}"; do
        WORKER_ENTRIES+="\"$w:$PORT\","
    done
    WORKER_ENTRIES="${WORKER_ENTRIES%,}"  # trim trailing comma

    TFCONF_CHIEF=$(cat <<JSON
{"cluster":{"chief":["$CHIEF_HOST:$PORT"],"worker":[${WORKER_ENTRIES}]},"task":{"type":"chief","index":0}}
JSON
)

    #Launch Chief
    CHIEF_LOG="$REMOTE_CODE_DIR/logs/chief_${DATASET_NAME}_${RUN_ID}.log"
    echo "[LAUNCH] Chief on $CHIEF_HOST (log: $CHIEF_LOG)"
    ssh "$SSH_USER@$CHIEF_HOST" "
        cd '$REMOTE_CODE_DIR' &&
        source ../.venv/bin/activate &&
        export TF_CONFIG='$TFCONF_CHIEF' &&
        export PYTHONUNBUFFERED=1 &&
        stdbuf -oL -eL python3 -u main.py --mode distributed \
            --data '$DATA_FILE' \
            --input_size $INPUT_SIZE \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LR \
            --rank 0 \
            --cluster '$CHIEF_HOST ${WORKERS[*]}' \
            2>&1 | tee '$CHIEF_LOG'
    " &
    sleep 10  # give chief time to start gRPC service

    #aunch Workers
    for i in "${!WORKERS[@]}"; do
        WORKER_HOST=${WORKERS[$i]}
        WORKER_LOG="$REMOTE_CODE_DIR/logs/worker_${i}_${DATASET_NAME}_${RUN_ID}.log"
        echo "[LAUNCH] Worker $i on $WORKER_HOST (log: $WORKER_LOG)"

        TFCONF_WORKER=$(cat <<JSON
{"cluster":{"chief":["$CHIEF_HOST:$PORT"],"worker":[${WORKER_ENTRIES}]},"task":{"type":"worker","index":$i}}
JSON
)
        ssh "$SSH_USER@$WORKER_HOST" "
            cd '$REMOTE_CODE_DIR' &&
            source ../.venv/bin/activate &&
            export TF_CONFIG='$TFCONF_WORKER' &&
            export PYTHONUNBUFFERED=1 &&
            stdbuf -oL -eL python3 -u main.py --mode distributed \
                --data '$DATA_FILE' \
                --input_size $INPUT_SIZE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --rank $((i+1)) \
                --cluster '$CHIEF_HOST ${WORKERS[*]}' \
                2>&1 | tee '$WORKER_LOG'
        " &
    done

    echo
    echo "[INFO] Distributed training started successfully."
    echo "Chief log   : $CHIEF_LOG"
    echo "Worker logs : $REMOTE_CODE_DIR/logs/worker_*_${RUN_ID}.log"
    echo "To follow chief log live:"
    echo "   ssh $SSH_USER@$CHIEF_HOST 'tail -f $CHIEF_LOG'"
    echo
    exit 0
fi

echo "[ERROR] Invalid choice. Please enter 1 or 2."
exit 1

# launch_training.sh
# Unified launcher for both single-node GPU and distributed training.


set -euo pipefail
cd "$(dirname "$0")"
source cluster.env

# Ensure local log and model directories exist
mkdir -p logs models plots


echo "Unified Training Launcher"
echo "Choose training mode:"
echo "  1) Single-node GPU training"
echo "  2) Multi-worker distributed training"
read -p "Enter choice [1/2]: " MODE_CHOICE


# Common argument prompts
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
DATASET_NAME = $(basename "$DATA_FILE" .csv)
RUN_ID = $(date + "%Y%m%d_%H%M%S" )


#Single-node GPU training

if [[ "$MODE_CHOICE" == "1" ]]; then
    echo "Launching SINGLE-NODE GPU training locally..."
    source ../.venv/bin/activate
    LOG_FILE= "$REMOTE_CODE_DIR/logs/local_gpu_${DATASET_NAME}_${RUN_ID}.log"

    python3 main.py --mode local-gpu \
        --data "$DATA_FILE" \
        --input_size "$INPUT_SIZE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" | tee "$LOG_FILE"

    echo "Local GPU training complete. Logs saved to $LOG_FILE."
    exit 0
fi


#Distributed training

if [[ "$MODE_CHOICE" == "2" ]]; then
    echo "Launching DISTRIBUTED multi-worker training..."
    echo "Chief:  $CHIEF_HOST"
    echo "Worker: ${WORKERS[*]}"
    CHIEF_LOG="$REMOTE_CODE_DIR/logs/chief_${DATASET_NAME}_${RUN_ID}.log"

    # Sync files (ensure same code and env everywhere)
    for HOST in "$CHIEF_HOST" "${WORKERS[@]}"; do
        echo "Syncing code to $HOST ..."
        ssh "$SSH_USER@$HOST" "mkdir -p '$REMOTE_CODE_DIR/logs' '$REMOTE_CODE_DIR/models' '$REMOTE_CODE_DIR/plots'"
        scp -r src "$SSH_USER@$HOST:$REMOTE_CODE_DIR/"
        scp main.py cluster.env "$SSH_USER@$HOST:$REMOTE_CODE_DIR/"
    done

    # Create TF_CONFIG JSON DYNAMICALLY
    #{mteverest1.uwyo.edu:12345, ..., ....}
    WORKER_ENTRIES=""
    for w in "${WORKERS[@]}"; do
        WORKER_ENTRIES+="\"$w:$PORT\","
    done
    WORKER_ENTRIES="${WORKER_ENTRIES%,}"  # remove trailing comma

    TFCONF_CHIEF=$(cat <<JSON
{"cluster":{"chief":["$CHIEF_HOST:$PORT"],"worker":[${WORKER_ENTRIES}]},"task":{"type":"chief","index":0}}
JSON
)

    # Launch CHIEF
    echo "Starting chief on $CHIEF_HOST..."
    #> '$REMOTE_CODE_DIR/logs/chief.log' 2>&1 &
    ssh "$SSH_USER@$CHIEF_HOST" "
        mkdir -p '$REMOTE_CODE_DIR/logs';
        cd '$REMOTE_CODE_DIR';
        source ../.venv/bin/activate;
        echo 'Running chief on $(hostname)...';
        TF_CONFIG='$TFCONF_CHIEF' python3 main.py --mode distributed \
            --data '$DATA_FILE' \
            --input_size $INPUT_SIZE \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LR \
            --rank 0 \
            --cluster '$CHIEF_HOST' '${WORKERS[*]}' \
            | tee '$CHIEF_LOG'            
    " 

    sleep 3

    # Launch WORKER
    for i in "${!WORKERS[@]}"; do
        WORKER_HOST=${WORKERS[$i]}
        TFCONF_WORKER=$(cat <<JSON
{"cluster":{"chief":["$CHIEF_HOST:$PORT"],"worker":[${WORKER_ENTRIES}]},"task":{"type":"worker","index":$i}}
JSON
)
      #| tee '$REMOTE_CODE_DIR/logs/chief.log' 
        echo "Starting worker $i on $WORKER_HOST..."
        WORKER_LOG="$REMOTE_CODE_DIR/logs/worker_${i}_${DATASET_NAME}_${RUN_ID}.log"
        ssh "$SSH_USER@$WORKER_HOST" "
            mkdir -p '$REMOTE_CODE_DIR/logs';
            cd '$REMOTE_CODE_DIR';
            source ../.venv/bin/activate;
            echo 'Running WORKER-$i on $(hostname)...';
            TF_CONFIG='$TFCONF_WORKER' python3 main.py --mode distributed \
                --data '$DATA_FILE' \
                --input_size $INPUT_SIZE \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --rank $((i+1)) \
                --cluster '$CHIEF_HOST' ${WORKERS[*]} \
                | tee '$WORKER_LOG'
                
        "
    done

    echo " Distributed training started!"
    echo "Chief log:   $REMOTE_CODE_DIR/logs/chief.log"
    echo "Worker logs: $REMOTE_CODE_DIR/logs/worker_*.log"
    echo "To follow chief live:"
    echo "   ssh $SSH_USER@$CHIEF_HOST 'tail -f $REMOTE_CODE_DIR/logs/chief.log'"
    exit 0
fi

echo "Invalid choice. Please enter 1 or 2."
exit 1

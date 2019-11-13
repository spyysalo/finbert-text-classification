#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p gpu
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002085
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

OUTPUT_DIR="output/$SLURM_JOBID"

function on_exit {
    rm -rf "$OUTPUT_DIR"
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 model data_dir seq_len batch_size learning_rate epochs"
    exit 1
fi

MODEL="$1"
DATA_DIR="$2"
MAX_SEQ_LENGTH="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
EPOCHS="$6"

VOCAB="$(dirname "$MODEL")/vocab.txt"
CONFIG="$(dirname "$MODEL")/bert_config.json"

if [[ $MODEL =~ "uncased" ]]; then
    lower_case="true"
elif [[ $MODEL =~ "multilingual" ]]; then
    lower_case="true"
else
    lower_case="false"
fi

if [[ $DATA_DIR =~ "ylilauta" ]]; then
    task_name="ylilauta"
elif [[ $DATA_DIR =~ "yle" ]]; then
    task_name="yle"
else
    echo "Error: can't detemine task from data dir $DATA_DIR"
    exit 1
fi

rm -rf "OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

rm -f latest.out latest.err
ln -s logs/$SLURM_JOBID.out latest.out
ln -s logs/$SLURM_JOBID.err latest.err

module purge
module load tensorflow
source $HOME/venv/keras-bert/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun python run_classifier.py \
    --task_name "$task_name" \
    --do_train=true \
    --do_eval=true \
    --bert_config_file "$CONFIG" \
    --init_checkpoint "$MODEL" \
    --vocab_file "$VOCAB" \
    --do_lower_case="$lower_case" \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"

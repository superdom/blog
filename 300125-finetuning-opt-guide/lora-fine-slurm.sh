#!/bin/sh
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=xxxxx
#SBATCH --output=logs/%j.%x.log

MODEL_ID="google/gemma-2-2b"
#MODEL_ID="google/gemma-2-2b-it"
#MODEL_ID="meta-llama/Llama-3.2-3B"
#MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
#MODEL_ID="Qwen/Qwen2.5-3B"
#MODEL_ID="Qwen/Qwen2.5-3B-Instruct"

MODEL_TYPE="base"
#MODEL_TYPE="it"

BS=8
GC=4
LR=1e-4
LRS="linear"
EP=1
LORAR=8
LORAA=16

export CUDA_VISIBLE_DEVICES=0

module purge
module load Miniconda3/23.9.0-0
source activate py311-torch

srun python -u lora-fine.py --model $MODEL_ID --model_type $MODEL_TYPE --bs $BS --gc $GC --lr $LR --lrs $LRS --ep $EP --lorar $LORAR --loraa $LORAA

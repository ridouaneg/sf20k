#!/bin/bash
#SBATCH --job-name=leak_no_title_qwen3-4b
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.err

module load arch/h100
module load pytorch-gpu/py3/2.6.0

WEIGHTS_DIR=/lustre/fsmisc/dataset/HuggingFace_Models
OUTPUT_DIR=results
DATASETS="movieqa tvqa cinepile infinibench sf20k"
DATASETS_MCQA="movieqa tvqa cinepile"
MODELS="qwen3-0.6b qwen3-1.7b qwen3-14b gemma-3-270m"
N_SUBSAMPLE=256

cd /lustre/fswork/projects/rech/kcn/ucm72yx/code/sf20k
source .venv/bin/activate
cd experiments/data_leakage

for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --n_subsample $N_SUBSAMPLE
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --no_title \
            --n_subsample $N_SUBSAMPLE
    done
    for DATASET in $DATASETS_MCQA; do
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --mcqa \
            --n_subsample $N_SUBSAMPLE
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --mcqa \
            --no_title \
            --n_subsample $N_SUBSAMPLE
    done
done

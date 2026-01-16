#!/bin/bash
#SBATCH --job-name=sf20k
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.err

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/sf20k/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/experiments/instruction_tuning/
wandb offline

CONFIG=qwen25vl3b_32f_100000_lr_1e-5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results/${CONFIG}_${TIMESTAMP}"

# Train
accelerate launch \
    --num_processes 2 \
    --config_file default_config.yaml \
    train.py \
    --config configs/${CONFIG}.yaml \
    --output_dir ${OUTPUT_DIR}

# Run inference
python run_inference.py \
    --output_dir ${OUTPUT_DIR} \
    --data_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_expert.csv \
    --subtitles_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name qwen2.5-vl-3b \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/ \
    --modality vision_language \
    --fps 1.0 \
    --num_frames 32 \
    --n_subsample -1

python run_inference.py \
    --output_dir ${OUTPUT_DIR} \
    --adapter_path ${OUTPUT_DIR}/checkpoint-final/ \
    --data_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_expert.csv \
    --subtitles_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name qwen2.5-vl-3b \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/ \
    --modality vision_language \
    --fps 1.0 \
    --num_frames 32 \
    --n_subsample -1
#!/bin/bash
#SBATCH --job-name=sf20k
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
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

config=test_qwen25vl

# Train
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --config_file default_config.yaml \
    train.py \
    --config configs/${config}.yaml

# Run inference
CUDA_VISIBLE_DEVICES=1 python run_inference.py \
    --output_dir ./results/${config}/ \
    --data_path /users/ghermi/code/sf20k/data/test_expert.csv \
    --subtitles_path /users/ghermi/code/sf20k/data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name qwen2.5-vl-3b \
    --weights_dir /geovic/ghermi/weights/ \
    --modality vision_language \
    --num_frames 8 \
    --fps 1.0 \
    --n_subsample 8

python run_inference.py \
    --output_dir ./results/${config}/ \
    --adapter_path ./results/${config}/checkpoint-final/ \
    --data_path /users/ghermi/code/sf20k/data/test_expert.csv \
    --subtitles_path /users/ghermi/code/sf20k/data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name qwen2.5-vl-3b \
    --weights_dir /geovic/ghermi/weights/ \
    --modality vision_language \
    --num_frames 32 \
    --fps 1.0 \
    --n_subsample 8



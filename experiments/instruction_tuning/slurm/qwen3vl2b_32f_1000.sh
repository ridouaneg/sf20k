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

config_name=qwen3vl2b_32f_1000

# Train
accelerate launch \
    --num_processes 1 \
    --config_file default_config.yaml \
    train.py \
    --config configs/${config_name}.yaml

# Run inference
python run_inference.py \
    --output_dir ./results/${config_name}/ \
    --data_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_expert.csv \
    --subtitles_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name qwen3-vl-2b \
    --weights_dir /lustre/fsn1/projects/rech/kcn/ucm72yx/weights/ \
    --modality vision_language \
    --fps 1.0 \
    --num_frames 32 \
    --n_subsample -1

python run_inference.py \
    --output_dir ./results/${config_name}/ \
    --adapter_path ./results/${config_name}/checkpoint-final/ \
    --data_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_expert.csv \
    --subtitles_path /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name qwen3-vl-2b \
    --weights_dir /lustre/fsn1/projects/rech/kcn/ucm72yx/weights/ \
    --modality vision_language \
    --fps 1.0 \
    --num_frames 32 \
    --n_subsample -1
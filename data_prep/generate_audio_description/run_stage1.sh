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
#SBATCH --array=0-10

N_SAMPLES=10000
START_IDX=$((SLURM_ARRAY_TASK_ID * N_SAMPLES))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * N_SAMPLES))

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/autoad_zero/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data_prep/generate_audio_description/

python run_stage1.py \
    --output_path ./results/stage1_qwen2vl-7b-$START_IDX-$END_IDX.json \
    --data_path /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/questions/train.csv \
    --shots_path /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/shots.parquet \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/ \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --num_frames 8 \
    --start_idx $START_IDX \
    --end_idx $END_IDX

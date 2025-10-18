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
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/movie_star/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/experiments/baselines/

python run_inference.py     --output_path results/qwen2.5-vl-7b_vision_language_128.json     --data_path ../../data/test_expert.csv     --subtitles_path ../../data/test_subtitles.csv     --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/     --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/Qwen/     --model_name qwen2.5-vl-7b     --modality vision_language     --fps 1.0     --max_frames 128

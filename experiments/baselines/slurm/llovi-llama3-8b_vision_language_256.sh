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
cd /lustre/fswork/projects/rech/kcn/ucm72yx/code/sf20k/scripts/

python run_llovi_captioning.py \
    --data_path ../data/test_expert.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --output_path ../data/captions/llovi_qwen2.5-vl-3b_1fps_16f.json \
    --captioner_name qwen2.5-vl-3b \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/

python run_inference.py \
    --model_name llovi-captions+llama \
    --captions_path data/captions/llovi_qwen2.5-vl-3b_1fps_16f.json \
    --modality language \
    
    --output_dir results \
    --data_path ../data/test_expert.csv \
    --subtitles_path ../data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name llovi-llama3-8b \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/ \
    --modality vision_language \
    --num_frames 256

python run_inference.py \
    --output_dir results \
    --data_path ../data/test_expert.csv \
    --subtitles_path ../data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name llovi-llama3-1b \
    --weights_dir /geovic/ghermi/weights/ \
    --modality vision_language \
    --num_frames 8

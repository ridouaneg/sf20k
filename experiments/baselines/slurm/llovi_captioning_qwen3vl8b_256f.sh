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
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k
source .venv/bin/activate

python scripts/run_llovi_captioning.py \
    --data_path data/test_expert.csv \
    --output_path data/captions/llovi_qwen3-vl-8b_1fps_256f.json \
    --captioner_name qwen3-vl-8b \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/ \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --fps 1.0 \
    --max_frames 256

python scripts/run_llovi_inference.py \
    --data_path data/test_expert.csv \
    --output_path data/captions/llovi_qwen3-vl-8b_1fps_256f.json \
    --captioner_name qwen3-vl-8b \
    --weights_dir /geovic/ghermi/weights/ \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --fps 1.0 \
    --max_frames 256

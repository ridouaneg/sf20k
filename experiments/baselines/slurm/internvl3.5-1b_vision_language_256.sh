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
source .lmms/bin/activate

python scripts/run_inference.py \
    --output_dir scripts/results/ \
    --data_path data/test_expert.csv \
    --subtitles_path data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name internvl3.5-1b \
    --weights_dir /lustre/fsn1/projects/rech/kcn/ucm72yx/weights/ \
    --modality vision_language \
    --num_frames 32 --force_rerun

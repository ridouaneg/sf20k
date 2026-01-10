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
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/experiments/instruction_tuning

# Train
accelerate launch \
    --num_processes 1 \
    --config_file default_config.yaml \
    train.py \
    --config configs/test_jz.yaml

# Run inference
python run_inference.py \
    --output_dir ./results/test/test/ \
    --data_path /users/ghermi/code/sf20k/data/test_expert.csv \
    --subtitles_path /users/ghermi/code/sf20k/data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name qwen3-vl-2b \
    --weights_dir /geovic/ghermi/weights/ \
    --modality vision_language \
    --num_frames 8 \
    --fps 1.0 \
    --n_subsample 8

python run_inference.py \
    --output_dir ./results/test/test/ \
    --data_path /users/ghermi/code/sf20k/data/test_expert.csv \
    --subtitles_path /users/ghermi/code/sf20k/data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name qwen3-vl-2b \
    --weights_dir /geovic/ghermi/weights/ \
    --adapter_path ./results/test/test/checkpoint-final/ \
    --modality vision_language \
    --num_frames 8 \
    --fps 1.0 \
    --n_subsample 8

# Run evaluation
python run_evaluation.py \
    --pred_path ./results/test/test/model_qwen3-vl-2b_modality_vision_language_num_frames_8.json

python run_evaluation.py \
    --pred_path ./results/test/test/model_qwen3-vl-2b_modality_vision_language_num_frames_8_adapter_path_checkpoint-final.json

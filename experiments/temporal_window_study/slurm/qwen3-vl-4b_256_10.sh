
#!/bin/bash
#SBATCH --job-name=sf20k
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=8:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.err

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/sf20k/bin/activate
cd /lustre/fswork/projects/rech/kcn/ucm72yx/code/sf20k/scripts/

python run_inference.py \
    --output_dir ./results/ijcv_rebuttal/ \
    --data_path ../../data/test_expert.csv \
    --subtitles_path ../../data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --weights_dir /lustre/fsn1/projects/rech/kcn/ucm72yx/weights/ \
    --modality vision_language \
    --model_name qwen3-vl-4b \
    --num_frames 256 \
    --num_segments 10 \
    --num_generations 10


import itertools

TEMPLATE = """#!/bin/bash
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
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/experiments/

python run_inference.py \\
    --output_dir ./results/ijcv_rebuttal/ \\
    --data_path ../data/test_expert.csv \\
    --subtitles_path ../data/test_subtitles.csv \\
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/ \\
    --model_name {model_name} \\
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/ \\
    --modality {modality} \\
    --num_frames {num_frames}
"""

def create_slurm(model_name, modality, num_frames):
    return TEMPLATE.format(model_name=model_name, modality=modality, num_frames=num_frames)

if __name__ == "__main__":
    # Model size ablation
    model_names = ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b"]
    modalities = ["vision_language"]
    num_frames = [256]

    all_combinations = list(itertools.product(model_names, modalities, num_frames))
    for model_name, modality, num_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, num_frames)
        with open(f"./slurm/{model_name}_{modality}_{num_frames}.sh", "w") as f:
            f.write(slurm_script)
        
    print(f"Created {len(all_combinations)} slurm scripts")

    # Num. frames ablation
    model_names = ["qwen3-vl-8b"]
    modalities = ["vision_language"]
    num_frames = [8, 16, 32, 64, 128, 256]

    all_combinations = list(itertools.product(model_names, modalities, num_frames))
    for model_name, modality, num_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, num_frames)
        with open(f"./slurm/{model_name}_{modality}_{num_frames}.sh", "w") as f:
            f.write(slurm_script)
        
    print(f"Created {len(all_combinations)} slurm scripts")

    # Modality ablation
    model_names = ["qwen3-vl-8b"]
    modalities = ["vision", "language", "vision_language"]
    num_frames = [256]

    all_combinations = list(itertools.product(model_names, modalities, num_frames))
    for model_name, modality, num_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, num_frames)
        with open(f"./slurm/{model_name}_{modality}_{num_frames}.sh", "w") as f:
            f.write(slurm_script)
        
    print(f"Created {len(all_combinations)} slurm scripts")
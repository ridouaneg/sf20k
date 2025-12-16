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
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/sf20k/bin/activate
cd /lustre/fswork/projects/rech/kcn/ucm72yx/code/sf20k/scripts/

python run_inference.py \\
    --output_dir ./results/ijcv_rebuttal/ \\
    --data_path ../data/test_expert.csv \\
    --subtitles_path ../data/test_subtitles.csv \\
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --model_name {model_name} \\
    --weights_dir {weights_dir} \\
    --modality {modality} \\
    --num_frames {num_frames}
"""

def create_slurm(model_name, modality, num_frames, weights_dir):
    return TEMPLATE.format(model_name=model_name, modality=modality, num_frames=num_frames, weights_dir=weights_dir)

if __name__ == "__main__":
    # Model size ablation
    model_names = ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b", "qwen3-vl-32b"]
    modalities = ["vision_language"]
    num_frames = [256]

    all_combinations = list(itertools.product(model_names, modalities, num_frames))
    for model_name, modality, num_frames in all_combinations:
        if model_name in ["qwen3-vl-2b", "qwen3-vl-32b"]:
            weights_dir = "/lustre/fsn1/projects/rech/kcn/ucm72yx/weights/"
        else:
            weights_dir = "/lustre/fsmisc/dataset/HuggingFace_Models/"

        slurm_script = create_slurm(model_name, modality, num_frames, weights_dir)

        with open(f"./slurm/{model_name}_{modality}_{num_frames}.sh", "w") as f:
            f.write(slurm_script)
        
    print(f"Created {len(all_combinations)} slurm scripts")
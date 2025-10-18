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
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/experiments/baselines/

python run_inference.py \
    --output_path {output_path} \
    --data_path ../../data/test_expert.csv \
    --subtitles_path ../../data/test_subtitles.csv \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/ \
    --weights_dir /lustre/fsmisc/dataset/HuggingFace_Models/Qwen/ \
    --model_name {model_name} \
    --modality {modality} \
    --fps 1.0 \
    --max_frames {max_frames}
"""

def create_slurm(model_name, modality, max_frames):
    output_path = f"results/{model_name}_{modality}_{max_frames}.json"
    return TEMPLATE.format(model_name=model_name, modality=modality, max_frames=max_frames, output_path=output_path)

if __name__ == "__main__":
    model_names = [
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "qwen2.5-vl-32b",
        "qwen2.5-vl-72b",
    ]
    modalities = ["vision_language"]
    max_frames = [512]

    all_combinations = list(itertools.product(model_names, modalities, max_frames))
    for model_name, modality, max_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, max_frames)
        with open(f"./slurm/{model_name}_{modality}_{max_frames}.sh", "w") as f:
            f.write(slurm_script)
        
    print(f"Created {len(all_combinations)} slurm scripts")

    model_names = ["qwen2.5-vl-7b"]
    modalities = ["vision_language"]
    max_frames = [8, 16, 32, 64, 128, 256, 512]

    all_combinations = list(itertools.product(model_names, modalities, max_frames))
    for model_name, modality, max_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, max_frames)
        with open(f"./slurm/{model_name}_{modality}_{max_frames}.sh", "w") as f:
            f.write(slurm_script)

    print(f"Created {len(all_combinations)} slurm scripts")

    model_names = ["qwen2.5-vl-7b"]
    modalities = ["vision", "language", "vision_language"]
    max_frames = [512]
    
    all_combinations = list(itertools.product(model_names, modalities, max_frames))
    for model_name, modality, max_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, max_frames)
        with open(f"./slurm/{model_name}_{modality}_{max_frames}.sh", "w") as f:
            f.write(slurm_script)
    
    print(f"Created {len(all_combinations)} slurm scripts")

    model_names = ["qwen2.5-omni-7b"]
    modalities = ["vision", "language", "vision_language", "audio_vision", "audio_vision_language"]
    max_frames = [512]
    
    all_combinations = list(itertools.product(model_names, modalities, max_frames))
    for model_name, modality, max_frames in all_combinations:
        slurm_script = create_slurm(model_name, modality, max_frames)
        with open(f"./slurm/{model_name}_{modality}_{max_frames}.sh", "w") as f:
            f.write(slurm_script)
    
    print(f"Created {len(all_combinations)} slurm scripts")
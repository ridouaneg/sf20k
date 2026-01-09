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
cd /lustre/fswork/projects/rech/kcn/ucm72yx/code/sf20k/experiments/temporal_window_study/

python run_inference.py \\
    --output_dir results \\
    --data_path ../../data/test_expert.csv \\
    --subtitles_path ../../data/test_subtitles.csv \\
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \\
    --model_name {model_name} \\
    --weights_dir {weights_dir} \\
    --modality vision_language \\
    --fps 1.0 \\
    --max_frames {n_frames} \\
    --n_generations {num_generations} \\
    --n_scenes {num_scenes}
"""

def create_slurm(model_name, weights_dir, num_generations, num_scenes, n_frames):
    return TEMPLATE.format(
        model_name=model_name,
        weights_dir=weights_dir,
        num_generations=num_generations,
        num_scenes=num_scenes,
        n_frames=n_frames,
    )

if __name__ == "__main__":
    # Model size ablation
    model_names = [
        "qwen3-vl-2b", 
        #"qwen3-vl-4b", 
        #"qwen3-vl-8b",
    ]
    ng_x_ns_x_nframes = [
        (100, 1, 256),
        (10, 10, 32),
        (1, 100, 4),
    ]

    all_combinations = list(itertools.product(model_names, ng_x_ns_x_nframes))
    for model_name, ng_x_ns_x_nframes in all_combinations:
        if model_name in ["qwen3-vl-2b", "qwen3-vl-32b"]:
            weights_dir = "/lustre/fsn1/projects/rech/kcn/ucm72yx/weights/"
        else:
            weights_dir = "/lustre/fsmisc/dataset/HuggingFace_Models/"
        num_generations, num_scenes, n_frames = ng_x_ns_x_nframes
        slurm_script = create_slurm(
            model_name=model_name,
            weights_dir=weights_dir,
            num_generations=num_generations,
            num_scenes=num_scenes,
            n_frames=n_frames
        )
        with open(f"./slurm/model_{model_name}_ng_{num_generations}_ns_{num_scenes}_nframes_{n_frames}.sh", "w") as f:
            f.write(slurm_script)
        
    print(f"Created {len(all_combinations)} slurm scripts")
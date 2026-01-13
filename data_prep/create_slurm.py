import os

# --- Configuration ---
NUM_CHUNKS = 20
JOB_NAME_BASE = "sf20k"
OUTPUT_DIR = "slurm"
LOG_DIR = "/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k"

# Script content template
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output={log_dir}/%j.out
#SBATCH --error={log_dir}/%j.err

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/sf20k/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/sf20k/data_prep

python generate_captions_vllm.py \\
    --input_path train_video_ids.json \\
    --output_path results/captions.parquet \\
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \\
    --shots_path shots.parquet \\
    --weights_dir /lustre/fsn1/projects/rech/kcn/ucm72yx/weights/ \\
    --model_id Qwen/Qwen3-VL-2B-Instruct \\
    --gpu_memory_utilization 0.9 \\
    --tensor_parallel_size 1 \\
    --save_interval 100 \\
    --num_chunks {num_chunks} \\
    --chunk_idx {chunk_idx}
"""

def main():
    # Create directory for scripts if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    submit_commands = []

    for i in range(NUM_CHUNKS):
        job_name = f"{JOB_NAME_BASE}_{i}"
        filename = os.path.join(OUTPUT_DIR, f"submit_chunk_{i}.sh")
        
        # Fill template
        script_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            log_dir=LOG_DIR,
            num_chunks=NUM_CHUNKS,
            chunk_idx=i
        )
        
        # Write to file
        with open(filename, "w") as f:
            f.write(script_content)
        
        submit_commands.append(f"sbatch {filename}")
        print(f"Generated: {filename}")

    # Create a master shell script to submit everything
    submit_all_path = os.path.join(OUTPUT_DIR, "submit_all.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n".join(submit_commands))
        f.write("\n")
    
    # Make submit script executable
    os.chmod(submit_all_path, 0o755)
    
    print("-" * 30)
    print(f"Done! To submit all jobs, run:\n{submit_all_path}")

if __name__ == "__main__":
    main()
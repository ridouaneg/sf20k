#!/bin/bash
#SBATCH --job-name=leak_no_title_qwen3-0.6b
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/sf20k/%j.err

module load arch/h100
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/sf20k/bin/activate

CODE_DIR=/lustre/fswork/projects/rech/kcn/ucm72yx/code/sf20k/experiments/data_leakage
WEIGHTS_DIR=/lustre/fsmisc/dataset/HuggingFace_Models
OUTPUT_DIR=$CODE_DIR/results
MODEL=qwen3-0.6b

cd $CODE_DIR

python run_inference.py --model_name $MODEL --weights_dir $WEIGHTS_DIR \
    --input_path data/movieqa.csv --output_dir $OUTPUT_DIR --no_title --n_subsample 256
python run_inference.py --model_name $MODEL --weights_dir $WEIGHTS_DIR \
    --input_path data/sf20k.csv   --output_dir $OUTPUT_DIR --no_title --n_subsample 256

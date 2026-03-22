#!/bin/bash
set -e
cd /users/ghermi/code/sf20k/experiments/data_leakage

OUTPUT=results
N_SUBSAMPLE=256
#MODELS="gpt-5-nano gpt-5-mini gpt-5"
MODELS="gpt-5-nano gpt-5-mini"
DATASETS="movieqa tvqa cinepile infinibench sf20k"
DATASETS_MCQA="movieqa tvqa cinepile"

for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        python run_inference.py --model_name $MODEL --output_dir $OUTPUT --input_path data/$DATASET.csv --n_subsample $N_SUBSAMPLE
        python run_inference.py --model_name $MODEL --output_dir $OUTPUT --input_path data/$DATASET.csv --n_subsample $N_SUBSAMPLE --no_title
    done
    for DATASET in $DATASETS_MCQA; do
        python run_inference.py --model_name $MODEL --output_dir $OUTPUT --input_path data/$DATASET.csv --n_subsample $N_SUBSAMPLE --mcqa
        python run_inference.py --model_name $MODEL --output_dir $OUTPUT --input_path data/$DATASET.csv --n_subsample $N_SUBSAMPLE --mcqa --no_title
    done
done

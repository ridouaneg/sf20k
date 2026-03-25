WEIGHTS_DIR=/geovic/ghermi/weights/
OUTPUT_DIR=results
#DATASETS="movieqa tvqa cinepile infinibench sf20k"
DATASETS=""
DATASETS_MCQA="movieqa tvqa cinepile"
#DATASETS_MCQA=""
MODELS="qwen3-0.6b qwen3-1.7b gemma-3-270m"
#MODELS="qwen3-0.6b qwen3-1.7b qwen3-14b gemma-3-270m"
N_SUBSAMPLE=256

for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --n_subsample $N_SUBSAMPLE
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --no_title \
            --n_subsample $N_SUBSAMPLE
    done
    for DATASET in $DATASETS_MCQA; do
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --mcqa \
            --n_subsample $N_SUBSAMPLE
        python run_inference.py \
            --model_name $MODEL \
            --weights_dir $WEIGHTS_DIR \
            --input_path data/$DATASET.csv \
            --output_dir $OUTPUT_DIR \
            --mcqa \
            --no_title \
            --n_subsample $N_SUBSAMPLE
    done
done

# Train
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --config_file default_config.yaml \
    train.py \
    --config configs/full.yaml

# Run inference
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --output_dir ./results/full/ \
    --data_path /users/ghermi/code/sf20k/data/test_expert.csv \
    --subtitles_path /users/ghermi/code/sf20k/data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name qwen3-vl-2b \
    --weights_dir /geovic/ghermi/weights/ \
    --modality vision_language \
    --num_frames 8 \
    --fps 1.0 \
    --n_subsample 8

CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --output_dir ./results/full/ \
    --data_path /users/ghermi/code/sf20k/data/test_expert.csv \
    --subtitles_path /users/ghermi/code/sf20k/data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --model_name qwen3-vl-2b \
    --weights_dir /geovic/ghermi/weights/ \
    --adapter_path ./results/full/checkpoint-final/ \
    --modality vision_language \
    --num_frames 8 \
    --fps 1.0 \
    --n_subsample 8

# Run evaluation
python run_evaluation.py \
    --pred_path ./results/full/model_qwen3-vl-2b_modality_vision_language_num_frames_8.json

python run_evaluation.py \
    --pred_path ./results/full/model_qwen3-vl-2b_modality_vision_language_num_frames_8_adapter_path_checkpoint-final.json

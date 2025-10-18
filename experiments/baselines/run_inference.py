import os
import argparse
from tqdm import tqdm
import json

from sf20k.datasets import SF20KDataset
from sf20k.models import get_model

cmd_line = """
python run_inference.py \
    --output_path results/tmp.json \
    --data_path ../../data/test_expert.csv \
    --subtitles_path ../../data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights/ \
    --model_name qwen2.5-vl-3b \
    --modality vision_language \
    --fps 1.0 \
    --max_frames 8 \
    --n_subsample 4 \
    --force_rerun \
    --print_prediction

python run_inference.py \
    --output_path results/tmp.json \
    --data_path ../../data/test_expert.csv \
    --subtitles_path ../../data/test_subtitles.csv \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights/ \
    --model_name qwen2.5-omni-3b \
    --modality audio_vision \
    --fps 1.0 \
    --max_frames 8 \
    --n_subsample 4 \
    --force_rerun \
    --print_prediction

What models I need?
- qwen2.5-vl with size/frame/modality ablations
- gpt-5 and gemini-2.5-pro for sota
- gpt-5-mini for metric evaluation
- qwen2.5-omni for subtitles vs. audio
- qwen3-vl for basic vs. reasoning
"""


def parse_args():
    parser = argparse.ArgumentParser()
    # Output config
    parser.add_argument("--output_path", type=str, default="submission.json")
    # Dataset config
    parser.add_argument("--data_path", type=str, default="../data/test.csv")
    parser.add_argument("--subtitles_path", type=str, default="../data/test_subtitles.csv")
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    # Model config
    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="qwen2.5-vl-3b", choices=[
        # VLM
        "qwen2.5-vl-7b",
        "qwen2.5-vl-32b",
        "qwen2.5-vl-72b",
        # AVLM
        "qwen2.5-omni-3b",
        "qwen2.5-omni-7b",
    ])
    parser.add_argument("--modality", type=str, default="vision", choices=["vision", "language", "vision_language"])
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=8)
    # Generation config
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    # Others
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--print_prediction", action="store_true")
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    # Prepare dataset
    dataset = SF20KDataset(
        modality=args.modality,
        data_path=args.data_path,
        subtitles_path=args.subtitles_path,
        video_dir=args.video_dir,
        n_subsample=args.n_subsample,
        seed=args.seed,
    )

    # Prepare model
    model = get_model(
        modality=args.modality,
        weights_dir=args.weights_dir,
        model_name=args.model_name,
        num_frames=args.num_frames,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    # Resume inference
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results_dict = json.load(open(args.output_path)) if os.path.exists(args.output_path) and not args.force_rerun else {}

    # Run inference
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # Check if the sample has already been processed
        if sample['question_id'] in results_dict and not args.force_rerun:
            continue

        # Get response
        prediction = model.generate(
            query=sample['query'],
            video_path=sample['video_path'],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            system_prompt=sample['system_prompt'],
        )

        # Print the prediction
        if args.print_prediction:
            print(sample['question'])
            print('-' * 100)
            print(prediction)
            print('-' * 100)
            print()

        # Store the prediction
        sample['prediction'] = prediction
        results_dict[sample['question_id']] = sample

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
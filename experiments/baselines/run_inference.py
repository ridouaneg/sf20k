import os
import argparse
from tqdm import tqdm
import json

from sf20k.datasets import SF20KDataset
from sf20k.models import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    # Output config
    parser.add_argument("--output_path", type=str, default="submission.json")
    # Dataset config
    parser.add_argument("--data_path", type=str, default="../data/test.csv")
    parser.add_argument("--subtitles_path", type=str, default="../data/test_subtitles.csv")
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    # Model config
    parser.add_argument("--weights_dir", type=str, default="")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", choices=[
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
    ])
    parser.add_argument("--num_frames", type=int, default=8)
    # Generation config
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main(args):
    # Prepare dataset
    dataset = SF20KDataset(
        data_path=args.data_path,
        subtitles_path=args.subtitles_path,
        video_dir=args.video_dir,
        n_subsample=args.n_subsample,
        seed=args.seed,
    )

    # Prepare model
    model = get_model(
        weights_dir=args.weights_dir,
        model_id=args.model_id,
        num_frames=args.num_frames,
    )

    # Resume inference
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results_dict = json.load(open(args.output_path)) if os.path.exists(args.output_path) else {}

    # Run inference
    for i, sample in tqdm(dataset, total=len(dataset)):
        # Check if the sample has already been processed
        if sample['question_id'] in results_dict:
            continue

        # Get response
        prediction = model.generate(
            video_path=sample['video_path'],
            query=sample['query'],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )

        # Store the prediction
        sample['prediction'] = prediction
        results_dict[sample['question_id']] = sample

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
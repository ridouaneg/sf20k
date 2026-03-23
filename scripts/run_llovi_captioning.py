"""
Stage 1 of the two-step LLoVi pipeline: caption all videos in the dataset and
save results to a JSON file.  The output can then be used by any model via:

    --model_name llovi-captions+<llm_name> --captions_path <output_path>

Example:

    python scripts/run_llovi_captioning.py \
        --data_path data/test_expert.csv \
        --video_dir /path/to/videos \
        --output_path data/captions/llovi_captions_qwen2.5-vl-3b_1fps_16f.json \
        --captioner_name qwen2.5-vl-3b \
        --weights_dir /path/to/weights \
        --fps 1.0 \
        --max_frames 16
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm
import json

from sf20k.models import LLoViCaptioner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--captioner_name", type=str, default="qwen2.5-vl-3b")
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=16)
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.data_path)

    # Build video_id -> path mapping (deduplicated)
    video_ids = df["video_id"].unique()
    video_paths = {
        vid: os.path.join(args.video_dir, f"{vid}.mkv")
        for vid in video_ids
    }
    print(f"Found {len(video_paths)} unique videos")

    captioner = LLoViCaptioner(args.captioner_name, weights_dir=args.weights_dir)

    output_path = args.output_path
    resume = True

    if isinstance(video_paths, list):
        video_paths = {p: p for p in video_paths}

    captions: dict = {}
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            captions = json.load(f)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    for video_id, video_path in tqdm(video_paths.items(), total=len(video_paths)):
        if video_id in captions:
            continue
        captions[video_id] = captioner.caption_video(video_path, fps=args.fps, max_frames=args.max_frames)
        with open(output_path, "w") as f:
            json.dump(captions, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Captions saved to {args.output_path}")


if __name__ == "__main__":
    main(parse_args())

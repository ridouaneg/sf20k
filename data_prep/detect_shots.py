import subprocess
import logging
import pandas as pd
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
import numpy as np
from scenedetect import detect, ContentDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="video_ids.json",
        help="Path to the json file with video ids",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="shots.parquet",
        help="Path to the output file",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Path to the video directory",
    )
    return parser.parse_args()


def main(args):
    video_ids = json.load(open(args.input_path))
    video_ids = list(set(video_ids))
    video_paths = {vid: os.path.join(args.video_dir, f"{vid}.mkv") for vid in video_ids}
    video_paths = {k: v for k, v in video_paths.items() if os.path.exists(v)}
    print(len(video_ids), len(video_paths))

    if os.path.exists(args.output_path):
        results = pd.read_parquet(args.output_path)
        processed_video_ids = results.video_id.unique()
        video_paths = {k: v for k, v in video_paths.items() if k not in processed_video_ids}
        print(len(results), len(processed_video_ids), len(video_paths))
    else:
        results = pd.DataFrame()

    all_results = []
    for i, (video_id, video_path) in tqdm(enumerate(video_paths.items()), total=len(video_paths)):
        try:
            scenes = detect(video_path, ContentDetector(threshold=27.0))
            
            for i, (start, end) in enumerate(scenes):
                duration = end - start                
                all_results.append({
                    'Scene Number': i + 1,
                    'Start Frame': start.get_frames(),
                    'Start Timecode': start.get_timecode(),
                    'Start Time (seconds)': start.get_seconds(),
                    'End Frame': end.get_frames(),
                    'End Timecode': end.get_timecode(),
                    'End Time (seconds)': end.get_seconds(),
                    'Length (frames)': duration.get_frames(),
                    'Length (timecode)': duration.get_timecode(),
                    'Length (seconds)': duration.get_seconds(),
                    'video_id': video_id,
                    'shot_id': f"{video_id}_{i+1}" # Example shot_id format
                })
        except Exception as e:
            print(f"Failed to process {video_id}: {e}")

        if i % 500:
            new_results_df = pd.DataFrame(all_results)
            final_df = pd.concat([results, new_results_df], ignore_index=True)
            final_df.to_parquet(args.output_path, index=False)
            
    new_results_df = pd.DataFrame(all_results)
    final_df = pd.concat([results, new_results_df], ignore_index=True)
    final_df.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

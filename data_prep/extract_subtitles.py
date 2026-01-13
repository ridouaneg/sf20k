import argparse
import os
import subprocess
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import whisper


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
        default="subtitles.parquet",
        help="Path to the output file",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Path to the video directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        choices=["tiny", "small", "medium", "base", "large", "large-v2", "large-v3"],
        help="Name of the model to use for extracting subtitles",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="translate",
        choices=["transcribe", "translate"],
        help="Diarize the audio before extracting subtitles",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Diarize the audio before extracting subtitles",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
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

    print(f"Loading Whisper model: {args.model} on {args.device}...")
    model = whisper.load_model(args.model, device=args.device)

    all_results = []
    for i, (video_id, video_path) in tqdm(enumerate(video_paths.items()), total=len(video_paths)):
        try:
            result = model.transcribe(
                video_path,
                task=args.task,
                #verbose=False,
            )
            language = result.get("language", "unknown")

            for i, seg in enumerate(result['segments']):
                row = {
                    "subtitle_nb": i + 1,
                    "seek": seg.get("seek"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "tokens": seg.get("tokens"),
                    "temperature": seg.get("temperature"),
                    "avg_logprob": seg.get("avg_logprob"),
                    "compression_ratio": seg.get("compression_ratio"),
                    "no_speech_prob": seg.get("no_speech_prob"),
                    "words": seg.get("words"), # Only present if word_timestamps=True
                    "video_id": video_id,
                    "language": language,
                    "library": "whisper",
                    "model": args.model,
                    "task": args.task,
                    "diarize": args.diarize,
                    "subtitle_id": f"{video_id}_{i+1}"
                }
                all_results.append(row)

        except Exception as e:
            print(f"Failed to process {video_id}: {e}")
        
        if i % 500:
            new_results_df = pd.DataFrame(all_results)
            new_results_df['tokens'] = new_results_df['tokens'].apply(json.dumps)
            new_results_df['words'] = new_results_df['words'].apply(json.dumps)
            final_df = pd.concat([results, new_results_df], ignore_index=True)
            final_df.to_parquet(args.output_path, index=False)
            
    new_results_df = pd.DataFrame(all_results)
    new_results_df['tokens'] = new_results_df['tokens'].apply(json.dumps)
    new_results_df['words'] = new_results_df['words'].apply(json.dumps)
    final_df = pd.concat([results, new_results_df], ignore_index=True)
    final_df.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

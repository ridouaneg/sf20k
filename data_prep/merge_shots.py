import logging
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='video_paths.csv', help="Path to the csv file with video paths")
    parser.add_argument("--input_dir", type=str, default='../data/shots', help="Path to the output directory")
    parser.add_argument("--output_path", type=str, default='../data/shots.csv', help="Path to the merged output file")
    return parser.parse_args()

def main(args):
    # Get parameters
    input_file = args.input_file
    input_dir = args.input_dir
    output_path = args.output_path

    # Get files
    files = list(Path(input_dir).glob('*/*-Scenes.csv'))
    video_paths = pd.read_csv(input_file).video_path.tolist()
    video_ids = [Path(video_path).stem for video_path in video_paths]
    files = [file for file in files if file.stem.replace('-Scenes', '') in video_ids]

    logging.info(f"Found {len(files)} files")

    # Merge individual files into a single CSV file
    results = []
    for file in tqdm(files, desc="Merging shot information", total=len(files)):
        video_id = file.parent.name
        df = pd.read_csv(file, skiprows=1)
        df['video_id'] = video_id
        df['shot_id'] = df.apply(lambda row: f"{row['video_id']}_{row['Scene Number']}", axis=1)
        results.append(df)
    
    results = pd.concat(results)
    results.to_csv(output_path, index=False)
    logging.info(f"Merged shot information saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

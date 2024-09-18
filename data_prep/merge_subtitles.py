import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="video_ids.csv",
        help="Path to the csv file with video ids",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../data/subtitles",
        help="Path to the directory with individual face CSV files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/subtitles.csv",
        help="Path to the output CSV file",
    )
    return parser.parse_args()


def main(args):
    # Get parameters
    input_file = args.input_file
    input_dir = args.input_dir
    output_path = args.output_path

    # Get files
    video_ids = pd.read_csv(input_file).video_id.tolist()

    files = list(Path(input_dir).glob("*.csv"))
    files = [file for file in files if file.stem in video_ids]

    logging.info(f"Found {len(files)} files")

    # Merge individual files into a single CSV file
    results = []
    for file in tqdm(files, desc="Merging subtitles", total=len(files)):
        try:
            data = json.load(open(file))
            df = pd.DataFrame(data["segments"])
            df["video_id"] = file.stem
            df = df[["video_id", "start", "end", "text", "no_speech_prob"]]
            df = df.drop_duplicates(subset="text", keep="first")
            df.sort_values(by=["video_id", "start", "end"], inplace=True)
            results.append(df)
        except Exception as e:
            logging.error(
                f"Error occurred while merging subtitles from {file}: {str(e)}"
            )
            continue

    if results:
        results = pd.concat(results)
        results.to_csv(output_path, index=False)
        logging.info(f"Merged subtitles saved to {output_path}")
    else:
        logging.info("No subtitles to merge.")

    logging.info(f"Merged faces saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

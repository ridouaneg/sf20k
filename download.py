import argparse
import subprocess
import os
import logging
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='./data/sfd.csv', help="Path to the csv file")
    parser.add_argument("--output_dir", type=str, default='./data/videos', help="Path to the output directory")
    parser.add_argument("--config_location", type=str, default='yt-dlp.conf', help="Path to the yt-dlp configuration file")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def download_video(video_url, output_path, config_location):
    try:
        cmd = [
            "yt-dlp",
            "--config",
            config_location,
            "-o",
            output_path,
            video_url,
        ]
        subprocess.run(cmd, check=True)
        logging.info(f"Downloaded: {video_url}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download {video_url}")

def main(args):
    # Load data
    df = pd.read_csv(args.input_file)
    video_ids = df['video_id'].unique()
    logging.info(f"{len(video_ids)} videos to download.")

    # Download videos
    os.makedirs(args.output_dir, exist_ok=True)
    for video_id in tqdm(video_ids, total=len(video_ids)):
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        output_path = os.path.join(args.output_dir, f"{video_id}.%(ext)s")
        if os.path.exists(os.path.join(args.output_dir, f"{video_id}.mkv")):
            logging.info(f"Video already downloaded: {video_id}")
            continue
        download_video(video_url, output_path, args.config_location)
    logging.info("All videos downloaded.")

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)

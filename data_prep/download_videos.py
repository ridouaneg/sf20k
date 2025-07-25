import argparse
import subprocess
import os
import logging
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="../data/test.csv", help="Path to the csv file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/videos/",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--config_location",
        type=str,
        default="yt-dlp.conf",
        help="Path to the yt-dlp configuration file",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Don't download videos",
    )
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
    df = pd.read_csv(args.input_file).sample(n=10)
    video_ids = df["video_id"].unique()
    logging.info(f"{len(video_ids)} videos to download.")

    # Download videos
    if not args.no_download:
        os.makedirs(args.output_dir, exist_ok=True)
        for video_id in tqdm(video_ids, total=len(video_ids)):
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            output_path = os.path.join(args.output_dir, f"{video_id}.%(ext)s")
            if os.path.exists(os.path.join(args.output_dir, f"{video_id}.mkv")):
                logging.info(f"Video already downloaded: {video_id}")
                continue
            download_video(video_url, output_path, args.config_location)

    # Check if all videos are downloaded
    downloaded_videos = os.listdir(args.output_dir)
    downloaded_video_ids = [x.split(".")[0] for x in downloaded_videos]
    missing_videos = set(video_ids) - set(downloaded_video_ids)

    logging.error(f"Failed to download {len(missing_videos)} videos.")
    logging.info(
        f"Percentage of videos missing: {len(missing_videos) / len(video_ids) * 100:.2f}%"
    )


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)

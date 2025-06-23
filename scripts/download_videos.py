import argparse
import os
import pandas as pd
import subprocess
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    return parser.parse_args()


def main(args):
    # Prepare data
    df = pd.read_csv(args.data_path)

    for i, sample in tqdm(df.iterrows(), total=len(df)):
        video_id = sample['video_id']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
        cmd_line = f"yt-dlp -f 'bestvideo[height<=360]+bestaudio/best[height<=360]' -i -o {video_path} --merge-output-format mp4 {video_url}"
        subprocess.run(cmd_line, shell=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)

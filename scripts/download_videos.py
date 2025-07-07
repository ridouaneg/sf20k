import argparse
import subprocess
import os
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_expert", choices=["train", "test", "test_silent", "test_expert"])
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--resolution", type=int, default=360, choices=[144, 240, 360, 480, 720, 1080], help="Resolution for video download (e.g., 360, 720, 1080)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading videos if they already exist in the target directory")
    return parser.parse_args()


def main(args):
    # Setup
    video_dir = os.path.join(args.video_dir, f"{args.resolution}p")
    os.makedirs(video_dir, exist_ok=True)
    df = load_dataset("rghermi/sf20k", split=args.split).to_pandas()
    df = df[['video_id', 'video_url']].drop_duplicates()
    total_videos = len(df)

    # Initialize counts
    downloaded_count = 0
    failed_count = 0
    skipped_count = 0
    failed_videos = {}

    # Download loop
    for _, row in tqdm(df.iterrows(), total=total_videos, desc="Downloading Videos"):
        video_id = row.video_id
        video_url = row.video_url
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # Skip if video already exists
        if os.path.exists(video_path) and args.skip_existing:
            skipped_count += 1
            continue

        # Construct yt-dlp command
        cmd_args = [
            "yt-dlp",
            "-f", f"bestvideo[height<={args.resolution}]+bestaudio/best[height<={args.resolution}]",
            "-o", video_path,
            "--merge-output-format", "mp4",
            #"--quiet",
            "--no-warnings",
            "--ignore-errors",
            video_url
        ]
        
        # Execute command and check for errors
        result = subprocess.run(cmd_args)

        if result.returncode == 0:
            downloaded_count += 1
        else:
            failed_count += 1
            failed_videos[video_id] = video_url

    # --- Final Logging ---
    print("\n" + "="*50)
    print("📊 DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Target Resolution:  {args.resolution}p")
    print(f"Total Videos in Split: {total_videos}")
    print("-" * 25)
    print(f"✅ Successfully Downloaded: {downloaded_count}")
    print(f"⏩ Skipped (already exist): {skipped_count}")
    print(f"❌ Failed to Download:     {failed_count}")
    print("-" * 25)
    if failed_videos:
        print("Failed video IDs and URLs:")
        for video_id, url in failed_videos.items():
            print(f"  - {video_id}: {url}")
    print("="*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)

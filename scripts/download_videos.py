import argparse
import subprocess
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_expert", choices=["train", "test", "test_silent", "test_expert"])
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--resolution", type=int, default=360, choices=[144, 240, 360, 480, 720, 1080], help="Resolution for video download (e.g., 360, 720, 1080)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading videos if they already exist in the target directory")
    parser.add_argument("--silence_errors", action="store_true", help="Silence errors")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for downloading videos")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for downloading videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to download (for debug runs). If None, downloads all videos.")
    parser.add_argument("--failed-videos-file", type=str, default=None, help="Path to CSV file containing failed video IDs to retry. If provided, only downloads videos listed in this file. CSV should have columns: video_id, video_url (optional), error_reason (optional)")
    parser.add_argument("--cookies", type=str, default=None, help="Path to a .txt file containing cookies. Ignored if not passed")
    return parser.parse_args()


def download_video(row, video_dir, args):
    """Download a single video. Returns tuple (video_id, status, video_url, error_reason) where status is 'success', 'failed', or 'skipped'."""
    video_id = row.video_id
    video_url = row.video_url
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    # Skip if video already exists
    if os.path.exists(video_path) and args.skip_existing:
        return (video_id, 'skipped', video_url, None)

    # Construct yt-dlp command
    cmd_args = [
        "yt-dlp",
        "-S", "vcodec:h264,res,acodec:m4a",  # Quicktime compatible; h264 decodes faster as well
        "-f", f"bestvideo[height<={args.resolution}]+bestaudio/best[height<={args.resolution}]",
        "-o", video_path,
        #"--quiet",
        "--concurrent-fragments", str(args.threads),
        video_url
    ]
    if args.silence_errors:
        cmd_args.extend([
            "--no-warnings", "--ignore-errors"
        ])
    if args.cookies:
        cmd_args.extend([
            "--cookies", args.cookies
        ])

    # Execute command and check for errors
    result = subprocess.run(cmd_args, capture_output=True, text=True)

    if result.returncode == 0:
        status = 'success'
        error_reason = None
    else:
        status = 'failed'
        error_reason = _extract_error_reason(result)

    return (video_id, status, video_url, error_reason)


def _extract_error_reason(result) -> str | None:
    # Successful download, no error reason
    if result.returncode == 0:
        return None

    # Extract error reason from stderr
    error_reason = None
    if result.stderr:
        # Try to extract a meaningful error message
        stderr_lines = result.stderr.strip().split('\n')
        # Look for common error patterns
        for line in reversed(stderr_lines):  # Check from bottom up (most recent errors)
            line_lower = line.lower()
            if 'error' in line_lower or 'unavailable' in line_lower or 'private' in line_lower or 'deleted' in line_lower:
                error_reason = line.strip()
                break
        # If no specific error found, use last line or first meaningful line
        if not error_reason:
            for line in reversed(stderr_lines):
                if line.strip() and not line.strip().startswith('['):
                    error_reason = line.strip()[:200]  # Limit length
                    break
    if not error_reason:
        error_reason = "Unknown error"

    return error_reason

def load_failed_videos_from_file(filepath):
    """Load failed video IDs from a CSV file. Returns a set of video_ids."""
    df = pd.read_csv(filepath)
    if 'video_id' not in df.columns:
        raise ValueError(f"CSV file must contain 'video_id' column. Found columns: {df.columns.tolist()}")
    return set(df['video_id'].dropna())


def save_failed_videos_to_file(failed_videos: dict, filepath):
    """Save failed videos to a CSV file with video_id, video_url, and error_reason."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    df = pd.DataFrame([
        {
            'video_id': video_id,
            'video_url': video_url,
            'error_reason': error_reason
        }
        for video_id, (video_url, error_reason) in failed_videos.items()
    ])
    df.to_csv(filepath, index=False)


def main(args):
    start_time = time.time()
    
    # Setup
    video_dir = os.path.join(args.video_dir, f"{args.resolution}p")
    os.makedirs(video_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset("rghermi/sf20k", split=args.split).to_pandas()
    df = df[['video_id', 'video_url']].drop_duplicates()
    
    # If failed-videos-file is provided, filter to only those videos
    if args.failed_videos_file:
        failed_video_ids = load_failed_videos_from_file(args.failed_videos_file)
        df = df[df['video_id'].isin(failed_video_ids)]
        print(f"📋 Loading {len(df)} videos from failed videos file: {args.failed_videos_file}")
    
    if args.max_videos is not None:
        df = df.head(args.max_videos)
    total_videos = len(df)

    # Initialize counts
    downloaded_count = 0
    failed_count = 0
    skipped_count = 0
    failed_videos = {}  # {video_id: (video_url, error_reason)}

    # Parallel download using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all download tasks
        future_to_row = {
            executor.submit(download_video, row, video_dir, args): row
            for _, row in df.iterrows()
        }

        # Process completed downloads with progress bar
        with tqdm(total=total_videos, desc="Downloading Videos") as pbar:
            for future in as_completed(future_to_row):
                video_id, status, video_url, error_reason = future.result()
                
                if status == 'success':
                    downloaded_count += 1
                elif status == 'failed':
                    failed_count += 1
                    failed_videos[video_id] = (video_url, error_reason)
                elif status == 'skipped':
                    skipped_count += 1
                
                pbar.update(1)

    # Save failed videos to file
    if failed_videos:
        # If retrying from a file, save to a new file with timestamp or different name
        if args.failed_videos_file:
            base_name = os.path.splitext(args.failed_videos_file)[0]
            failed_file_path = f"{base_name}_retry_failed.csv"
        else:
            failed_file_path = f"failed_videos_{args.split}_{args.resolution}p.csv"
        save_failed_videos_to_file(failed_videos, failed_file_path)
        print(f"\n💾 Saved {len(failed_videos)} failed videos to: {failed_file_path}")

    # Calculate total runtime
    end_time = time.time()
    duration_seconds = end_time - start_time
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    
    # Format duration string
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"

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
    print(f"⏱️  Total Runtime:          {duration_str}")
    print("="*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)

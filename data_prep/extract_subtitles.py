import argparse
import os
import subprocess
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='video_paths.csv', help="Path to the csv file with video paths")
    parser.add_argument("--output_dir", type=str, default='../data/subtitles/', help="Path to the output directory")
    parser.add_argument("--library", type=str, default="whisperx", choices=["whisper", "whisperx"], help="Library to use for extracting subtitles")
    parser.add_argument("--model", type=str, default="large-v3", choices=["tiny", "small", "medium", "base", "large", "large-v2", "large-v3"], help="Name of the model to use for extracting subtitles")
    parser.add_argument("--task", type=str, default="translate", choices=["transcribe", "translate"], help="Diarize the audio before extracting subtitles")
    parser.add_argument("--diarize", action="store_true", help="Diarize the audio before extracting subtitles")
    parser.add_argument("--access_token", type=str, default=None, help="HuggingFace token to access pyannote")
    return parser.parse_args()

def setup_logging():
    """
    Set up logging configuration to output informational messages.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def save_config(config, output_dir):
    """
    Save the configuration settings to a JSON file in the output directory.

    :param config: Dictionary containing configuration settings.
    :param output_dir: Path to the output directory.
    """
    config_path = os.path.join(output_dir, "config.json")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(config_path):
        logging.warning(f"Config file already exists at {config_path}")
        config = json.load(open(config_path))
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    return config

def extract_subtitles_whisper(video_path, output_dir, model, task):
    """
    Extracts subtitles from a video using the Whisper CLI.
    
    :param video_path: Path to the video file.
    :param output_dir: Directory to save the extracted subtitles.
    :param model: Whisper model to use.
    :param task: Task type (transcribe or translate).
    """
    try:
        cmd = [
            "whisper",
            video_path,
            "--model",
            model,
            "--output_dir",
            output_dir,
            "--output_format",
            "all",
            "--task",
            task,
            "--verbose",
            "False",
        ]
        subprocess.run(cmd, check=True)
        logging.info(f"Processed: {video_path}")
    except Exception as e:
        logging.error(f"Error occurred while extracting subtitles: {str(e)}")

def extract_subtitles_whisperx(video_path, output_dir, model, task, diarize=False, access_token=None):
    """
    Extracts subtitles from a video using the WhisperX CLI with optional diarization.

    :param video_path: Path to the video file.
    :param output_dir: Directory to save the extracted subtitles.
    :param model: WhisperX model to use.
    :param task: Task type (transcribe or translate).
    :param diarize: Boolean flag to enable diarization.
    """
    try:
        cmd = [
            "whisperx",
            video_path,
            "--model",
            model,
            "--output_dir",
            output_dir,
            "--output_format",
            "all",
            "--task",
            task,
            "--verbose",
            "False",
        ]
        if diarize:
            cmd.extend([
                "--diarize",
                "--hf_token",
                access_token,
            ])
        subprocess.run(cmd, check=True)
        logging.info(f"Processed: {video_path}")
    except Exception as e:
        logging.error(f"Error occurred while extracting subtitles: {str(e)}")

def main(args):
    # Get parameters
    input_file = args.input_file
    output_dir = args.output_dir
    library = args.library
    model = args.model
    task = args.task
    diarize = args.diarize
    access_token = args.access_token
    n_subsample = args.n_subsample

    # Set up logging
    setup_logging()

    # Save config
    config = {
        #"input_file": input_file,
        #"output_dir": output_dir,
        "library": library,
        "model": model,
        "task": task,
        "diarize": diarize,
    }
    config = save_config(config, output_dir)

    logging.info(f"Config: {config}")

    # Get video paths
    video_paths = pd.read_csv(input_file).video_path.tolist()
    if n_subsample > 0:
        video_paths = np.random.choice(video_paths, n_subsample, replace=False)
    
    logging.info(f"Found {len(video_paths)} video paths")
    
    for _, video_path in tqdm(enumerate(video_paths), total=len(video_paths), desc="Processing videos"):
        video_id = Path(video_path).stem
        output_path = f"{output_dir}/{video_id}.srt"
        if Path(output_path).exists():
            #logging.info(f"Faces already extracted for {video_id}")
            continue
        
        if library == "whisper":
            extract_subtitles_whisper(video_path, f"{output_dir}", model, task)
        elif library == "whisperx":
            extract_subtitles_whisperx(video_path, f"{output_dir}", model, task, diarize, access_token)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

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
    parser.add_argument("--input_file", type=str, default='video_paths.csv', help="Path to the csv file with video paths")
    parser.add_argument("--output_dir", type=str, default='../data/subtitles/', help="Path to the output directory")
    parser.add_argument("--model", type=str, default="large-v3", choices=["tiny", "small", "medium", "base", "large", "large-v2", "large-v3"], help="Name of the model to use for extracting subtitles")
    parser.add_argument("--task", type=str, default="translate", choices=["transcribe", "translate"], help="Diarize the audio before extracting subtitles")
    parser.add_argument("--n_subsample", type=int, default=-1, help="Number of videos to subsample")
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

def main(args):
    # Get parameters
    input_file = args.input_file
    output_dir = args.output_dir
    model = args.model
    task = args.task
    n_subsample = args.n_subsample

    # Set up logging
    setup_logging()

    # Save config
    config = {
        "model": model,
        "task": task,
    }
    config = save_config(config, output_dir)

    logging.info(f"Config: {config}")

    # Get video paths
    video_paths = pd.read_csv(input_file).video_path.tolist()
    if n_subsample > 0:
        video_paths = np.random.choice(video_paths, n_subsample, replace=False)
    
    logging.info(f"Found {len(video_paths)} video paths")

    model = whisper.load_model(model)

    for video_path in tqdm(video_paths, total=len(video_paths), desc="Extracting subtitles"):
        video_id = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_id}.csv")
        if Path(output_path).exists():
            #logging.info(f"Subtitles already extracted for {video_id}")
            continue
            
        result = model.transcribe(video_path)

        df = pd.DataFrame(result['segments'])
        df = df[['start', 'end', 'text', 'no_speech_prob']]
        #df = df.drop_duplicates(subset='text', keep='first')
        df.to_csv(os.path.join(output_dir, '.csv'), index=False)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

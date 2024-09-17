import subprocess
import logging
import pandas as pd
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='video_paths.csv', help="Path to the csv file with video paths")
    parser.add_argument("--output_dir", type=str, default='../data/shots/', help="Path to the output directory")
    parser.add_argument("--save_images", action="store_true", help="Save images for each shot")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to save for each shot (if --save_images)")
    parser.add_argument("--n_subsample", type=int, default=-1, help="Number of videos to subsample")
    return parser.parse_args()

def setup_logging():
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

def find_scenes(input_path, output_path, save_images=False, num_images=1):
    try:
        cmd = [
            "scenedetect",
            "--input",
            input_path,
            "--output",
            output_path,
            "--quiet",
            "detect-adaptive",
            "list-scenes",
        ]
        if save_images:
            cmd.extend(
                [
                    "save-images",
                    "--num-images",
                    f"{num_images}",
                ]
            )
        subprocess.run(cmd)
    except Exception as e:
        logging.error(f"Error occurred while finding scenes: {str(e)}")
        return

def main(args):
    # Get parameters
    input_file = args.input_file
    output_dir = args.output_dir
    save_images = args.save_images
    num_images = args.num_images
    n_subsample = args.n_subsample

    # Setup logging
    setup_logging()

    # Save config
    config = {
        "input_file": input_file,
        "output_dir": output_dir,
        "save_images": save_images,
        "num_images": num_images,
    }
    config = save_config(config, output_dir)

    logging.info(f"Config: {config}")

    # Get video paths
    video_paths = pd.read_csv(input_file).video_path.tolist()
    if n_subsample > 0:
        video_paths = np.random.choice(video_paths, n_subsample, replace=False)
    
    logging.info(f"Found {len(video_paths)} video paths")
    
    for _, video_path in tqdm(video_paths.iterrows(), total=video_paths.shape[0]):
        video_id = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_id}")
        os.makedirs(output_path, exist_ok=True)

        # Check if the scenes file already exists
        scenes_file = os.path.join(output_path, f"{video_id}-Scenes.csv")
        if os.path.exists(scenes_file):
            logging.info(f"Shots already detected for {video_path}")
            continue

        find_scenes(video_path, output_path, save_images, num_images)

if __name__ == "__main__"
    args = parse_args()
    main(args)

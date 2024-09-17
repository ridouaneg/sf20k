import argparse
import logging
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='video_paths.csv', help="Path to the csv file with video paths")
    parser.add_argument("--output_dir", type=str, default='../data/faces', help="Path to the output directory")
    parser.add_argument("--fps", type=float, default=3.0, help="Frames per second to extract from the video")
    parser.add_argument("--detector_backend", type=str, default="retinaface", choices=["retinaface", "ssd", "opencv", "mtcnn", "dlib"], help="Name of the face detector to use")
    #parser.add_argument("--model_name", type=str, default="VGG-Face", choices=["ArcFace", "VGG-Face", "OpenFace", "Facenet", "DeepFace", "DeepID", "Dlib"], help="Name of the face recognition model to use")
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

def read_video(video_path, fps=1.0):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(original_fps / fps)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
        
        if frame_count % frame_skip_interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            frames.append(frame)
        
        frame_count += 1

    cap.release()
    return frames

def main(args):
    input_file = args.input_file
    output_dir = args.output_dir
    fps = args.fps
    detector_backend = args.detector_backend
    #model_name = args.model_name
    n_subsample = args.n_subsample

    # Set up logging
    setup_logging()
    
    # Save config
    config = {
        #"input_file": input_file,
        #"output_dir": output_dir,
        "fps": fps,
        "detector_backend": detector_backend,
        #"model_name": model_name,
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
        output_path = f"{output_dir}/{video_id}.csv"
        if Path(output_path).exists():
            #logging.info(f"Faces already extracted for {video_id}")
            continue

        frames = read_video(video_path, fps=fps)

        results = []
        for frame_number, frame in tqdm(enumerate(frames), total=len(frames), desc="Extracting faces", leave=False):
            frame_id = f"{video_id}_{frame_number:04d}"

            objs = DeepFace.extract_faces(frame, detector_backend=detector_backend, enforce_detection=False)
            #objs = DeepFace.represent(frame, model_name=model_name, enforce_detection=False, detector_backend=detector_backend)
            
            for face_number, obj in enumerate(objs):
                face_id = f"{frame_id}_{face_number:02d}"
                
                facial_area = obj['facial_area']
                face_confidence = obj['confidence']

                #embedding = obj['embedding']
                #facial_area = obj['facial_area']
                #face_confidence = obj['face_confidence']

                #if face_confidence > 0.0:
                results.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "face_id": face_id,
                    "frame_number": frame_number,
                    "face_number": face_number,
                    #"embedding": embedding,
                    "facial_area": facial_area,
                    "face_confidence": face_confidence,
                })
    
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)

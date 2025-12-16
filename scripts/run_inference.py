import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from sf20k.models import get_model
from sf20k.datasets.sf20k import SF20KDataset
from sf20k.prompts import OEQAPrompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines on SF20K dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--subtitles_path", type=str, required=True, help="Path to the subtitles CSV file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory containing model weights")
    parser.add_argument("--modality", type=str, default="vision_language", choices=["vision", "language", "vision_language"], help="Modality to use")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to sample")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for sampling")
    parser.add_argument("--n_subsample", type=int, default=-1, help="Number of samples to run (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerunning all samples")
    return parser.parse_args()


def main(args):
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"model_{args.model_name}_modality_{args.modality}_num_frames_{args.num_frames}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Results will be saved to {output_path}")

    # Initialize prompt
    prompt = OEQAPrompt(modality=args.modality)

    # Initialize dataset
    dataset = SF20KDataset(
        prompt=prompt,
        data_path=args.data_path,
        video_dir=args.video_dir,
        subtitles_path=args.subtitles_path,
        n_subsample=args.n_subsample,
        seed=args.seed
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    # Initialize model
    print(f"Loading model {args.model_name}...")
    model = get_model(
        model_name=args.model_name,
        weights_dir=args.weights_dir,
        modality=args.modality,
        fps=args.fps,
        max_frames=args.num_frames,
        load_in_4bit=args.load_in_4bit
    )
    print("Model loaded successfully")

    # Generation loop
    results = {}
    # Check if output file exists and load existing results to resume
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results")
    
    existing_ids = set(results.keys())

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        question_id = sample["question_id"]
        
        if question_id in existing_ids and not args.force_rerun:
            continue

        try:
            response = model.generate(
                query=sample["query"],
                video_path=sample["video_path"],
                system_prompt=None,
            )
            
            prediction = prompt.postprocess_response(response)

            results[question_id] = {
                "question_id": question_id,
                "video_id": sample["video_id"],
                "question": sample["question"],
                "answer": sample["answer"], # Ground truth
                "response": response,
                "prediction": prediction,
                "model": args.model_name,
                "modality": args.modality,
                "num_frames": args.num_frames,
            }
                            
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            
        except Exception as e:
            print(f"Error processing sample {question_id}: {e}")
            continue

    print(f"Saved results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
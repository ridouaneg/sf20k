import argparse
import os
import json
import pandas as pd
from tqdm import tqdm

from sf20k.models.llovi import LLoViModel
from sf20k.datasets.sf20k import SF20KDataset
from sf20k.prompts import OEQAPrompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLoVi stage-2 inference on SF20K dataset")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--subtitles_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, help="LLoVi LLM backend, e.g. llovi-llama3-8b")
    parser.add_argument("--captions_path", type=str, required=True, help="Path to pre-computed captions JSON")
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rerun", action="store_true")
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    captions_tag = os.path.splitext(os.path.basename(args.captions_path))[0]
    output_filename = f"model_{args.model_name}_captions_{captions_tag}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Results will be saved to {output_path}")

    prompt = OEQAPrompt(modality="language")

    dataset = SF20KDataset(
        prompt=prompt,
        data_path=args.data_path,
        video_dir=args.video_dir,
        subtitles_path=args.subtitles_path,
        n_subsample=args.n_subsample,
        seed=args.seed,
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    print(f"Loading model {args.model_name}...")
    model = LLoViModel(
        model_name=args.model_name,
        captioner_name=None,
        captions_path=args.captions_path,
        weights_dir=args.weights_dir,
    )
    print("Model loaded successfully")

    results = {}
    if os.path.exists(output_path) and not args.force_rerun:
        with open(output_path) as f:
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
            )

            prediction = prompt.postprocess_response(response)

            results[question_id] = {
                "question_id": question_id,
                "video_id": sample["video_id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "response": response,
                "prediction": prediction,
                "model": args.model_name,
                "captions_path": args.captions_path,
            }

            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Error processing sample {question_id}: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main(parse_args())

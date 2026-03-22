"""
Smoke-tests for LLaVA-Video, LVAgent, and LLoVi on a single dataset sample.

Usage:
    python tests/test_llava_lvagent_llovi.py
"""

import argparse
import os

from sf20k.prompts import OEQAPrompt
from sf20k.datasets import SF20KDataset
from sf20k.models import get_model


_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)

DATA_PATH       = os.path.join(_ROOT, "data", "test_expert.csv")
VIDEO_DIR       = "/geovic/geovic/SF20K/videos/"
SUBTITLES_PATH  = os.path.join(_ROOT, "data", "test_subtitles.csv")
WEIGHTS_DIR     = "/geovic/ghermi/weights"
SAMPLE_IDX      = 42
FPS             = 1.0
MAX_FRAMES      = 8


def get_dataset(prompt):
    return SF20KDataset(
        prompt=prompt,
        data_path=DATA_PATH,
        video_dir=VIDEO_DIR,
        subtitles_path=SUBTITLES_PATH,
    )


def run(model_name, model_kwargs, dataset, prompt):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print('='*60)

    model = get_model(model_name=model_name, **model_kwargs)

    sample = dataset[SAMPLE_IDX]
    query = sample["query"]
    video_path = sample["video_path"]

    print(f"Query      : {query}")
    print(f"Video path : {video_path}")

    response = model.generate(query=query, video_path=video_path)
    prediction = prompt.postprocess_response(response)

    print(f"Response   : {response}")
    print(f"Prediction : {prediction}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava-video", "lvagent", "llovi", "all"],
                        default="all", help="Which model(s) to test")
    args = parser.parse_args()

    prompt = OEQAPrompt()
    dataset = get_dataset(prompt)

    tests = {
        "llava-video": (
            "llava-video-7b",
            dict(weights_dir=WEIGHTS_DIR, modality="vision_language",
                 fps=FPS, max_frames=MAX_FRAMES),
        ),
        "lvagent": (
            "lvagent-qwen2.5-7b",
            dict(weights_dir=WEIGHTS_DIR, fps=FPS, max_frames=MAX_FRAMES),
        ),
        "llovi": (
            # Uses Qwen2.5-VL-3B captioner (stage 1) + LLaMA-3.1-8B LLM (stage 2).
            # Switch to "llovi-gpt4o-mini" if OPENAI_API_KEY is set and local weights
            # are unavailable.
            "llovi-llama3-8b",
            dict(weights_dir=WEIGHTS_DIR, captioner_name="qwen2.5-vl-3b",
                 captioner_fps=FPS, captioner_max_frames=MAX_FRAMES),
        ),
    }

    selected = list(tests.keys()) if args.model == "all" else [args.model]

    for key in selected:
        model_name, model_kwargs = tests[key]
        run(model_name, model_kwargs, dataset, prompt)

    print("\nAll selected models tested successfully.")


if __name__ == "__main__":
    main()

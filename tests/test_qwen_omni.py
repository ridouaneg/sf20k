import unittest
import os
import sys
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from sf20k.models import get_model
from sf20k.models import QwenOmniModel
from sf20k.prompts import OEQAPrompt
from sf20k.constants import WEIGHTS_DIR
from sf20k.datasets import SF20KDataset

from qwen_vl_utils import process_vision_info


def main():
    # Prepare dataset
    prompt = OEQAPrompt()
    dataset = SF20KDataset(
        prompt=prompt,
        data_path="../data/test_expert.csv",
        video_dir="/geovic/geovic/SF20K/videos/",
        subtitles_path="../data/test_subtitles.csv",
    )

    # Prepare model
    model_name = "qwen2.5-omni-3b"

    fps = 1.0
    max_frames = 8
    
    model = QwenOmniModel(
        model_name=model_name,
        modality="audio_vision_language",
        fps=fps,
        max_frames=max_frames,
    )

    # Generate response
    idx = 42
    sample = dataset[idx]

    query = sample["query"]
    video_path = sample["video_path"]
    max_new_tokens = 256
    do_sample = True
    temperature = 1.0

    total_pixels=20480 * 32 * 32
    min_pixels=64 * 32 * 32
    
    response = model.generate(
        query=query,
        video_path=video_path,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        total_pixels=total_pixels,
        min_pixels=min_pixels,
    )
    
    prediction = prompt.postprocess_response(response)

    print(query)
    print(video_path)
    print(response)
    print(prediction)


if __name__ == "__main__":
    main()
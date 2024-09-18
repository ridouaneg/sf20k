from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX

from decord import VideoReader, cpu
import torch
import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test LongVA")
    parser.add_argument("--model_name", type=str, default="lmms-lab/LongVA-7B-DPO")
    parser.add_argument(
        "--video_path", type=str, default="../../../data/media/test.mkv"
    )
    parser.add_argument("--query", type=str, default="Caption this video.")
    parser.add_argument("--num_frames", type=int, default=8)
    return parser.parse_args()


def read_video(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames


def main(args):
    model_name = args.model_name
    video_path = args.video_path
    query = args.query
    num_frames = args.num_frames

    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.5,
        "top_p": None,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": 1024,
    }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_name, None, "llava_qwen", device_map="auto", load_4bit=True
    )
    frames = read_video(video_path, num_frames)
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")[
        "pixel_values"
    ].to(model.device, dtype=torch.float16)
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    with torch.inference_mode():
        outputs = model.generate(
            input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(response)


if __name__ == "__main__":
    args = parse_args()
    main(args)

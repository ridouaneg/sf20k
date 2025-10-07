import os
import argparse
from tqdm import tqdm
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import numpy as np
import cv2
from PIL import Image

# Ensure you have 'pip install qwen-vl-utils'
from qwen_vl_utils import process_vision_info


STAGE1_PROMPT = (
    "Please describe the movie clip in the following four steps: "
    "1. Identify main characters; "
    "2. Describe the actions of characters in one sentence, i.e., who is doing what, focusing on the movements; "
    "3. Describe the interactions between characters in one sentence, such as looking; "
    "4. Describe the facial expressions of characters in one sentence. "
    "Make sure you do not hallucinate information. "
    "###ANSWER TEMPLATE###: 1. Main characters: ''; 2. Actions: ''; 3. Character-character interactions: ''; 4. Facial expressions: ''."
)


cmd_lines = """

CUDA_VISIBLE_DEVICES=1 python run_stage1_vllm.py \
    --output_path ./results/stage1_qwen2vl-3b_vllm.json \
    --data_path /users/ghermi/code/sf20k/data/train.csv \
    --shots_path /users/ghermi/code/sf20k/data/shots.parquet \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --num_frames 8 \
    --print_prediction \
    --force_rerun \
    --n_subsample -1 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --batch_size 256

python run_stage1_vllm.py \
    --output_path ./results/stage1_qwen2vl-7b_vllm-0-1024.json \
    --data_path /users/ghermi/code/sf20k/data/train.csv \
    --shots_path /users/ghermi/code/sf20k/data/shots.parquet \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --num_frames 8 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --batch_size 256 \
    --start_idx 0 \
    --end_idx 1024

"""


def parse_args():
    parser = argparse.ArgumentParser()
    # Output config
    parser.add_argument("--output_path", type=str, default="stage1.json")
    # Dataset config
    parser.add_argument("--data_path", type=str, default="../data/test.csv")
    parser.add_argument("--shots_path", type=str, default="../data/test_subtitles.csv")
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    # Model config
    parser.add_argument("--weights_dir", type=str, default="")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", choices=[
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ])
    parser.add_argument("--num_frames", type=int, default=8)
    # Generation config
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0)
    # vLLM config
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    # Others
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples to process in a single batch.")
    parser.add_argument("--force_rerun", action='store_true')
    parser.add_argument("--print_prediction", action='store_true')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SF20KDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        shots_path: str,
        video_dir: str,
        n_subsample: int = -1,
        seed: int = 42,
        start_idx: int = 0,
        end_idx: int = -1,
    ):
        df = pd.read_csv(data_path)
        video_ids = df['video_id'].unique()

        df_shots = pd.read_parquet(shots_path) if shots_path.endswith('.parquet') else pd.read_csv(shots_path)
        df_shots.rename(columns={
            'video_id': 'video_id',
            'shot_id': 'shot_id',
            'Start Time (seconds)': 'start_time',
            'End Time (seconds)': 'end_time',
        }, inplace=True)
        df_shots = df_shots[['shot_id', 'video_id', 'start_time', 'end_time']]
        df_shots = df_shots[df_shots['video_id'].isin(video_ids)]
        df_shots['video_path'] = df_shots['video_id'].apply(lambda x: os.path.join(video_dir, f"{x}.mkv"))

        if n_subsample > -1:
            df_shots = df_shots.sample(n=n_subsample, random_state=seed)
        if end_idx > -1:
            df_shots = df_shots.iloc[start_idx:end_idx]
        df_shots = df_shots[df_shots['video_path'].apply(lambda x: os.path.exists(x))]
        all_clips = df_shots.to_dict(orient='records')
        
        print(f"Loaded {len(all_clips)} clips")
        self.all_clips = all_clips

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        sample = self.all_clips[idx]
        return {
            'shot_id': sample['shot_id'],
            'video_id': sample['video_id'],
            'query': STAGE1_PROMPT,
            'video_path': sample['video_path'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
        }


def format_chat_messages(
    query: str,
    video_path: str,
    num_frames: int,
    start_time: float,
    end_time: float,
):
    """Helper function to format the messages payload for the processor."""
    content = [
        {"type": "video", "video": video_path, "nframes": num_frames, "video_start": start_time, "video_end": end_time},
        {"type": "text", "text": query},
    ]
    return [{"role": "user", "content": content}]


def main(args):
    # Prepare dataset
    dataset = SF20KDataset(
        data_path=args.data_path,
        shots_path=args.shots_path,
        video_dir=args.video_dir,
        n_subsample=args.n_subsample,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    # Prepare model
    model_path = os.path.join(args.weights_dir, args.model_id)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
    )

    # Resume inference
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results_dict = json.load(open(args.output_path, 'r')) if os.path.exists(args.output_path) and not args.force_rerun else {}

    # Filter out samples that have already been processed
    samples_to_process = []
    for sample in dataset:
        if sample['shot_id'] not in results_dict or args.force_rerun:
            samples_to_process.append(sample)
    
    print(f"Found {len(samples_to_process)} samples to process.")

    for i in tqdm(range(0, len(samples_to_process), args.batch_size), desc="Processing Batches"):
        # Get the slice of samples for the current batch
        batch_start_index = i
        batch_end_index = min(i + args.batch_size, len(samples_to_process))
        current_batch_samples = samples_to_process[batch_start_index:batch_end_index]

        if not current_batch_samples:
            continue

        # Prepare inputs for the current batch
        multimodal_data_batch = []
        for sample in tqdm(current_batch_samples, total=len(current_batch_samples), leave=False):
            messages = format_chat_messages(
                query=sample['query'],
                video_path=sample['video_path'],
                num_frames=args.num_frames,
                start_time=sample['start_time'],
                end_time=sample['end_time'],
            )

            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            mm_data = {}
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": text_prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
            multimodal_data_batch.append(llm_inputs)

        # Run inference on the current batch
        print(f"\nRunning inference on batch of {len(multimodal_data_batch)} samples...")
        outputs = llm.generate(
            multimodal_data_batch,
            sampling_params=sampling_params,
        )

        # Store results from the current batch
        for j, output in enumerate(outputs):
            sample = current_batch_samples[j]
            prediction = output.outputs[0].text
            sample['prediction'] = prediction
            results_dict[sample['shot_id']] = sample

            if args.print_prediction:
                print(f"--- Sample: {sample['shot_id']} ---")
                print(sample['query'])
                print('-' * 100)
                print(prediction)
                print('-' * 100)
        
        # Save results to disk after each batch is processed
        print(f"Saving results after batch... Total saved: {len(results_dict)}")
        with open(args.output_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

    print(f"\nProcessing complete. Final results for {len(results_dict)} samples saved to {args.output_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
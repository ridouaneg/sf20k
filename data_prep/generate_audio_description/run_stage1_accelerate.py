import os
import argparse
from tqdm import tqdm
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from accelerate import Accelerator

# 'pip install qwen-vl-utils'
from qwen_vl_utils import process_vision_info

# Example usage with 'accelerate' for multi-GPU inference.
# The '--num_processes' flag specifies the number of GPUs to use.
# The new '--batch_size' argument sets the per-GPU batch size.
cmd_lines = """
# Run on 2 GPUs with a per-GPU batch size of 4 (total batch size = 2 * 4 = 8)
accelerate launch --num_processes 2 run_stage1_accelerate.py \
    --output_path ./results/stage1_qwen2vl-3b_accelerate.json \
    --data_path /users/ghermi/code/sf20k/data/train.csv \
    --shots_path /users/ghermi/code/sf20k/data/shots.parquet \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --num_frames 8 \
    --batch_size 4 \
    --print_prediction \
    --force_rerun

# Run on 4 GPUs with a per-GPU batch size of 2 (total batch size = 4 * 2 = 8)
accelerate launch --num_processes 4 run_stage1_accelerate.py \
    --output_path ./results/stage1_qwen2vl-7b_accelerate.json \
    --data_path /users/ghermi/code/sf20k/data/train.csv \
    --shots_path /users/ghermi/code/sf20k/data/shots.parquet \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --num_frames 8 \
    --batch_size 2
"""

STAGE1_PROMPT = (
    "Please describe the movie clip in the following four steps: "
    "1. Identify main characters; "
    "2. Describe the actions of characters in one sentence, i.e., who is doing what, focusing on the movements; "
    "3. Describe the interactions between characters in one sentence, such as looking; "
    "4. Describe the facial expressions of characters in one sentence. "
    "Make sure you do not hallucinate information. "
    "###ANSWER TEMPLATE###: 1. Main characters: ''; 2. Actions: ''; 3. Character-character interactions: ''; 4. Facial expressions: ''."
)


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
        self.all_clips = df_shots.to_dict(orient='records')
        print(f"Loaded {len(self.all_clips)} clips")

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


class QwenVLModel:
    def __init__(
        self,
        weights_dir: str,
        model_id: str,
        num_frames: int = 8,
        load_in_4bit: bool = False,
    ):
        model_path = os.path.join(weights_dir, model_id)
        model, processor = self.load_model(model_path=model_path, load_in_4bit=load_in_4bit)

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.num_frames = num_frames

    def load_model(self, model_path: str, load_in_4bit: bool = False):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        
        # NOTE: Removed device_map="auto". 'accelerate' will handle device placement.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            quantization_config=bnb_config,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=True,
        )
        return model, processor

    @staticmethod
    def format_chat_template(
        query: str,
        video_path: str,
        num_frames: int,
        start_time: float,
        end_time: float,
    ):
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": num_frames, "video_start": start_time, "video_end": end_time},
                {"type": "text", "text": query},
            ]
        }]
        return messages

    def generate_batch(
        self,
        batch: dict,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        # Unpack batch data
        queries = batch['query']
        video_paths = batch['video_path']
        start_times = batch['start_time']
        end_times = batch['end_time']

        # Prepare inputs for the entire batch
        batch_messages = []
        for i in range(len(queries)):
            messages = self.format_chat_template(
                query=queries[i],
                video_path=video_paths[i],
                start_time=start_times[i].item(),
                end_time=end_times[i].item(),
                num_frames=self.num_frames,
            )
            batch_messages.append(messages)

        # Process text and vision info for the batch
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        all_video_inputs = []
        all_video_kwargs = {}
        for msg in batch_messages:
            _, video_inputs, video_kwargs = process_vision_info(msg, return_video_kwargs=True)
            all_video_inputs.extend(video_inputs)
            # Assuming video_kwargs are the same for all videos in the batch
            if not all_video_kwargs:
                all_video_kwargs = video_kwargs

        inputs = self.processor(
            text=texts,
            videos=all_video_inputs,
            padding=True,
            return_tensors="pt",
            **all_video_kwargs,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        predictions = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return predictions


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
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0)
    # Others
    parser.add_argument("--force_rerun", action='store_true')
    parser.add_argument("--print_prediction", action='store_true')
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    # Initialize Accelerator
    accelerator = Accelerator()

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

    # Resume inference: Only main process handles file I/O
    results_dict = {}
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        if os.path.exists(args.output_path) and not args.force_rerun:
            with open(args.output_path, 'r') as f:
                results_dict = json.load(f)
    
    # Filter out already processed samples
    processed_shot_ids = set(results_dict.keys())
    if not args.force_rerun and len(processed_shot_ids) > 0:
        original_size = len(dataset.all_clips)
        dataset.all_clips = [sample for sample in dataset.all_clips if sample['shot_id'] not in processed_shot_ids]
        accelerator.print(f"Resuming inference. Found {len(processed_shot_ids)} completed samples. Remaining: {len(dataset.all_clips)}/{original_size}")
    
    # Prepare DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Prepare model
    model_wrapper = QwenVLModel(
        weights_dir=args.weights_dir,
        model_id=args.model_id,
        num_frames=args.num_frames,
    )

    # Use accelerator to prepare model and dataloader for distributed training/inference
    model, dataloader = accelerator.prepare(model_wrapper.model, dataloader)
    model_wrapper.model = model # Update the model reference in the wrapper
    
    # Run inference
    # TQDM progress bar only on the main process
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    for batch in dataloader:
        # Get responses for the batch
        predictions = model_wrapper.generate_batch(
            batch=batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )

        # Store predictions for the current batch
        for i in range(len(predictions)):
            shot_id = batch['shot_id'][i]
            sample = {key: val[i] for key, val in batch.items()}
            # Convert tensors to native python types for JSON serialization if necessary
            sample['start_time'] = sample['start_time'].item()
            sample['end_time'] = sample['end_time'].item()
            sample['prediction'] = predictions[i]
            
            results_dict[shot_id] = sample

            if args.print_prediction and accelerator.is_local_main_process:
                print(f"\n--- Shot ID: {shot_id} ---")
                print(f"Query: {sample['query']}")
                print('-' * 50)
                print(f"Prediction: {sample['prediction']}")
                print('-' * 100)

        progress_bar.update(1)

    # Save results - only the main process writes the file
    if accelerator.is_main_process:
        with open(args.output_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        accelerator.print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
import os
import argparse
from tqdm import tqdm
import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import cv2
from PIL import Image
import decord

# 'pip install qwen-vl-utils'
from qwen_vl_utils import process_vision_info


cmd_lines = """

CUDA_VISIBLE_DEVICES=1 python run_stage1.py \
    --output_path ./results/stage1_qwen2vl-3b.json \
    --data_path /users/ghermi/code/sf20k/data/train.csv \
    --shots_path /users/ghermi/code/sf20k/data/shots.parquet \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --num_frames 8 \
    --print_prediction \
    --force_rerun \
    --n_subsample 2

python run_stage1.py \
    --output_path ./results/stage1_qwen2vl-7b.json \
    --data_path /users/ghermi/code/sf20k/data/train.csv \
    --shots_path /users/ghermi/code/sf20k/data/shots.parquet \
    --video_dir /geovic/geovic/SF20K/videos/ \
    --weights_dir /geovic/ghermi/weights \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --num_frames 8 \
    --start_idx 0 \
    --end_idx 1024

"""


STAGE1_PROMPT = (
    "Please describe the movie clip in the following four steps: "
    #"1. Identify main characters (if {label_type} are available){char_text}; "
    "1. Identify main characters; "
    "2. Describe the actions of characters in one sentence, i.e., who is doing what, focusing on the movements; " 
    "3. Describe the interactions between characters in one sentence, such as looking; "
    "4. Describe the facial expressions of characters in one sentence. "
    #"Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "   
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


class QwenVLModel:

    def __init__(
        self,
        weights_dir: str,
        model_id: str,
        num_frames: int = 8,
        load_in_4bit: bool = False,
    ):
        assert model_id in [
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ]

        model_path = os.path.join(weights_dir, model_id)
        model, processor = self.load_model(model_path=model_path, load_in_4bit=load_in_4bit)

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.num_frames = num_frames

    def load_model(self, model_path: str, load_in_4bit: bool = False):
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
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
        messages = []
        content = []
        content.append({"type": "video", "video": video_path, "nframes": num_frames, "video_start": start_time, "video_end": end_time})
        #content.append({"type": "video"})
        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})
        return messages

    def generate(
        self,
        query: str,
        video_path: str, 
        start_time: float = 0.0,
        end_time: float = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        messages = self.format_chat_template(
            query=query,
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            num_frames=self.num_frames,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        prediction = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return prediction


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
    # Others
    parser.add_argument("--force_rerun", action='store_true')
    parser.add_argument("--print_prediction", action='store_true')
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    model = QwenVLModel(
        weights_dir=args.weights_dir,
        model_id=args.model_id,
        num_frames=args.num_frames,
    )

    # Resume inference
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results_dict = json.load(open(args.output_path, 'r')) if os.path.exists(args.output_path) and not args.force_rerun  else {}

    # Run inference
    for sample in tqdm(dataset, total=len(dataset)):
        if sample['shot_id'] in results_dict and not args.force_rerun:
            continue

        # Get response
        prediction = model.generate(
            video_path=sample['video_path'],
            query=sample['query'],
            start_time=sample['start_time'],
            end_time=sample['end_time'],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )

        # Store the prediction
        sample['prediction'] = prediction
        results_dict[sample['shot_id']] = sample

        if args.print_prediction:
            print(sample['query'])
            print('-' * 100)
            print(sample['prediction'])
            print('-' * 100)
            print('-' * 100)

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
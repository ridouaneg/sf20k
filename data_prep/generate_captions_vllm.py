import os
import cv2
import pandas as pd
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

VLM_PROMPT = (
    "Please describe the movie clip in the following four steps: "
    "1. Describe the main characters; "
    "2. Describe the actions of characters in one sentence, i.e., who is doing what, focusing on the movements; " 
    "3. Describe the interactions between characters in one sentence, such as looking; "
    "4. Describe the facial expressions of characters in one sentence. "
    "Make sure you do not hallucinate information. "
    "###ANSWER TEMPLATE###: 1. Main characters: ''; 2. Actions: ''; 3. Character-character interactions: ''; 4. Facial expressions: ''."
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="train_video_ids.json")
    parser.add_argument("--output_path", type=str, default="captions.parquet")
    parser.add_argument("--video_dir", type=str, default="/geovic/geovic/SF20K/videos/")
    parser.add_argument("--shots_path", type=str, default="/geovic/geovic/SF20K/shots.parquet")
    parser.add_argument("--weights_dir", type=str, default="/geovic/ghermi/weights/")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    # Qwen2-VL-2B-Instruct, Qwen2.5-VL-3B-Instruct, Qwen3-VL-2B-Instruct
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Prepare Data
    video_ids = json.load(open(args.input_path))
    video_paths = {video_id: os.path.join(args.video_dir, video_id + ".mkv") for video_id in video_ids}
    video_paths = {k: v for k, v in video_paths.items() if os.path.exists(v)}
    
    shots = pd.read_parquet(args.shots_path)
    # Filter valid videos
    valid_ids = set(shots['video_id'].unique()).intersection(set(video_paths.keys()))
    video_paths = {k: v for k, v in video_paths.items() if k in valid_ids}
    shots = shots[shots['video_id'].isin(valid_ids)]

    # 2. Initialize vLLM
    # We use the processor only for text formatting (applying chat template)
    model_full_path = os.path.join(args.weights_dir, args.model_id)
    processor = AutoProcessor.from_pretrained(model_full_path, trust_remote_code=True)

    print(f"Loading vLLM model from {model_full_path}...")
    llm = LLM(
        model=model_full_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"video": 1}, # Allow 1 video per prompt
    )

    sampling_params = SamplingParams(
        temperature=0.2, # Low temp for factual descriptions
        max_tokens=2048,
    )

    all_results = []

    # 3. Process Video by Video
    for video_id, video_path in tqdm(video_paths.items(), desc="Processing Videos"):
        # --- A. Frame Extraction (CPU Bound) ---
        scene_data = shots[shots['video_id'] == video_id]
        if scene_data.empty:
            continue

        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate all needed frames for this video's shots to avoid decoding unused frames
        # However, your original logic sampled the whole video at 1 FPS (approx). 
        # We will keep your logic of sampling the whole video to ensure coverage.
        frame_indices_to_capture = range(0, total_frames, int(video_fps)) # 1 FPS
        
        frames_cache = {}
        for fid in frame_indices_to_capture:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                # vLLM supports PIL or numpy. PIL is safer for resize/transforms.
                frames_cache[fid] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # --- B. Batch Preparation ---
        batch_prompts = []
        batch_mm_data = []
        batch_metadata = [] # To track which shot corresponds to which output
        
        for _, row in scene_data.iterrows():
            shot_id = row['shot_id']
            start_frame, end_frame = int(row['Start Frame']), int(row['End Frame'])

            # Filter frames for this shot
            shot_frame_ids = [fid for fid in frame_indices_to_capture if start_frame <= fid <= end_frame]
            shot_frame_ids.sort()
            
            # Skip if insufficient frames
            if len(shot_frame_ids) < 1:
                all_results.append({
                    "video_id": video_id, "shot_id": shot_id,
                    "caption": None, # or empty string
                    "start_second": row['Start Time (seconds)'], "end_second": row['End Time (seconds)'],
                    "start_frame": start_frame, "end_frame": end_frame
                })
                continue
            
            shot_frames = [frames_cache[fid] for fid in shot_frame_ids]

            # Prepare Message for Processor
            # We use the processor to generate the text prompt with correct special tokens
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": shot_frames}, # Placeholder for processor
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ]
            
            # Apply chat template to get the raw text prompt (e.g. "<|video_pad|> Describe...")
            prompt_text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            batch_prompts.append(prompt_text)
            batch_mm_data.append({"video": shot_frames}) # vLLM handles PIL images list as video
            batch_metadata.append({
                "video_id": video_id,
                "shot_id": shot_id,
                "start_second": row['Start Time (seconds)'],
                "end_second": row['End Time (seconds)'],
                "start_frame": start_frame,
                "end_frame": end_frame
            })

        # --- C. Batch Inference (GPU Bound) ---
        if batch_prompts:
            outputs = llm.generate(
                prompts=batch_prompts,
                multi_modal_data=batch_mm_data,
                sampling_params=sampling_params,
                use_tqdm=False 
            )

            # --- D. Collect Results ---
            for i, output in enumerate(outputs):
                meta = batch_metadata[i]
                generated_text = output.outputs[0].text
                meta["caption"] = generated_text
                all_results.append(meta)

        # Force garbage collection of frames to free RAM
        del frames_cache
        
    # 4. Save
    pd.DataFrame(all_results).to_parquet(args.output_path, index=False)
    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    main()
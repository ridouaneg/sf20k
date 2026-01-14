import os
import cv2
import pandas as pd
import json
import argparse
import math
import torch
from tqdm import tqdm
from PIL import Image

# vLLM and Qwen Utils
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Set spawn method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

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
    parser.add_argument("--output_path", type=str, default="results/captions.parquet")
    parser.add_argument("--video_dir", type=str, default="/geovic/geovic/SF20K/videos/")
    parser.add_argument("--shots_path", type=str, default="shots.parquet")
    parser.add_argument("--weights_dir", type=str, default="/geovic/ghermi/weights/")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100, help="Save parquet every N videos")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of parallel jobs")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of current job (0 to num_chunks-1)")
    return parser.parse_args()

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True, 
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def get_chunk(full_list, num_chunks, chunk_idx):
    """
    Splits a list into chunks and returns the requested slice.
    """
    if num_chunks <= 1:
        return full_list
    
    # Sort to ensure deterministic splitting across jobs
    full_list = sorted(full_list)
    chunk_size = math.ceil(len(full_list) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(full_list))
    
    return full_list[start_idx:end_idx]

def main():
    args = parse_args()

    # 1. Prepare Data
    video_ids = json.load(open(args.input_path))
    video_paths = {video_id: os.path.join(args.video_dir, video_id + ".mkv") for video_id in video_ids}
    video_paths = {k: v for k, v in video_paths.items() if os.path.exists(v)}
    
    shots = pd.read_parquet(args.shots_path)
    valid_ids = set(shots['video_id'].unique()).intersection(set(video_paths.keys()))
    
    # --- SUBSET LOGIC ---
    # Convert to sorted list to ensure consistency across jobs
    valid_id_list = sorted(list(valid_ids))
    
    # Slice the list for this specific job
    my_video_ids = get_chunk(valid_id_list, args.num_chunks, args.chunk_idx)
    
    print(f"Job {args.chunk_idx}/{args.num_chunks}: Processing {len(my_video_ids)} videos (Total dataset: {len(valid_id_list)})")
    
    # Filter dictionary and dataframe
    video_paths = {k: v for k, v in video_paths.items() if k in my_video_ids}
    shots = shots[shots['video_id'].isin(my_video_ids)]

    # --- OUTPUT PATH LOGIC ---
    # If running in chunks, modify filename to prevent overwrites (e.g. captions_part_0.parquet)
    if args.num_chunks > 1:
        base, ext = os.path.splitext(args.output_path)
        final_output_path = f"{base}_part_{args.chunk_idx}{ext}"
    else:
        final_output_path = args.output_path

    # 2. Initialize Model & Processor
    model_full_path = os.path.join(args.weights_dir, args.model_id)
    processor = AutoProcessor.from_pretrained(model_full_path, trust_remote_code=True)

    print(f"Loading vLLM model from {model_full_path}...")
    llm = LLM(
        model=model_full_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.2, 
        max_tokens=2048,
    )

    all_results = []
    
    # Check if we should resume (optional simple check)
    if os.path.exists(final_output_path):
        print(f"Warning: Output file {final_output_path} already exists. New results will overwrite/append depending on logic.")
        # Optional: Load existing results to skip processing? 
        # For now, we overwrite to keep it simple as per prompt instructions.

    # 3. Process Video by Video
    for i, (video_id, video_path) in enumerate(tqdm(video_paths.items(), desc=f"Job {args.chunk_idx}")):
        scene_data = shots[shots['video_id'] == video_id]
        if scene_data.empty:
            continue

        # --- A. Frame Extraction ---
        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = range(0, total_frames, int(video_fps)) # Sample 1 FPS
        
        frames_cache = {}
        for frame_id in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frames_cache[frame_id] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # --- B. Batch Preparation ---
        batch_inputs = []
        batch_metadata = []

        for _, row in scene_data.iterrows():
            shot_id = row['shot_id']
            start_frame, end_frame = int(row['Start Frame']), int(row['End Frame'])

            shot_frame_ids = [frame_id for frame_id in frames_cache.keys() if start_frame <= frame_id <= end_frame]
            shot_frame_ids.sort()
            if len(shot_frame_ids) < 2:
                continue 
            
            shot_frames = [frames_cache[frame_id] for frame_id in shot_frame_ids]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video", 
                            "video": shot_frames,
                        }, 
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ]
            
            vllm_input = prepare_inputs_for_vllm(messages, processor)
            batch_inputs.append(vllm_input)
            batch_metadata.append({
                "video_id": video_id,
                "shot_id": shot_id,
                "start_second": row['Start Time (seconds)'],
                "end_second": row['End Time (seconds)'],
                "start_frame": start_frame,
                "end_frame": end_frame
            })

        # --- C. Batch Inference ---
        if batch_inputs:
            outputs = llm.generate(
                batch_inputs,
                sampling_params=sampling_params,
                use_tqdm=False
            )

            # --- D. Collect Results ---
            for j, output in enumerate(outputs):
                meta = batch_metadata[j]
                meta["caption"] = output.outputs[0].text
                all_results.append(meta)

        # Clear memory
        del frames_cache

        # --- E. Intermediate Saving ---
        if (i + 1) % args.save_interval == 0:
            pd.DataFrame(all_results).to_parquet(final_output_path, index=False)
            tqdm.write(f"Saved {len(all_results)} shots to {final_output_path}")
        
    # 4. Final Save
    pd.DataFrame(all_results).to_parquet(final_output_path, index=False)
    print(f"Finished Job {args.chunk_idx}. Saved to {final_output_path}")

if __name__ == "__main__":
    main()
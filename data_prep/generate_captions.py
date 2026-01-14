import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import argparse
from decord import VideoReader, cpu
import numpy as np

VLM_PROMPT = (
    "Please describe the movie clip in the following four steps: "
    "1. Describe the main characters; "
    "2. Describe the actions of characters in one sentence, i.e., who is doing what, focusing on the movements; " 
    "3. Describe the interactions between characters in one sentence, such as looking; "
    "4. Describe the facial expressions of characters in one sentence. "
    "Make sure you do not hallucinate information. "
    "###ANSWER TEMPLATE###: 1. Main characters: ''; 2. Actions: ''; 3. Character-character interactions: ''; 4. Facial expressions: ''."
)

cmd_line = """
python generate_captions.py \
    --input_path train_video_ids.json \
    --output_path results/captions.parquet \
    --video_dir /lustre/fswork/projects/rech/kcn/ucm72yx/data/SF20K/videos/ \
    --shots_path shots.parquet \
    --weights_dir /lustre/fsn1/projects/rech/kcn/ucm72yx/weights/ \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --save_interval 100 \
    --num_chunks 20 \
    --chunk_idx 4 \
    --use_flash_attn
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="train_video_ids.json")
    parser.add_argument("--output_path", type=str, default="captions.parquet")
    parser.add_argument("--video_dir", type=str, default="/geovic/geovic/SF20K/videos/")
    parser.add_argument("--shots_path", type=str, default="/geovic/geovic/SF20K/shots.parquet")
    parser.add_argument("--weights_dir", type=str, default="/geovic/ghermi/weights/")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--use_flash_attn", action="store_true")
    return parser.parse_args()

def inference(
    video, 
    prompt, 
    model,
    processor,
    max_new_tokens=2048, 
    total_pixels=20480 * 32 * 32, 
    min_pixels=64 * 32 * 32, 
    max_frames=2048, 
    sample_fps=2,
):
    messages = [
        {"role": "user", "content": [
                {"video": video,
                "total_pixels": total_pixels, 
                "min_pixels": min_pixels, 
                "max_frames": max_frames,
                'sample_fps':sample_fps},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True, 
        image_patch_size=16,
        return_video_metadata=True
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        video_metadata=video_metadatas, 
        **video_kwargs, 
        do_resize=False, 
        return_tensors="pt"
    ).to(
        device=model.device, 
        dtype=model.dtype
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens
        )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return output_text[0]

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

    # Prepare data
    video_ids = json.load(open(args.input_path))
    video_paths = {video_id: os.path.join(args.video_dir, video_id + ".mkv") for video_id in video_ids}
    video_paths = {k: v for k, v in video_paths.items() if os.path.exists(v)}

    shots = pd.read_parquet(args.shots_path)
    valid_ids = set(shots['video_id'].unique()).intersection(set(video_paths.keys()))

    valid_id_list = sorted(list(valid_ids))
    my_video_ids = get_chunk(valid_id_list, args.num_chunks, args.chunk_idx)
    print(f"Job {args.chunk_idx}/{args.num_chunks}: Processing {len(my_video_ids)} videos (Total dataset: {len(valid_id_list)})")

    video_paths = {k: v for k, v in video_paths.items() if k in my_video_ids}
    shots = shots[shots['video_id'].isin(my_video_ids)]

    # Prepare VLM
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    
    model_path = os.path.join(args.weights_dir, args.model_id)
    processor = AutoProcessor.from_pretrained(model_path)

    if args.use_flash_attn:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            quantization_config=bnb_config,
        )

    # Resume
    if args.num_chunks > 1:
        base, ext = os.path.splitext(args.output_path)
        final_output_path = f"{base}_part_{args.chunk_idx}{ext}"
    else:
        final_output_path = args.output_path
    
    all_results = []
    if os.path.exists(final_output_path):
        print(f"Warning: Output file {final_output_path} already exists. New results will overwrite/append depending on logic.")
    
    # Run captioning
    for i, (video_id, video_path) in enumerate(tqdm(video_paths.items(), desc=f"Job {args.chunk_idx}")):
        scene_data = shots[shots['video_id'] == video_id]
        if scene_data.empty:
            continue

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        results = []

        for _, row in tqdm(scene_data.iterrows(), total=len(scene_data), leave=True):
            shot_id = row['shot_id']
            start_frame, end_frame = int(row['Start Frame']), int(row['End Frame'])

            step = int(fps) if fps > 0 else 1
            #indices = list(range(start_frame, min(end_frame, len(vr)), step))
            num_frames = 8
            indices = list(np.linspace(start_frame, end_frame - 1, num=num_frames))
            if len(indices) < 2:
                continue
            
            frames = vr.get_batch(indices).asnumpy() 
            shot_frames = [Image.fromarray(f) for f in frames]
            
            caption = inference(video=shot_frames, prompt=VLM_PROMPT, model=model, processor=processor)
            #print(caption)

            results.append({
                "video_id": video_id,
                "shot_id": shot_id,
                #"start_second": start_s,
                #"end_second": end_s,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "caption": caption,
            })

        all_results.extend(results)

        if (i + 1) % args.save_interval == 0:
            pd.DataFrame(all_results).to_parquet(final_output_path, index=False)
            tqdm.write(f"Saved {len(all_results)} shots to {final_output_path}")

    # Save results
    pd.DataFrame(all_results).to_parquet(final_output_path, index=False)
    print(f"Finished Job {args.chunk_idx}. Saved to {final_output_path}")
    

if __name__ == "__main__":
    main()
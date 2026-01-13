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
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--load_in_4bit", action="store_true")
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


def main():
    args = parse_args()

    # Prepare data
    video_ids = json.load(open(args.input_path))
    video_paths = {video_id: os.path.join(args.video_dir, video_id + ".mkv") for video_id in video_ids}
    video_paths = {k: v for k, v in video_paths.items() if os.path.exists(v)}
    shots = pd.read_parquet(args.shots_path)
    video_ids = set(shots['video_id'].unique()).intersection(set(video_paths.keys()))
    video_paths = {k: v for k, v in video_paths.items() if k in video_ids}
    shots = shots[shots['video_id'].isin(video_ids)]

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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Resume
    all_results = []

    # Run captioning
    for i, (video_id, video_path) in enumerate(tqdm(video_paths.items(), total=len(video_paths))):
        scenes = shots[shots['video_id'] == video_id]

        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = list(range(0, total_frames, int(video_fps)))
        
        frames = {}
        for frame_id in tqdm(frame_ids, total=len(frame_ids), leave=True):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frames[frame_id] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = []
        for i, row in tqdm(scenes.iterrows(), total=len(scenes), leave=True):
            shot_id = row['shot_id']
            start_s, end_s = float(row['Start Time (seconds)']), float(row['End Time (seconds)'])
            start_frame, end_frame = int(row['Start Frame']), int(row['End Frame'])

            scene_frame_ids = [fid for fid in frame_ids if start_frame <= fid <= end_frame]
            scene_frame_ids.sort()
            scene_frames = [frames[fid] for fid in scene_frame_ids]
            
            if len(scene_frames) < 2:
                caption = None
            else:
                caption = inference(video=scene_frames, prompt=VLM_PROMPT, model=model, processor=processor)
                print(video_id, shot_id, caption)

            results.append({
                "video_id": video_id,
                "shot_id": shot_id,
                "start_second": start_s,
                "end_second": end_s,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "caption": caption,
            })

        cap.release()

        all_results.extend(results)

    # Save results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_parquet(args.output_path, index=False)
    

if __name__ == "__main__":
    main()
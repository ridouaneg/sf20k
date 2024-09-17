from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX

from decord import VideoReader, cpu
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import logging
import os
import argparse
import random
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../../data/videos/")
    parser.add_argument("--subtitles_path", type=str, default="../../data/subtitles.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="lmms-lab/LongVA-7B-DPO")
    parser.add_argument("--question_mode", type=str, default="mcqa", choices=["mcqa", "oeqa"])
    parser.add_argument("--with_frames", action="store_true", default=False)
    parser.add_argument("--with_subtitles", action="store_true", default=False)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--force_rerun", action="store_true")
    return parser.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def convert_to_timestamp(decimal_seconds):
    whole_seconds, _ = str(decimal_seconds).split(".")
    total_seconds = int(whole_seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
    return timestamp

def format_subtitle(row):
    start_time = convert_to_timestamp(row.start)
    end_time = convert_to_timestamp(row.end)
    return f"{start_time} - {end_time} {row.text}"

def read_video(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames

def get_query(question_mode):
    pass

def main(args):
    # Parameters
    gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
    
    # Set seed and logging
    setup_logging()
    set_seed(args.seed)

    # Log config
    logging.info(args)

    # Prepare data
    dataset = load_dataset("rghermi/sfd", data_files={"test": f"data.csv"})
    df = pd.DataFrame(dataset['test'])
    df.drop_duplicates(subset=['question_id'], inplace=True)

    logging.info(f"Number of movies to process: {df.video_id.nunique()}")
    logging.info(f"Number of questions to process: {df.question_id.nunique()}")

    if args.n_subsample > 0:
        df = df.sample(n=args.n_subsample, random_state=args.seed)

    logging.info(f"Number of movies to process (after subsampling): {df.video_id.nunique()}")
    logging.info(f"Number of questions to process (after subsampling): {df.question_id.nunique()}")

    # Prepare subtitles
    subtitles = pd.read_csv(args.subtitles_path)
    subtitles = subtitles[(subtitles.video_id.isin(df.video_id.unique()))]

    logging.info(f"Number of movies with subtitles: {subtitles.video_id.nunique()}")

    # Prepare model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_name, None, "llava_qwen", device_map="auto",
        #load_8bit=True,
        load_4bit=True,
    )

    # Resume inference
    os.makedirs(args.output_dir, exist_ok=True)
    model_name_ = args.model_name.split("/")[-1]
    with_frames_ = f'with_frames_{args.with_frames}_num_frames_{args.num_frames}' if args.with_frames else f'with_frames_{args.with_frames}'
    output_file = f'sfd_{model_name_}_with_frames_{with_frames_}_with_subtitles_{args.with_subtitles}_question_mode_{args.question_mode}.json'
    output_path = os.path.join(args.output_dir, output_file)
    results = json.load(open(output_path)) if os.path.exists(output_path) and not args.force_rerun else {}

    logging.info(f"Results loaded from {output_path}")

    # Run inference
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Get ids
        question_id = row.question_id
        video_id = row.video_id

        # Check if already processed
        if question_id in results and not args.force_rerun:
            logging.info(f"Question {question_id} already processed")
            continue

        # Check if video (or features) exists
        video_path = os.path.join(args.video_dir, f"{video_id}.mkv")
        if not os.path.exists(video_path):
            logging.info(f"Video {video_id} not found")
            continue

        # Check if subtitles exists:
        if subtitles[(subtitles.video_id == video_id)].empty:
            logging.info(f"Subtitles for video {video_id} not found")
            continue

        # Prepare video
        frames = None
        if args.with_frames:
            frames = read_video(video_path, args.num_frames)
            video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

        # Prepare subtitles
        subs = None
        if args.with_subtitles:
            subs = subtitles[(subtitles.video_id == video_id)]
            subs = subs.sort_values('start')
            subs['aggregated_text'] = subs.apply(format_subtitle, axis=1)
            subs = '\n'.join(subs['aggregated_text'])

        # Prepare query
        query = "You will be given a question about a movie."

        if args.with_frames and args.with_subtitles:
            query += "Try to answer it based on the subtitles and the frames from the movie."
        elif args.with_frames and not args.with_subtitles:
            query += "Try to answer it based on the frames from the movie."
        elif not args.with_frames and args.with_subtitles:
            query += "Try to answer it based on the subtitles from the movie."
        elif not args.with_frames and not args.with_subtitles:
            query += "Try to answer it based on the movie."
        else:
            raise ValueError("Invalid combination of arguments")
        
        if args.with_subtitles:
            query += f"\n\nSubtitles: {subs}"

        if args.question_mode == "mcqa":
            query += f"""\n\nQuestion: {row.question}
Possible answer choices:
(1) {row.option_0}
(2) {row.option_1}
(3) {row.option_2}
(4) {row.option_3}
(5) {row.option_4}
Output the final answer in the format "(X)" where X is the correct digit choice. DO NOT OUTPUT with the full answer text or any other words."""
        elif args.question_mode == "oeqa":
            query += f"""\n\nQuestion: {row.question}
Answer it shortly and directly without repeating the question."""
        else:
            raise ValueError("Invalid question mode")

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        try:
            # Get response
            with torch.inference_mode():
                outputs = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            # Save results
            results[question_id] = {
                'question_id': row['question_id'],
                'video_id': row['video_id'],
                'question': row['question'],
                'answer': row[f'option_{row.correct_answer}'],
                'option_0': row['option_0'],
                'option_1': row['option_1'],
                'option_2': row['option_2'],
                'option_3': row['option_3'],
                'option_4': row['option_4'],
                'correct_answer': row['correct_answer'],
                'prediction': response,
            }
            with open(output_path, 'w') as file:
                json.dump(results, file, indent=4)
        
        except Exception as e:
            logging.error(f"Error processing movie {video_id}: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

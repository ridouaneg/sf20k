import os
import argparse
import numpy as np
import pandas as pd
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


TEMPLATE = (
    "You will be given a question about a movie. Try to answer it based on the subtitles and the frames from the movie.\n\n"
    "Subtitles:\n{subtitles}\n\n"
    "Question: {question}\n\n"
    "Answer it shortly and directly without repeating the question."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--subtitles_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--input_modality", type=str, default="vl", choices=["v", "l", "vl"])
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    return parser.parse_args()


def load_video(video_path: str, num_frames: int = 16):
    """Load video frames from a given path."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[-1] if frames else np.zeros((64, 64, 3), dtype=np.uint8)))

    cap.release()
    return frames


def main(args):
    # Prepare data
    df = pd.read_csv(args.data_path)
    df_subs = pd.read_csv(args.subtitles_path)

    # Prepare model
    model_path = os.path.join(args.model_dir, args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    predictions = []
    for i, sample in tqdm(df.iterrows(), total=len(df)):
        # Create the prompt
        video_path = os.path.join(args.video_dir, f"{sample['video_id']}.mp4")
        subtitles = '\n'.join(df_subs[(df_subs.video_id == sample['video_id'])].text.tolist())
        prompt = TEMPLATE.format(subtitles=subtitles, question=sample['question'])

        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "nframes": args.num_frames},
            {"type": "text", "text": prompt}
        ]}]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process the visual information
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # Prepare the inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        # Generate the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        prediction = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        predictions.append(prediction)

    df['prediction'] = predictions
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

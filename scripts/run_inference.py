import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 'pip install qwen-vl-utils'
from qwen_vl_utils import process_vision_info


#WIDTHS = {144: 256, 240: 426, 360: 640, 480: 854, 720: 1280, 1080: 1920}


TEMPLATE = (
    "You will be given a question about a movie. Try to answer it based on the subtitles and the frames from the movie.\n\n"
    "Subtitles:\n{subtitles}\n\n"
    "Question: {question}\n\n"
    "Answer it shortly and directly without repeating the question."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_expert")
    parser.add_argument("--subtitles_path", type=str, default="../data/test_subtitles.csv")
    parser.add_argument("--output_path", type=str, default="submission.csv")
    parser.add_argument("--video_dir", type=str, default="../data/videos/")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=8)
    #parser.add_argument("--resolution", type=int, default=360, choices=[144, 240, 360, 480, 720, 1080], help="Resolution for video download (e.g., 360, 720, 1080)")    parser.add_argument("--input_modality", type=str, default="vl", choices=["v", "l", "vl"])
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    return parser.parse_args()


def main(args):
    # Prepare data
    #video_dir = os.path.join(args.video_dir, f"{args.resolution}p")
    df = load_dataset("rghermi/sf20k", split=args.split).to_pandas()
    df = df[['question_id', 'video_id', 'video_url', 'question']]
    df = df.sample(n=32, random_state=42)
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
        #video_path = os.path.join(video_dir, f"{sample['video_id']}.mp4")
        #video_path = os.path.join(args.video_dir, f"{sample['video_id']}.mp4")
        video_path = os.path.join(args.video_dir, f"{sample['video_id']}.mkv")
        subtitles = '\n'.join(df_subs[(df_subs.video_id == sample['video_id'])].text.tolist())
        prompt = TEMPLATE.format(subtitles=subtitles, question=sample['question'])

        messages = [{"role": "user", "content": [
            {
                "type": "video",
                "video": video_path,
                "nframes": args.num_frames,
                #"resized_height": args.resolution,
                #"resized_width": WIDTHS[args.resolution],
            },
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

        # Save intermediate results
        #if i % 50:
        #    df['prediction'] = predictions
        #    df.to_csv(args.output_path, index=False)

    # Save results
    df['prediction'] = predictions
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

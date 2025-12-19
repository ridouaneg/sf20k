import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


#### prompts.py
class OEQAPrompt:

    TEMPLATE_VL = (
        "You will be given a question about a movie. Try to answer it based on the subtitles and the frames from the movie.\n\n"
        "Subtitles:\n{subtitles}\n\n"
        "Question: {question}\n\n"
        "Answer it shortly and directly without repeating the question."
    )

    TEMPLATE_L = (
        "You will be given a question about a movie. Try to answer it based on the subtitles from the movie.\n\n"
        "Subtitles:\n{subtitles}\n\n"
        "Question: {question}\n\n"
        "Answer it shortly and directly without repeating the question."
    )

    TEMPLATE_V = (
        "You will be given a question about a movie. Try to answer it based on the frames from the movie.\n\n"
        "Question: {question}\n\n"
        "Answer it shortly and directly without repeating the question."
    )

    def __init__(self, modality="vision_language"):
        self.modality = modality

    def get_query(self, sample):
        question = sample['question']
        subtitles = sample['subtitles']
        
        if self.modality == "vision_language":
            return self.TEMPLATE_VL.format(question=question, subtitles=subtitles)
        elif self.modality == "language":
            return self.TEMPLATE_L.format(question=question, subtitles=subtitles)
        elif self.modality == "vision":
            return self.TEMPLATE_V.format(question=question)
        else:
            raise ValueError(f"Invalid modality: {self.modality}")

    def get_response(self, sample):
        return f"{sample['answer']}"

    def postprocess_response(self, response):
        return response.strip() if response is not None else None


#### datasets.py
import os
import pandas as pd
from torch.utils.data import Dataset
import ast
import cv2


class SF20KSceneDataset(Dataset):

    def __init__(
        self,
        prompt = None,
        data_path: str = None,
        video_dir: str = None,
        subtitles_path: str = None,
        num_segments: int = 10,
        n_subsample: int = -1,
        seed: int = 42,
    ):
        df = pd.read_csv(data_path)
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        self.df = df
        self.df_subs = pd.read_csv(subtitles_path)
        self.video_dir = video_dir
        self.num_segments = num_segments
        self.prompt = prompt

    def get_video_duration(self, path):
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count / fps if fps > 0 else 0
        
    def __len__(self):
        return len(self.df) * self.num_segments

    def __getitem__(self, idx):
        q_idx = idx // self.num_segments
        seg_idx = idx % self.num_segments
        
        sample = self.df.iloc[q_idx].copy()
        video_id = sample['video_id']
        video_path = os.path.join(self.video_dir, f"{video_id}.mkv")

        duration = self.get_video_duration(video_path)
        seg_duration = duration / self.num_segments
        start_t = seg_idx * seg_duration
        end_t = (seg_idx + 1) * seg_duration

        relevant_subs = self.df_subs[
            (self.df_subs.video_id == video_id) & 
            (self.df_subs.start >= start_t) & 
            (self.df_subs.end <= end_t)
        ]
        sub_text = '\n'.join(relevant_subs.text.fillna('').astype(str).tolist())
        sample['subtitles'] = sub_text if sub_text.strip() else "No subtitles in this segment."

        query = self.prompt.get_query(sample)
        response = self.prompt.get_response(sample)

        return {
            'question_id': f"{sample['question_id']}_seg_{seg_idx}",
            'original_question_id': sample['question_id'],
            'segment_idx': seg_idx,
            'video_id': video_id,
            'video_path': video_path,
            'start_time': start_t,
            'end_time': end_t,
            'question': sample['question'],
            'answer': sample['answer'],
            'options': [sample[f'option_{i}'] for i in range(5)] if 'option_0' in sample else None,
            'answer_id': sample['answer_id'] if 'answer_id' in sample else None,
            'query': query,
            'response': response,
        }


#### models.py
import os
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    )
except:
    Qwen2_5_VLForConditionalGeneration = None
    Qwen3VLForConditionalGeneration = None
    Qwen3VLMoeForConditionalGeneration = None

from qwen_vl_utils import process_vision_info


class QwenVLModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        adapter_path: str = None,
        modality: str = "vision_language",
        load_in_4bit: bool = False,
        **kwargs,
    ):
        assert modality in ["vision", "language", "vision_language"]

        assert model_name in [
            # Qwen2.5-VL
            "qwen2.5-vl-3b",
            "qwen2.5-vl-7b",
            "qwen2.5-vl-32b",
            "qwen2.5-vl-72b",
            # Qwen3-VL - Dense
            "qwen3-vl-2b",
            "qwen3-vl-4b",
            "qwen3-vl-8b",
            "qwen3-vl-32b",
            # Qwen3-VL - Dense - Thinking
            "qwen3-vl-2b-think",
            "qwen3-vl-4b-think",
            "qwen3-vl-8b-think",
            # Qwen3-VL - MoE
            "qwen3-vl-30b-a3b",
            "qwen3-vl-235b-a22b",
        ]

        dict_model_name_to_model_id = {
            "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
            "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
            "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
            "qwen3-vl-2b-think": "Qwen/Qwen3-VL-2B-Thinking",
            "qwen3-vl-4b-think": "Qwen/Qwen3-VL-4B-Thinking",
            "qwen3-vl-8b-think": "Qwen/Qwen3-VL-8B-Thinking",
            "qwen3-vl-32b-think": "Qwen/Qwen3-VL-32B-Thinking",
            "qwen3-vl-30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "qwen3-vl-235b-a22b": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        }
        model_id = dict_model_name_to_model_id[model_name]

        if model_id in [
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ]:
            self.model_class_ = Qwen2_5_VLForConditionalGeneration
            self.processor_class_ = AutoProcessor
        elif model_id in [
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-2B-Thinking",
            "Qwen/Qwen3-VL-4B-Thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-32B-Thinking",
        ]:
            self.model_class_ = Qwen3VLForConditionalGeneration
            self.processor_class_ = AutoProcessor
        elif model_id in [
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
        ]:
            self.model_class_ = Qwen3VLMoeForConditionalGeneration
            self.processor_class_ = AutoProcessor
        else:
            raise ValueError(f"Model {model_id} not supported")

        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id
        model, processor = self.load_model(
            model_path=model_path, 
            load_in_4bit=load_in_4bit, 
            adapter_path=adapter_path,
        )

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.modality = modality
        
    def load_model(
        self, 
        model_path: str, 
        load_in_4bit: bool = False,
        adapter_path: str = None,
    ):
        processor = self.processor_class_.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=True,
        )

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None


        model = self.model_class_.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )

        if isinstance(adapter_path, list):
            for path in adapter_path:
                if path is not None:
                    peft_model = PeftModel.from_pretrained(model, path)
                    model = peft_model.merge_and_unload()
        elif isinstance(adapter_path, str):
            if adapter_path is not None:
                peft_model = PeftModel.from_pretrained(model, adapter_path)
                model = peft_model.merge_and_unload()
        
        return model, processor

    @staticmethod
    def format_chat_template(
        query: str,
        video_path: str,
        start_time: float,
        end_time: float,
        modality: str = "vision_language",
        system_prompt: str = None,
        ground_truth: str = None,
        fps: float = 1.0,
        max_frames: int = 8,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        
        content = []
        if modality in ["vision", "vision_language"]:
            content.append({
                "type": "video",
                "video": video_path,
                "total_pixels": total_pixels, 
                "min_pixels": min_pixels, 
                "max_frames": max_frames,
                "fps": fps,
                "start_time": start_time,
                "end_time": end_time,
            })
        content.append({"type": "text", "text": query})

        messages.append({"role": "user", "content": content})
        if ground_truth is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ground_truth}]})

        return messages

    def generate(
        self,
        query: str,
        video_path: str, 
        start_time: float,
        end_time: float,
        fps: float = 1.0,
        max_frames: int = 8,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
        **kwargs,
    ):
        messages = self.format_chat_template(
            query=query,
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            fps=fps,
            max_frames=max_frames,
            modality=self.modality,
            system_prompt=system_prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text_inputs = [text]

        if self.modality in ["vision", "vision_language"]:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages],
                return_video_kwargs=True, 
                image_patch_size=16,
                return_video_metadata=True
            )
        else:
            image_inputs = None
            video_inputs = None
            video_kwargs = {}

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        
        inputs = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return response


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines on SF20K dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--subtitles_path", type=str, required=True, help="Path to the subtitles CSV file")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory containing model weights")
    parser.add_argument("--modality", type=str, default="vision_language", choices=["vision", "language", "vision_language"], help="Modality to use")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to sample")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for sampling")
    parser.add_argument("--n_subsample", type=int, default=-1, help="Number of samples to run (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerunning all samples")
    parser.add_argument("--num_segments", type=int, default=1, help="Number of segments to sample")
    parser.add_argument("--num_generations", type=int, default=1, help="Number of generations to sample")
    return parser.parse_args()


def main(args):
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.modality in ['vision', 'vision_language']:
        output_filename = f"model_{args.model_name}_modality_{args.modality}_num_frames_{args.num_frames}_num_segments_{args.num_segments}_num_generations_{args.num_generations}.json"
    else:
        output_filename = f"model_{args.model_name}_modality_{args.modality}_num_generations_{args.num_generations}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Results will be saved to {output_path}")

    # Initialize prompt
    prompt = OEQAPrompt(modality=args.modality)

    # Initialize dataset
    dataset = SF20KSceneDataset(
        prompt=prompt,
        data_path=args.data_path,
        video_dir=args.video_dir,
        subtitles_path=args.subtitles_path,
        n_subsample=args.n_subsample,
        seed=args.seed,
        num_segments=args.num_segments,
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    # Initialize model
    print(f"Loading model {args.model_name}...")
    model = QwenVLModel(
        model_name=args.model_name,
        weights_dir=args.weights_dir,
        modality=args.modality,
        load_in_4bit=args.load_in_4bit
    )
    print("Model loaded successfully")

    # Generation loop
    results = {}
    # Check if output file exists and load existing results to resume
    if os.path.exists(output_path) and not args.force_rerun:
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results")
    
    existing_ids = set(results.keys())

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        question_id = sample["question_id"]
        
        if question_id in existing_ids and not args.force_rerun:
            continue

        responses, predictions = [], []
        for gen_nb in range(args.num_generations):
            response = model.generate(
                query=sample["query"],
                video_path=sample["video_path"],
                system_prompt=None,
                start_time=sample["start_time"],
                end_time=sample["end_time"],
                fps=args.fps,
                max_frames=int(args.num_frames / args.num_segments),
            )
            prediction = prompt.postprocess_response(response)
            responses.append(response)
            predictions.append(prediction)

        results[question_id] = {
            "question_id": question_id,
            "video_id": sample["video_id"],
            "question": sample["question"],
            "answer": sample["answer"], # Ground truth
            "responses": responses,
            "predictions": predictions,
            "model": args.model_name,
            "modality": args.modality,
            "num_frames": args.num_frames,
            "num_segments": args.num_segments,
            "num_generations": args.num_generations,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Saved results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
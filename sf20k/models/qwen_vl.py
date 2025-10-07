import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# 'pip install qwen-vl-utils'
from qwen_vl_utils import process_vision_info


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
            "Qwen/Qwen2.5-VL-72B-Instruct",
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
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_ignore_ Moors=True,
            )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            bnb_config=bnb_config,
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
        end_time: float = None,
        ground_truth: str = None,
        system_prompt: str = None,
    ):
        messages = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if end_time is not None:
            content.append({"type": "video", "video": video_path, "nframes": num_frames})
        else:
            content.append({"type": "video", "video": video_path, "nframes": num_frames, "start_time": start_time})
        
        content.append({"type": "text", "text": query})

        messages.append({"role": "user", "content": content})

        if ground_truth is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ground_truth}]})

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
        )

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
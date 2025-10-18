import os
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# 'pip install qwen-omni-utils''
from qwen_omni_utils import process_mm_info


class QwenOmniModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = ".",
        modality: str = "vision_language",
        num_frames: int = 8,
        fps: float = None,
        max_frames: int = None,
        load_in_4bit: bool = False,
    ):
        assert modality in [
            "vision",
            "language",
            "vision_language",
            "audio_vision",
            "audio_vision_language",
        ]

        assert model_name in [
            "qwen2.5-omni-3b",
            "qwen2.5-omni-7b",
        ]

        dict_model_name_to_model_id = {
            "qwen2.5-omni-3b": "Qwen/Qwen2.5-Omni-3B",
            "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
        }

        model_id = dict_model_name_to_model_id[model_name]

        if model_id in [
            "Qwen/Qwen2.5-Omni-3B",
            "Qwen/Qwen2.5-Omni-7B",
        ]:
            self.model_class_ = Qwen2_5OmniForConditionalGeneration
            self.processor_class_ = Qwen2_5OmniProcessor
        else:
            raise ValueError(f"Model {model_id} not supported")

        model_path = os.path.join(weights_dir, model_id)
        model, processor = self.load_model(model_path=model_path, load_in_4bit=load_in_4bit)
        model.disable_talker()

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.num_frames = num_frames
        self.fps = fps
        self.max_frames = max_frames
        self.modality = modality
        self.use_audio_in_video = modality in ["audio_vision", "audio_vision_language"]

    def load_model(self, model_path: str, load_in_4bit: bool = False):
        bnb_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit) if load_in_4bit else None
        model = self.model_class_.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            quantization_config=bnb_config,
        )
        processor = self.processor_class_.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=True,
        )
        return model, processor

    @staticmethod
    def format_chat_template(
        query: str,
        video_path: str,
        modality: str = "vision_language",
        start_time: float = None,
        end_time: float = None,
        ground_truth: str = None,
        system_prompt: str = None,
        num_frames: int = 8,
        fps: float = None,
        max_frames: int = None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        content = []

        if modality in ["vision", "vision_language", "audio_vision", "audio_vision_language"]:
            video_content = {
                "type": "video",
                "video": video_path,
            }
            if num_frames is not None:
                video_content["nframes"] = num_frames
            elif fps is not None:
                if max_frames is not None:
                    video_content["fps"] = fps
                    video_content["max_frames"] = max_frames
                else:
                    video_content["fps"] = fps
            if start_time is not None and end_time is not None:
                video_content["start_time"] = start_time
                video_content["end_time"] = end_time
            content.append(video_content)

        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})
        if ground_truth is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ground_truth}]})

        return messages

    def generate(
        self,
        query: str,
        video_path: str, 
        system_prompt: str = None,
        start_time: float = None,
        end_time: float = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        messages = self.format_chat_template(
            query=query,
            video_path=video_path,
            modality=self.modality,
            start_time=start_time,
            end_time=end_time,
            system_prompt=system_prompt,
            num_frames=self.num_frames,
            fps=self.fps,
            max_frames=self.max_frames,
        )

        text_inputs = [self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )]

        if modality in ["vision", "vision_language", "audio_vision", "audio_vision_language"]:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            audio_inputs = None
            image_inputs = None
            video_inputs = None

        inputs = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=self.use_audio_in_video,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False,
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
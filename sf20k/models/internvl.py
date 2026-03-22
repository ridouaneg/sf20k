import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from ..utils import load_video


class InternVLModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        modality: str = "vision_language",
        fps: float = 1.0,
        max_frames: int = 32,
        load_in_4bit: bool = False,
        **kwargs,
    ):
        dict_model_name_to_model_id = {
            "internvl3.5-1b": "OpenGVLab/InternVL3_5-1B-HF",
            "internvl3.5-2b": "OpenGVLab/InternVL3_5-2B-HF",
            "internvl3.5-4b": "OpenGVLab/InternVL3_5-4B-HF",
            "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B-HF",
            "internvl3.5-14b": "OpenGVLab/InternVL3_5-14B-HF",
        }
        model_id = dict_model_name_to_model_id[model_name]
        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if load_in_4bit else None

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        ).eval()

        self.modality = modality
        self.fps = fps
        self.max_frames = max_frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if self.modality in ["vision", "vision_language"]:
            content.append({"type": "video"})
        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        frames = load_video(video_path, desired_fps=self.fps, max_frames=self.max_frames, return_as="pil")

        inputs = self.processor(
            text=[text],
            videos=[frames],
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

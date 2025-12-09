import os
import sys

# Add vendor directory to path
vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vendor/LongVA'))
if vendor_path not in sys.path:
    sys.path.append(vendor_path)

try:
    from longva.model.builder import load_pretrained_model
    from longva.mm_utils import tokenizer_image_token, process_images
    from longva.constants import IMAGE_TOKEN_INDEX
except:
    load_pretrained_model = None
    tokenizer_image_token = None
    process_images = None
    IMAGE_TOKEN_INDEX = None

from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np

from ..utils import load_video


class LongVAModel:
    
    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        fps: float = 1.0,
        max_frames: int = 16,
        **kwargs,
    ):
        dict_model_name_to_model_id = {
             "longva-7b": "lmms-lab/LongVA-7B", 
             "longva-7b-dpo": "lmms-lab/LongVA-7B-DPO", 
        }

        self.model_name = model_name
        self.model_id = dict_model_name_to_model_id.get(model_name, model_name)
        
        model_path = os.path.join(weights_dir, self.model_id) if weights_dir is not None else self.model_id
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            "llava_qwen",
            device_map="auto",
        )

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.fps = fps
        self.max_frames = max_frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.5,
        top_p: float = None,
        num_beams: int = 1,
        use_cache: bool = True,
        **kwargs,
    ):
        gen_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "use_cache": use_cache,
            "max_new_tokens": max_new_tokens
        }

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        frames = load_video(
            video_path=video_path,
            desired_fps=self.fps,
            max_frames=self.max_frames,
            return_as="numpy",
        )
    
        video_tensor = self.image_processor.preprocess(
            frames,
            return_tensors="pt",
        )["pixel_values"].to(
            self.model.device,
            dtype=torch.float16,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                **gen_kwargs
            )

        response = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()

        return response
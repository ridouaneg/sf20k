import os
import sys
import numpy as np
from decord import cpu, VideoReader
import torch

# Add vendor directory to path
vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vendor/LongVU'))
if vendor_path not in sys.path:
    sys.path.append(vendor_path)

try:
    from longvu.builder import load_pretrained_model
    from longvu.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from longvu.conversation import conv_templates, SeparatorStyle
    from longvu.mm_datautils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )
except:
    load_pretrained_model = None
    DEFAULT_IMAGE_TOKEN = None
    IMAGE_TOKEN_INDEX = None
    conv_templates = None
    SeparatorStyle = None
    KeywordsStoppingCriteria = None
    process_images = None
    tokenizer_image_token = None

from ..constants import WEIGHTS_DIR


class LongVUModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = WEIGHTS_DIR,
        fps: float = 1.0,
        max_frames: int = 8,
        **kwargs,
    ):
        dict_model_name_to_model_id = {
            "longvu-3b": "Vision-CAIR/LongVU_Llama3_2_3B", 
            "longvu-7b": "Vision-CAIR/LongVU_Qwen2_7B", 
        }

        self.model_name = model_name
        self.model_id = dict_model_name_to_model_id.get(model_name, model_name)
        
        model_path = os.path.join(weights_dir, self.model_id)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            None,
            "cambrian_qwen",
            device_map="auto",
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.fps = fps
        self.max_frames = max_frames

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
        **kwargs,
    ):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())

        desired_num_frames = int(total_frames / video_fps * self.fps)
        num_frames = min(desired_num_frames, self.max_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        video = np.stack(video)

        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self.image_processor, self.model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + query
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response
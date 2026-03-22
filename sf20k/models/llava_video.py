from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import os

from ..utils import load_video

# ── Vendor LLaVA imports ──────────────────────────────────────────────────────
#_vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vendor/LLaVA'))
#if _vendor_path not in sys.path:
#    sys.path.insert(0, _vendor_path)

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    _llava_available = True
except Exception as e:
    load_pretrained_model = None
    _llava_available = False
    _llava_import_error = e

def load_video_dep(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time


class LlavaVideoModel:

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
        assert modality in ["vision", "language", "vision_language"]

        if not _llava_available:
            raise RuntimeError(
                f"LLaVA vendor package could not be imported: {_llava_import_error}"
            )

        dict_model_name_to_model_id = {
            "llava-video-7b":  "lmms-lab/LLaVA-Video-7B-Qwen2",
            "llava-video-72b": "lmms-lab/LLaVA-Video-72B-Qwen2",
        }
        assert model_name in dict_model_name_to_model_id, (
            f"Unknown model '{model_name}'. Choose from: {list(dict_model_name_to_model_id)}"
        )
        model_id = dict_model_name_to_model_id[model_name]
        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            "llava_qwen",
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="eager",
        )

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.modality = modality
        self.fps = fps
        self.max_frames = max_frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        **kwargs,
    ):
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])

        if self.modality in ["vision", "vision_language"]:
            frames = load_video(
                video_path,
                desired_fps=self.fps,
                max_frames=self.max_frames,
                return_as="numpy",
            )
            #video, frame_time, video_time = load_video(
            #    video_path=video_path,
            #    max_frames_num=self.max_frames,
            #    fps=1,
            #    force_sample=True,
            #)
            video_tensor = self.image_processor.preprocess(
                frames, return_tensors="pt"
            )["pixel_values"].to(self.model.device, dtype=torch.bfloat16)
            images = [video_tensor]
            modalities = ["video"]
            question = DEFAULT_IMAGE_TOKEN + "\n" + query
        else:
            images = None
            modalities = ["text"]
            question = query

        if system_prompt is not None:
            conv.system = system_prompt

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                modalities=modalities,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        # Strip the input tokens
        generated = output_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response

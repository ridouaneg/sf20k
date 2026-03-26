import os
import copy
import warnings
from decord import VideoReader, cpu
import numpy as np
import torch

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

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
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

        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path,
            None,
            "llava_qwen", 
            torch_dtype="bfloat16", 
            device_map="auto", 
            attn_implementation="eager",
        )
        model.eval()

        self.modality = modality
        self.fps = fps
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.model = model

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
        if self.modality not in ["vision", "vision_language"]:
            raise NotImplementedError
        
        video, frame_time, video_time = load_video(video_path, self.max_frames, 1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(self.model.device, self.model.dtype)
        video = [video]

        conv_template = "qwen_1_5"
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{query}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt",
        ).unsqueeze(0).to(
            self.model.device,
        )
        
        with torch.no_grad():
            cont = self.model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
        
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

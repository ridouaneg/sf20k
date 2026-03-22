import os
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )
except ImportError:
    Qwen2_5OmniForConditionalGeneration = None
    Qwen2_5OmniProcessor = None

# 'pip install qwen-vl-utils'
try:
    from qwen_omni_utils import process_mm_info
except:
    process_mm_info = None


class QwenOmniModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        modality: str = "vision_language",
        fps: float = 1.0,
        max_frames: int = 8,
        load_in_4bit: bool = False,
        **kwargs,
    ):
        assert modality in [
            "vision", 
            "language", 
            "audio",
            "vision_language",
            "audio_vision",
            "audio_language",
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
        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id
        model, processor = self.load_model(
            model_path=model_path,
            load_in_4bit=load_in_4bit,
        )

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.fps = fps
        self.max_frames = max_frames
        self.modality = modality
        self.use_audio_in_video = "audio" in self.modality
        
    def load_model(
        self, 
        model_path: str, 
        load_in_4bit: bool = False,
    ):
        #bnb_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit) if load_in_4bit else None

        # transformers requires torch >= 2.6 for torch.load (CVE-2025-32434) but
        # the speaker weights shipped with Qwen2.5-Omni are plain .pt files.
        # We patch the safety check locally so loading works on torch 2.5.x.
        import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
        _orig = getattr(_omni_mod, "check_torch_load_is_safe", None)
        if _orig is not None:
            _omni_mod.check_torch_load_is_safe = lambda: None
        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )
        finally:
            if _orig is not None:
                _omni_mod.check_torch_load_is_safe = _orig
        model.disable_talker()

        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

        return model, processor

    def generate(
        self,
        query: str,
        video_path: str, 
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        max_new_tokens: int = 256,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        out_seq_length: int = 16384,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
        **kwargs,
    ):
        messages = []
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

        content = []
        if "vision" in self.modality:
            content.append({"type": "video", "video": video_path})
        content.append({"type": "text", "text": query})

        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=self.use_audio_in_video
        )

        inputs = self.processor(
            text=[text],
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video
        ).to(
            self.model.device, 
            self.model.dtype
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                #presence_penalty=presence_penalty,
                #out_seq_length=out_seq_length,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
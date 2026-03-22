import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


class LlavaOneVisionModel:

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

        dict_model_name_to_model_id = {
            "llava-onevision-1.5-4b": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
            "llava-onevision-1.5-8b": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        }
        assert model_name in dict_model_name_to_model_id, (
            f"Unknown model '{model_name}'. Choose from: {list(dict_model_name_to_model_id)}"
        )
        model_id = dict_model_name_to_model_id[model_name]
        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )

        self.model = model
        self.processor = processor
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
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if self.modality in ["vision", "vision_language"]:
            content.append({
                "type": "video",
                "video": video_path,
                "fps": self.fps,
                "max_frames": self.max_frames,
            })
        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self.modality in ["vision", "vision_language"]:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        if "second_per_grid_ts" in inputs:
            del inputs["second_per_grid_ts"]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

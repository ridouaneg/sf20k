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

from sf20k.qwen_vl_utils import process_vision_info


class QwenVLModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        adapter_path: str = None,
        modality: str = "vision_language",
        fps: float = 1.0,
        max_frames: int = 8,
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
        self.fps = fps
        self.max_frames = max_frames
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
        modality: str = "vision_language",
        system_prompt: str = None,
        response: str = None,
        fps: float = 1.0,
        # min_frames: int = 4,
        max_frames: int = 8,
        video_start: float = 0.0,
        video_end: float = 100.0,
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
                "fps": fps,
                # "min_frames": min_frames,
                "max_frames": max_frames,
                # "sample_fps":sample_fps,
                "video_start": video_start,
                "video_end": video_end,
            })
        content.append({"type": "text", "text": query})

        messages.append({"role": "user", "content": content})
        if response is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        return messages

    def generate(
        self,
        query: str,
        video_path: str, 
        video_start: float,
        video_end: float,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
        n_generations: int = 1,
        **kwargs,
    ):
        messages = self.format_chat_template(
            query=query,
            video_path=video_path,
            modality=self.modality,
            system_prompt=system_prompt,
            response=None,
            fps=self.fps,
            max_frames=self.max_frames,
            video_start=video_start,
            video_end=video_end,
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

        responses = []
        for _ in range(n_generations):
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

            responses.append(response)

        return responses


def get_model(model_name: str, **kwargs):
    if model_name in [
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "qwen2.5-vl-32b",
        "qwen2.5-vl-72b",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
        "qwen3-vl-32b",
    ]:
        return QwenVLModel(model_name=model_name, **kwargs)
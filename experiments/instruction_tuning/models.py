import os
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils_v2 import process_vision_info

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


class QwenVLModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        adapter_path: str = None,
        load_in_4bit: bool = False,
        modality: str = "vision_language",
        fps: float = 1.0,
        max_frames: int = 8,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
        **kwargs,
    ):
        assert model_name in [
            "qwen3-vl-2b",
            "qwen3-vl-4b",
            "qwen3-vl-8b",
            "qwen3-vl-32b",
        ]

        dict_model_name_to_model_id = {
            "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
        }
        model_id = dict_model_name_to_model_id[model_name]

        self.model_class_ = Qwen3VLForConditionalGeneration
        self.processor_class_ = AutoProcessor

        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id
        model, processor = self.load_model(
            model_path=model_path, 
            load_in_4bit=load_in_4bit, 
            adapter_path=adapter_path,
        )

        self.model_name = model_name
        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.fps = fps
        self.max_frames = max_frames
        self.modality = modality
        self.total_pixels = total_pixels
        self.min_pixels = min_pixels
        self.response_template = "<|im_start|>assistant\n"
        self.response_token_ids = self.processor.tokenizer.encode(self.response_template, add_special_tokens=False)
        self.ignore_index = -100
        
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
            #device_map="auto",
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
        sample: dict,
        modality: str = "vision_language",
        sample_fps: float = 1.0,
        max_frames: int = 8,
        total_pixels: int = 20480 * 32 * 32,
        min_pixels: int = 64 * 32 * 32,
    ):
        video_path = sample['video_path']
        system_prompt = sample['system_prompt']
        query = sample['query']
        response = sample['response']
        
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
                "max_frames": max_frames,
                "sample_fps":sample_fps
            })
        content.append({"type": "text", "text": query})

        messages.append({"role": "user", "content": content})
        if response is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        return messages

    def generate(
        self,
        sample: dict,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        **kwargs,
    ):
        messages = self.format_chat_template(
            sample=sample,
            modality=self.modality,
            sample_fps=self.fps,
            max_frames=self.max_frames,
            total_pixels=self.total_pixels,
            min_pixels=self.min_pixels,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(
            self.model.device,
            self.model.dtype,
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
            )

        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return response

    def collate_fn(self, samples):
        messages_list = []
        text_inputs = []
        
        for sample in samples:
            messages = self.format_chat_template(
                sample=sample,
                modality=self.modality,
                sample_fps=self.fps,
                max_frames=self.max_frames,
                total_pixels=self.total_pixels,
                min_pixels=self.min_pixels,
            )

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            messages_list.append(messages)
            text_inputs.append(text)

        if self.modality in ["vision", "vision_language"]:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages_list,
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

        batch = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            #video_metadata=video_metadatas,
            **video_kwargs,
            return_tensors="pt",
            padding=True,
        )

        labels = batch['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index
        labels[labels == self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)] = self.ignore_index
        labels[labels == self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)] = self.ignore_index

        for i in range(len(labels)):
            response_start_idx = -1
            for idx in np.where(labels[i].cpu() == self.response_token_ids[0])[0]:
                if labels[i, idx : idx + len(self.response_token_ids)].tolist() == self.response_token_ids:
                    response_start_idx = idx
                    break
            
            if response_start_idx == -1:
                labels[i, :] = self.ignore_index
            else:
                response_end_idx = response_start_idx + len(self.response_token_ids)
                labels[i, :response_end_idx] = self.ignore_index

        batch["labels"] = labels
        return batch
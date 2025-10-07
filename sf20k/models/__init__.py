from .qwen_vl import QwenVLModel
from .gemini import GeminiModel
from .gpt import GPTModel

__all__ = ["QwenVLModel"]


def get_model(weights_dir, model_id, num_frames):
    if model_id in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]:
        return QwenVLModel(weights_dir, model_id, num_frames)
    elif model_id in ["gemini-2.5-flash", "gemini-2.5-pro"]:
        return GeminiModel(weights_dir, model_id, num_frames)
    elif model_id in ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]:
        return GPTModel(weights_dir, model_id, num_frames)
    else:
        raise ValueError(f"Model {model_id} not supported")
from .qwen_vl import QwenVLModel
from .qwen_omni import QwenOmniModel
from .gemini import Gemini
from .gpt import GPT


def get_model(model_name: str, **kwargs):
    if model_name in [
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "qwen2.5-vl-32b",
        "qwen2.5-vl-72b",
    ]:
        return QwenVLModel(model_name=model_name, **kwargs)
    elif model_name in [
        "qwen2.5-omni-3b",
        "qwen2.5-omni-7b",
    ]:
        return QwenOmniModel(model_name=model_name, **kwargs)
    elif model_name in [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]:
        return Gemini(model_name=model_name, **kwargs)
    elif model_name in [
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
    ]:
        return GPT(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported")
from .qwen_vl import QwenVLModel
from .qwen_omni import QwenOmniModel
from .gemini import GeminiModel
from .gpt import GPTModel
from .internvl import InternVLModel
from .longva import LongVAModel
from .longvu import LongVUModel


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
        #"qwen3-vl-2b-think",
        #"qwen3-vl-4b-think",
        #"qwen3-vl-8b-think",
        #"qwen3-vl-32b-think",
        #"qwen3-vl-30b-a3b",
        #"qwen3-vl-235b-a22b",
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
        "gemini-3-pro",
    ]:
        return GeminiModel(model_name=model_name, **kwargs)
    elif model_name in [
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
    ]:
        return GPTModel(model_name=model_name, **kwargs)
    elif model_name in [
        "internvl3.5-1b",
        "internvl3.5-2b",
        "internvl3.5-4b",
        "internvl3.5-8b",
        "internvl3.5-14b",
    ]:
        return InternVLModel(model_name=model_name, **kwargs)
    elif model_name in [
        "longva-7b", 
        "longva-7b-dpo",
    ]:
        return LongVAModel(model_name=model_name, **kwargs)
    elif model_name in [
        "longvu-3b", 
        "longvu-7b",
    ]:
        return LongVUModel(model_name=model_name, **kwargs)
    # baselines: video-llava, llava-video
    # reasoning: longvt, video-r1
    # memory: ma-lmm, moviechat
    # keyframe selection: tspo, tcot
    # others: videotree, llovi, langrepo, videoagent, video-salmonn-2+
    else:
        raise ValueError(f"Model {model_name} not supported")
from .qwen_vl import QwenVLModel
from .qwen_omni import QwenOmniModel
from .gemini import GeminiModel
from .gpt import GPTModel
from .claude import ClaudeModel
from .internvl import InternVLModel
from .llava_video import LlavaVideoModel
from .llava_onevision import LlavaOneVisionModel
from .longva import LongVAModel
from .longvu import LongVUModel
from .llovi import LLoViModel, LLoViCaptioner, LLoViCaptionsModel
from .lvagent import LVAgentModel


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
        "qwen3-vl-2b-think",
        "qwen3-vl-4b-think",
        "qwen3-vl-8b-think",
        "qwen3-vl-32b-think",
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
        "gemini-3-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro",
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
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ]:
        return ClaudeModel(model_name=model_name, **kwargs)
    elif model_name in [
        "internvl3.5-1b",
        "internvl3.5-2b",
        "internvl3.5-4b",
        "internvl3.5-8b",
        "internvl3.5-14b",
    ]:
        return InternVLModel(model_name=model_name, **kwargs)
    elif model_name in [
        "llava-video-7b",
        "llava-video-72b",
    ]:
        return LlavaVideoModel(model_name=model_name, **kwargs)
    elif model_name in [
        "llava-onevision-1.5-4b",
        "llava-onevision-1.5-8b",
    ]:
        return LlavaOneVisionModel(model_name=model_name, **kwargs)
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
    elif model_name in [
        "llovi-gpt4o-mini",
        "llovi-gpt4o",
        "llovi-llama3-8b",
        "llovi-llama3-1b",
    ]:
        return LLoViModel(model_name=model_name, **kwargs)
    elif model_name.startswith("llovi-captions+"):
        # e.g. "llovi-captions+gpt-4.1-mini" — step-2 only, requires captions_path kwarg
        llm_name = model_name[len("llovi-captions+"):]
        captions_path = kwargs.pop("captions_path")
        llm = get_model(llm_name, **kwargs)
        return LLoViCaptionsModel(llm=llm, captions_path=captions_path)
    elif model_name in [
        "lvagent-qwen2.5-3b",
        "lvagent-qwen2.5-7b",
        "lvagent-qwen2.5-3b+internvl3.5-1b",
        "lvagent-qwen2.5-7b+internvl3.5-8b",
        "lvagent-qwen2.5-7b+internvl3.5-8b+llava-video-7b",
    ]:
        return LVAgentModel(model_name=model_name, **kwargs)
    # baselines: video-llava
    # reasoning: longvt, video-r1
    # memory: ma-lmm, moviechat
    # keyframe selection: tspo, tcot
    # others: videotree, langrepo, videoagent, video-salmonn-2+
    else:
        raise ValueError(f"Model {model_name} not supported")
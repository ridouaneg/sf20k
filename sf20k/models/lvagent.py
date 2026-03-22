import os
import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import LlavaOnevisionForConditionalGeneration
except ImportError:
    LlavaOnevisionForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from ..utils import load_video
# Reuse InternVL's video-loading utilities
from .internvl import load_video as _internvl_load_video


# ── Agent implementations ─────────────────────────────────────────────────────

class _QwenAgent:
    """Qwen2.5-VL backbone agent."""

    def __init__(self, model_path: str, fps: float, max_frames: int):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.fps = fps
        self.max_frames = max_frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
    ) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": self.fps, "max_frames": self.max_frames},
            {"type": "text", "text": query},
        ]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to(self.model.device, self.model.dtype)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[0][inputs.input_ids.shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Text-only generation — used for the synthesis round."""
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(
            self.model.device, self.model.dtype
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[0][inputs.input_ids.shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


class _InternVLAgent:
    """InternVL3.5 backbone agent."""

    def __init__(self, model_path: str, num_frames: int):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attn=False,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.num_frames = num_frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
    ) -> str:
        pixel_values, num_patches_list = _internvl_load_video(
            video_path, num_segments=self.num_frames, max_num=1
        )
        pixel_values = pixel_values.to(self.model.device, self.model.dtype)
        video_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
        )
        question = video_prefix + query
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer, pixel_values, question, generation_config,
                num_patches_list=num_patches_list, history=None, return_history=False,
            )
        return response


class _LlavaVideoAgent:
    """LLaVA-Video backbone agent."""

    def __init__(self, model_path: str, fps: float, max_frames: int):
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.fps = fps
        self.max_frames = max_frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
    ) -> str:
        frames = load_video(video_path, desired_fps=self.fps, max_frames=self.max_frames, return_as="pil")
        conversation = []
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": query},
        ]})
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt, videos=[frames], return_tensors="pt"
        ).to(self.model.device, self.model.dtype)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


# ── LVAgent model ─────────────────────────────────────────────────────────────

class LVAgentModel:
    """
    LVAgent (https://github.com/64327069/LVAgent) — multi-agent video QA.

    Three model variants:
    • lvagent-qwen2.5-7b
        Single Qwen2.5-VL-7B agent (baseline).
    • lvagent-qwen2.5-7b+internvl3.5-2b
        Two agents: Qwen2.5-VL-7B + InternVL3.5-2B.
    • lvagent-qwen2.5-7b+internvl3.5-2b+llava-video-7b
        Three agents: Qwen2.5-VL-7B + InternVL3.5-2B + LLaVA-Video-7B.

    Discussion protocol (multi-agent variants):
        Round 1 — all agents answer independently from the video.
        Round 2 — primary agent (Qwen) receives all Round-1 answers and
                  produces a single refined response (text-only, no re-processing).
    """

    _configs = {
        "lvagent-qwen2.5-7b": {
            "qwen_id":      "Qwen/Qwen2.5-VL-7B-Instruct",
            "internvl_id":  None,
            "llava_id":     None,
        },
        "lvagent-qwen2.5-7b+internvl3.5-2b": {
            "qwen_id":      "Qwen/Qwen2.5-VL-7B-Instruct",
            "internvl_id":  "OpenGVLab/InternVL3_5-2B",
            "llava_id":     None,
        },
        "lvagent-qwen2.5-7b+internvl3.5-2b+llava-video-7b": {
            "qwen_id":      "Qwen/Qwen2.5-VL-7B-Instruct",
            "internvl_id":  "OpenGVLab/InternVL3_5-2B",
            "llava_id":     "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        },
    }

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        fps: float = 1.0,
        max_frames: int = 32,
        internvl_num_frames: int = 8,
        **kwargs,
    ):
        assert model_name in self._configs, (
            f"Unknown model '{model_name}'. Choose from: {list(self._configs)}"
        )
        cfg = self._configs[model_name]
        self.model_name = model_name

        def _path(model_id):
            return os.path.join(weights_dir, model_id) if weights_dir else model_id

        self.agent1 = _QwenAgent(_path(cfg["qwen_id"]), fps=fps, max_frames=max_frames)

        self.agent2 = (
            _InternVLAgent(_path(cfg["internvl_id"]), num_frames=internvl_num_frames)
            if cfg["internvl_id"] else None
        )

        self.agent3 = (
            _LlavaVideoAgent(_path(cfg["llava_id"]), fps=fps, max_frames=max_frames)
            if cfg["llava_id"] else None
        )

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        secondary_agents = [a for a in (self.agent2, self.agent3) if a is not None]

        # Single-agent path
        if not secondary_agents:
            return self.agent1.generate(
                query, video_path,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )

        # Round 1: all agents answer independently
        answers = [self.agent1.generate(
            query, video_path, system_prompt=system_prompt, max_new_tokens=max_new_tokens
        )]
        for agent in secondary_agents:
            answers.append(agent.generate(
                query, video_path, system_prompt=system_prompt, max_new_tokens=max_new_tokens
            ))

        # Round 2: primary agent synthesises all answers (text-only)
        experts = "\n".join(f"Expert {i+1} answered: {a}" for i, a in enumerate(answers))
        synthesis_prompt = (
            f"{len(answers)} experts were asked: \"{query}\"\n\n"
            f"{experts}\n\n"
            f"Considering all perspectives, provide a single, refined final answer."
        )
        return self.agent1.generate_text(synthesis_prompt, max_new_tokens=max_new_tokens)

import os
import sys
import json
import importlib.util
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm

try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    )
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    Qwen3VLForConditionalGeneration = None
    Qwen3VLMoeForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from ..utils import load_video

# LLoVi has a flat structure (model.py, prompts.py at root — not a package).
# We use importlib to avoid polluting sys.modules with generic names.
vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vendor/LLoVi'))


def _load_llovi_module(name):
    spec = importlib.util.spec_from_file_location(
        f"llovi_{name}",
        os.path.join(vendor_path, f"{name}.py"),
    )
    if spec is None:
        return None
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_llovi_llm(backend_type: str):
    """Lazily load GPT or LLaMA3 from the vendor module, surfacing real errors."""
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    if backend_type == "gpt":
        # model.py imports openai at the top — use importlib so the error is clear
        mod = _load_llovi_module("model")
        return mod.GPT
    elif backend_type == "llama3":
        # Import LLaMA3 directly; skip the top-level `import openai` in model.py
        # by importing only the class we need via exec after patching the namespace.
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("llovi_model", os.path.join(vendor_path, "model.py"))
        mod = _ilu.module_from_spec(spec)
        # Pre-populate openai with a stub so the top-level `import openai` doesn't crash
        import types
        if "openai" not in sys.modules:
            sys.modules["openai"] = types.ModuleType("openai")
        if "prompts" not in sys.modules:
            _p = _load_llovi_module("prompts")
            sys.modules["prompts"] = _p
        spec.loader.exec_module(mod)
        return mod.LLaMA3
    raise ValueError(f"Unknown backend_type: {backend_type}")


# ── Captioner ────────────────────────────────────────────────────────────────

class LLoViCaptioner:
    """
    Stage 1 of LLoVi: a small VLM that generates a text description for each
    sampled video frame. The concatenated captions are then fed to the LLM
    (Stage 2) as context for answering the question.
    """

    _model_ids = {
        # Qwen2.5-VL
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        # Qwen3-VL — Dense
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
        # Qwen3-VL — MoE
        "qwen3-vl-30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "qwen3-vl-235b-a22b": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    }

    _qwen3_moe_ids = {"qwen3-vl-30b-a3b", "qwen3-vl-235b-a22b"}

    CAPTION_PROMPT = "Briefly describe what is happening in this video frame."

    def __init__(self, model_name: str, weights_dir: str = None):
        assert model_name in self._model_ids, (
            f"Unknown captioner '{model_name}'. Choose from: {list(self._model_ids)}"
        )
        model_id = self._model_ids[model_name]
        model_path = os.path.join(weights_dir, model_id) if weights_dir else model_id

        if model_name in self._qwen3_moe_ids:
            model_class = Qwen3VLMoeForConditionalGeneration
        elif model_name.startswith("qwen3-vl"):
            model_class = Qwen3VLForConditionalGeneration
        else:
            model_class = Qwen2_5_VLForConditionalGeneration

        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def caption_frame(self, frame: Image.Image, max_new_tokens: int = 64) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": frame},
            {"type": "text", "text": self.CAPTION_PROMPT},
        ]}]
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
        return self.processor.decode(generated, skip_special_tokens=True).strip()
    
    def caption_video(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 16,
    ) -> str:
        """Sample frames and caption each one; return concatenated descriptions."""
        cap = cv2.VideoCapture(video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps_video if fps_video > 0 else 0
        cap.release()

        frames = load_video(video_path, desired_fps=fps, max_frames=max_frames, return_as="pil")
        num_sampled_frames = len(frames)
        time_step = duration / max(1, num_sampled_frames - 1) if num_sampled_frames > 1 else 0

        captions = []
        for i, frame in tqdm(enumerate(frames), total=num_sampled_frames, leave=True):
            t = i * time_step
            caption = self.caption_frame(frame)
            captions.append(f"[t={t:.1f}s] {caption}")
            
        return " ".join(captions)


# ── LLoVi model ───────────────────────────────────────────────────────────────

class LLoViModel:
    """
    LLoVi (https://github.com/CeeZh/LLoVi) — two-stage long-video QA pipeline.

    Stage 1 — Visual captioning: a small VLM (LLoViCaptioner) describes each
    sampled frame in natural language. This runs inline when captioner_name is
    set, or is skipped in favour of pre-computed captions loaded from
    captions_path.

    Stage 2 — LLM reasoning: a text-only LLM (GPT-4o / LLaMA-3-8B) receives
    the captions + question and produces the final answer.
    """

    _llm_backends = {
        "llovi-gpt4o-mini": ("gpt",    "gpt-4o-mini"),
        "llovi-gpt4o":      ("gpt",    "gpt-4o"),
        "llovi-gpt5-mini":  ("gpt",    "gpt-5-mini"),
        "llovi-gpt5":       ("gpt",    "gpt-5"),
        "llovi-llama3-8b":  ("llama3", "meta-llama/Llama-3.1-8B-Instruct"),
        "llovi-llama3-1b":  ("llama3", "meta-llama/Llama-3.2-1B-Instruct"),
    }

    def __init__(
        self,
        model_name: str,
        # Stage 1 — captioner
        captioner_name: str = "qwen3-vl-2b",
        captioner_fps: float = 1.0,
        captioner_max_frames: int = 16,
        # Stage 1 — captions cache (persisted across runs)
        captions_cache_path: str = None,
        # Stage 1 — pre-computed captions fallback (read-only)
        captions_path: str = None,
        # Stage 2 — LLM
        weights_dir: str = None,
        api_key: str = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        assert model_name in self._llm_backends, (
            f"Unknown model '{model_name}'. Choose from: {list(self._llm_backends)}"
        )
        backend_type, backend_id = self._llm_backends[model_name]

        self.model_name = model_name
        self.captioner_fps = captioner_fps
        self.captioner_max_frames = captioner_max_frames

        # Stage 1: inline captioner
        if captioner_name is not None:
            self.captioner = LLoViCaptioner(
                model_name=captioner_name,
                weights_dir=weights_dir,
            )
        else:
            self.captioner = None

        # Stage 1: caption cache — keyed by (video_path, fps, max_frames).
        # Defaults to data/captions/llovi_captions.json at the project root.
        # Loaded from disk on startup and written back after each new captioning.
        if captions_cache_path is None:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))
            captions_cache_path = os.path.join(_project_root, "data", "captions", "llovi_captions.json")
        self.captions_cache_path = captions_cache_path
        self.captions_cache: dict = {}
        if os.path.exists(captions_cache_path):
            with open(captions_cache_path) as f:
                self.captions_cache = json.load(f)

        # Stage 1: pre-computed captions fallback (read-only)
        self.captions: dict = {}
        if captions_path is not None:
            with open(captions_path) as f:
                self.captions = json.load(f)

        # Stage 2: LLM
        if backend_type == "gpt":
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            self.llm = _load_llovi_llm("gpt")(api_key, backend_id, temperature)
        elif backend_type == "llama3":
            model_path = os.path.join(weights_dir, backend_id) if weights_dir else backend_id
            self.llm = _load_llovi_llm("llama3")(model_path, temperature, max_new_tokens)

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        **kwargs,
    ) -> str:
        # Stage 1: get captions
        if self.captioner is not None:
            cache_key = f"{video_path}|fps={self.captioner_fps}|max_frames={self.captioner_max_frames}"
            if cache_key in self.captions_cache:
                captions = self.captions_cache[cache_key]
            else:
                captions = self.captioner.caption_video(
                    video_path,
                    fps=self.captioner_fps,
                    max_frames=self.captioner_max_frames,
                )
                self.captions_cache[cache_key] = captions
                if self.captions_cache_path is not None:
                    with open(self.captions_cache_path, "w") as f:
                        json.dump(self.captions_cache, f)
        else:
            captions = self.captions.get(video_path) or self.captions.get(
                os.path.splitext(os.path.basename(video_path))[0], ""
            )
            if isinstance(captions, list):
                captions = " ".join(captions)

        # Stage 2: LLM reasoning
        prompt = (
            f"Here are descriptions of a video:\n{captions}\n\n"
            f"Answer the following question based on the descriptions:\n{query}"
        )
        head = system_prompt or "You are a helpful expert in video analysis."
        response, _ = self.llm.forward(head=head, prompts=[prompt])
        return response


# ── LLoVi stage-2 only (bring your own LLM) ──────────────────────────────────

class LLoViCaptionsModel:
    """
    Stage 2 of LLoVi using pre-computed captions and *any* text LLM.

    Usage::

        # Step 1 — caption once (offline)
        captioner = LLoViCaptioner("qwen2.5-vl-3b", weights_dir=...)
        captioner.caption_video("vid1.mp4")

        # Step 2 — QA with any model
        from sf20k.models import get_model
        llm = get_model("gpt-4.1-mini", ...)
        model = LLoViCaptionsModel(llm=llm, captions_path="captions.json")
        answer = model.generate(query, video_path)

    The ``llm`` can be any object with a ``generate(query, video_path, **kwargs)``
    interface — i.e. anything returned by ``get_model()``.  For text-only LLMs
    the captions are folded into the query and ``video_path=None`` is passed.
    """

    _PROMPT_TEMPLATE = (
        "Here are descriptions of a video:\n{captions}\n\n"
        "Answer the following question based on the descriptions:\n{query}"
    )

    def __init__(self, llm, captions_path: str):
        self.llm = llm
        with open(captions_path) as f:
            self.captions: dict = json.load(f)

    def _lookup_captions(self, video_path: str) -> str:
        captions = self.captions.get(video_path) or self.captions.get(
            os.path.splitext(os.path.basename(video_path))[0], ""
        )
        if isinstance(captions, list):
            captions = " ".join(captions)
        return captions

    def generate(self, query: str, video_path: str, **kwargs) -> str:
        captions = self._lookup_captions(video_path)
        prompt = self._PROMPT_TEMPLATE.format(captions=captions, query=query)
        return self.llm.generate(prompt, video_path=None, **kwargs)

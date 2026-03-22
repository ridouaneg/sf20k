import base64
import numpy as np
import cv2
import os

try:
    import anthropic
except ImportError:
    anthropic = None

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", None)


class ClaudeModel:

    def __init__(
        self,
        model_name: str,
        modality: str = "vision_language",
        fps: float = 1.0,
        max_frames: int = 8,
        target_size: tuple = None,
        **kwargs,
    ):
        assert model_name in [
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ]

        assert modality in [
            "vision",
            "language",
            "vision_language",
        ]

        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        self.modality = modality
        self.fps = fps
        self.max_frames = max_frames
        self.target_size = target_size

    def load_video(
        self,
        video_path: str,
        fps: float = None,
        max_frames: int = None,
        target_size: tuple = None,
    ):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return []

        video_bytes = []
        try:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0 or video_fps <= 0:
                print("Warning: Video has no frames or FPS is not available.")
                return []

            num_frames_at_fps = int(round(total_frames / video_fps * (fps or 1.0)))
            num_frames = min(max_frames, num_frames_at_fps) if max_frames else num_frames_at_fps

            if num_frames <= 0:
                return []

            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
            for idx in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = video.read()
                if not success:
                    continue

                if target_size is not None:
                    frame = cv2.resize(frame, target_size)
                _, buffer = cv2.imencode(".jpg", frame)
                video_bytes.append(base64.b64encode(buffer).decode("utf-8"))
        finally:
            video.release()

        return video_bytes

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        start_time: float = None,
        end_time: float = None,
        **kwargs,
    ):
        if start_time is not None or end_time is not None:
            raise NotImplementedError("Start time and end time are not supported for Claude")
        if do_sample:
            raise NotImplementedError("Do sample is not supported for Claude")

        content = []

        if "vision" in self.modality:
            video = self.load_video(
                video_path,
                fps=self.fps,
                max_frames=self.max_frames,
                target_size=self.target_size,
            )
            for frame in video:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame,
                    },
                })

        content.append({
            "type": "text",
            "text": query,
        })

        messages = [{"role": "user", "content": content}]

        kwargs = {
            "model": self.model_name,
            "max_tokens": max_new_tokens,
            "messages": messages,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        try:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            print(f"Error getting response: {e}")
            return None

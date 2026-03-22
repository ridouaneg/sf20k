import os
import numpy as np
import cv2

try:
    from google import genai
    from google.genai import types
except:
    genai = None
    types = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)


class GeminiModel:

    def __init__(
        self,
        model_name: str,
        modality: str = "vision_language",
        fps: float = 1.0,
        max_frames: int = None,
        target_size: tuple = None,
        **kwargs,
    ):
        assert model_name in [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro",
        ]

        assert modality in [
            "vision",
            "language",
            "vision_language",
        ]

        self.model_name = model_name
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.modality = modality
        self.fps = fps
        self.max_frames = max_frames
        self.target_size = target_size

    def load_video(self, video_path: str):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return []

        frames = []
        try:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = video.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0 or video_fps <= 0:
                print("Warning: Video has no frames or FPS is not available.")
                return []

            num_frames = int(round(total_frames / video_fps * self.fps))
            if self.max_frames is not None:
                num_frames = min(self.max_frames, num_frames)

            if num_frames <= 0:
                return []

            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
            for idx in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = video.read()
                if not success:
                    continue
                if self.target_size is not None:
                    frame = cv2.resize(frame, self.target_size)
                _, buffer = cv2.imencode(".jpg", frame)
                frames.append(buffer.tobytes())
        finally:
            video.release()

        return frames

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = None,
        do_sample: bool = False,
        temperature: float = None,
        start_time: float = None,
        end_time: float = None,
        **kwargs,
    ):
        if start_time is not None or end_time is not None:
            raise NotImplementedError("Start time and end time are not supported for Gemini")
        if do_sample:
            raise NotImplementedError("Do sample is not supported for Gemini")
        if system_prompt is not None:
            raise NotImplementedError("System prompt is not supported for Gemini")

        content = []
        if "vision" in self.modality:
            for frame_bytes in self.load_video(video_path):
                content.append(
                    types.Part(
                        inline_data=types.Blob(data=frame_bytes, mime_type="image/jpeg")
                    )
                )

        content.append(types.Part(text=query))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=types.Content(parts=content),
        )

        return response.text

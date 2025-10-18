from google import genai
from google.genai import types
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Gemini:

    def __init__(
        self,
        model_id: str,
        weights_dir: str = None,
        num_frames: int = None,
        fps: float = 1.0,
        max_frames: int = None,
        target_size: tuple = None,
    ):
        if fps is None:
            raise NotImplementedError("FPS is required for Gemini")
        if max_frames is not None:
            raise NotImplementedError("Max frames is not supported for Gemini")
        if target_size is not None:
            raise NotImplementedError("Target size is not supported for Gemini")
        
        self.model_id = model_id
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.fps = fps
    
    def load_video(self, video_path: str):
        return open(video_path, "rb").read()

    def generate(
        self,
        query: str,
        video_path: str, 
        system_prompt: str = None,
        start_time: float = None,
        end_time: float = None,
        max_new_tokens: int = None,
        do_sample: bool = None,
        temperature: float = None,
    ):
        if start_time is not None or end_time is not None:
            raise NotImplementedError("Start time and end time are not supported for Gemini")
        if do_sample:
            raise NotImplementedError("Do sample is not supported for Gemini")
        if system_prompt is not None:
            raise NotImplementedError("System prompt is not supported for Gemini")

        video = self.load_video(video_path)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video, mime_type='video/mp4'), video_metadata=types.VideoMetadata(fps=self.fps)),
                    types.Part(text=query)
                ]
            ),
        )

        return response.text
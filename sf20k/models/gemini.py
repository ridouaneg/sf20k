import os

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
            "gemini-3-pro",
        ]

        assert modality in [
            "vision",
            "language",
            "vision_language",
        ]

        if fps is None:
            raise NotImplementedError("FPS is required for Gemini")
        if max_frames is not None:
            raise NotImplementedError("Max frames is not supported for Gemini")
        if target_size is not None:
            raise NotImplementedError("Target size is not supported for Gemini")
        
        self.model_name = model_name
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.modality = modality
        self.fps = fps
    
    def load_video(self, video_path: str):
        return open(video_path, "rb").read()

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
            video = self.load_video(video_path)
            content.append(
                types.Part(
                    inline_data=types.Blob(data=video, mime_type='video/mp4'), 
                    video_metadata=types.VideoMetadata(fps=self.fps)
                )
            )
        
        content.append(
            types.Part(text=query)
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=types.Content(
                parts=content
            ),
        )

        return response.text
import base64
import numpy as np
import cv2
import os

try:
    from openai import OpenAI
except:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", None)


class GPTModel:
    
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
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
            "gpt-5.1",
        ]

        assert modality in [
            "vision",
            "language",
            "vision_language",
        ]
        
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID)
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
            fps = video.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0 or fps <= 0:
                print("Warning: Video has no frames or FPS is not available.")
                return []

            if fps > 0.0:
                num_frames_1fps = int(round(total_frames / fps))
                num_frames = min(max_frames, num_frames_1fps)

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
            raise NotImplementedError("Start time and end time are not supported for GPT")
        if do_sample:
            raise NotImplementedError("Do sample is not supported for GPT")
        if system_prompt is not None:
            raise NotImplementedError("System prompt is not supported for GPT")

        content = [
            {
                "type": "input_text",
                "text": query,
            }
        ]
        
        if "vision" in self.modality:
            video = self.load_video(
                video_path,
                fps=self.fps,
                max_frames=self.max_frames,
                target_size=self.target_size,
            )

            content.extend([
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{frame}"
                }
                for frame in video
            ])
        
        messages = [{
            "role": "user",
            "content": content
        }]

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages
            )
            return response.output_text
        except Exception as e:
            print(f"Error getting response: {e}")
            return None
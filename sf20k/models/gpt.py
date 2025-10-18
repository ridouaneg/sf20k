import cv2
import base64
import numpy as np
from openai import OpenAI
#import os

from sf20k.constants import OPENAI_API_KEY, OPENAI_ORG_ID

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")


class GPT:
    
    def __init__(
        self,
        model_id: str,
        weights_dir: str,
        num_frames: int = 8,
        fps: float = None,
        max_frames: int = None,
        target_size: tuple = None,
    ):
        self.model_id = model_id
        self.client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID)
        self.num_frames = num_frames
        self.fps = fps
        self.max_frames = max_frames
        self.target_size = target_size
    
    def load_video(
        self,
        video_path: str,
        num_frames: int = None,
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
        start_time: float = None,
        end_time: float = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        if start_time is not None or end_time is not None:
            raise NotImplementedError("Start time and end time are not supported for GPT")
        if do_sample:
            raise NotImplementedError("Do sample is not supported for GPT")
        if system_prompt is not None:
            raise NotImplementedError("System prompt is not supported for GPT")

        video = self.load_video(
            video_path,
            num_frames=self.num_frames,
            fps=self.fps,
            max_frames=self.max_frames,
            target_size=self.target_size,
        )

        try:
            response = self.client.responses.create(
                model=self.model_id,
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": query,
                        },
                        *[
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{frame}"
                            }
                            for frame in video
                        ]
                    ]
                }],
            )
            return response.output_text
        except Exception as e:
            print(f"Error getting response: {e}")
            return None
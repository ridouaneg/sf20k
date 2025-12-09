import random
import numpy as np
import torch
from PIL import Image
import decord
import cv2


def set_seed(seed: int = 42):
    """Sets the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_video_cv2(
    video_path: str,
    start: float = None,
    end: float = None,
    desired_fps: float = 1.0,
    max_frames: int = 8,
    target_size: tuple = (640, 360),
    return_as: str = 'pil',
) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)

    #if not cap.isOpened():
    #    print(f"Error: Could not open video file at {video_path}")
    #    return [Image.fromarray(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))] * num_frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start * video_fps) if start is not None else 0
    end_frame = int(end * video_fps) if end is not None else total_frames - 1
    
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    #if start_frame >= end_frame:
    #    print("Error: Start time is after or same as end time, or time range is invalid.")
    #    return [Image.fromarray(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))] * num_frames

    desired_num_frames = int((end_frame - start_frame) / video_fps * desired_fps)
    num_frames = min(desired_num_frames, max_frames)

    frame_range = end_frame - start_frame
    if num_frames > frame_range:
        print(f"Warning: The requested number of frames ({num_frames}) is greater than the available frames in the range ({frame_range}). Loading all available frames.")
        indices_to_read = list(range(start_frame, end_frame + 1))
    else:
        indices_to_read = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames = []
    for i in indices_to_read:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at index {i}.")
            pil_image = Image.fromarray(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.resize(target_size)
        frames.append(pil_image)
    cap.release()

    if return_as == 'numpy':
        return np.array(frames)
    
    return frames


def load_video_decord(
    video_path: str,
    start: float = None,
    end: float = None,
    desired_fps: float = 1.0,
    max_frames: int = 8,
    target_size: tuple = (640, 360),
    return_as: str = 'pil',
):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))

    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    start_frame = int(start * video_fps) if start is not None else 0
    end_frame = int(end * video_fps) if end is not None else total_frames - 1

    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    desired_num_frames = int((end_frame - start_frame) / video_fps * desired_fps)
    num_frames = min(desired_num_frames, max_frames)

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()

    if return_as == 'pil':
        frames = [Image.fromarray(frame).resize(target_size) for frame in frames]

    return frames


def load_video(
    video_path: str,
    start: float = None,
    end: float = None,
    desired_fps: float = 1.0,
    max_frames: int = 8,
    target_size: tuple = (640, 360),
    return_as: str = 'pil',
    #backend: str = 'decord',
) -> list[Image.Image]:
    try:
        return load_video_decord(
            video_path=video_path,
            start=start,
            end=end,
            desired_fps=desired_fps,
            max_frames=max_frames,
            target_size=target_size,
            return_as=return_as,
        )
    except:
        return load_video_cv2(
            video_path=video_path,
            start=start,
            end=end,
            desired_fps=desired_fps,
            max_frames=max_frames,
            target_size=target_size,
            return_as=return_as,
        )
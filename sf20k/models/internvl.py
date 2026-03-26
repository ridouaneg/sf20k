import os
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

class InternVLModel:

    def __init__(
        self,
        model_name: str,
        weights_dir: str = None,
        modality: str = "vision_language",
        num_frames: int = 8,
        fps: float = None,
        max_frames: int = None,
        load_in_4bit: bool = False,
        target_size: tuple = None,
        **kwargs,
    ):
        assert modality in ["vision", "language", "vision_language"]

        dict_model_name_to_model_id = {
            "internvl3.5-1b": "OpenGVLab/InternVL3_5-1B",
            "internvl3.5-2b": "OpenGVLab/InternVL3_5-2B",
            "internvl3.5-4b": "OpenGVLab/InternVL3_5-4B",
            "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B",
            "internvl3.5-14b": "OpenGVLab/InternVL3_5-14B",
            "internvl3.5-38b": "OpenGVLab/InternVL3_5-38B",
        }
        model_id = dict_model_name_to_model_id[model_name]
        model_path = os.path.join(weights_dir, model_id) if weights_dir is not None else model_id

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            #load_in_4bit=load_in_4bit,
            #load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.fps = fps
        self.max_frames = max_frames
        self.modality = modality

    def generate(
        self,
        query: str,
        video_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        #generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        if self.modality in ["vision", "vision_language"]:
            pixel_values, num_patches_list = load_video(video_path, num_segments=self.max_frames, max_num=1)
            #pixel_values = pixel_values.to(torch.bfloat16).cuda()
            pixel_values = pixel_values.to(self.model.device, self.model.dtype)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + query
        else:
            pixel_values = None
            question = query
            
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        
        return response
    

if __name__ == "__main__":
    import argparse
    from sf20k.datasets.sf20k import SF20KDataset
    from sf20k.prompts import OEQAPrompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data/test_expert.csv")
    parser.add_argument("--subtitles_path", type=str, default="../../data/test_subtitles.csv")
    parser.add_argument("--video_dir", type=str, default="/geovic/geovic/SF20K/videos/")
    parser.add_argument("--model_name", type=str, default="internvl3.5-1b")
    parser.add_argument("--weights_dir", type=str, default="/geovic/ghermi/weights/")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    prompt = OEQAPrompt(modality="vision_language")
    dataset = SF20KDataset(
        prompt=prompt,
        data_path=args.data_path,
        video_dir=args.video_dir,
        subtitles_path=args.subtitles_path,
        n_subsample=1,
        seed=42,
    )
    sample = dataset[0]
    print(f"Question ID : {sample['question_id']}")
    print(f"Video ID    : {sample['video_id']}")
    print(f"Question    : {sample['question']}")
    print(f"Ground truth: {sample['answer']}")
    print(f"Query       :\n{sample['query']}\n")

    model = InternVLModel(
        model_name=args.model_name,
        weights_dir=args.weights_dir,
        modality="vision_language",
        max_frames=args.num_frames,
    )

    response = model.generate(
        query=sample["query"],
        video_path=sample["video_path"],
    )
    prediction = prompt.postprocess_response(response)
    print(f"Response    : {response}")
    print(f"Prediction  : {prediction}")

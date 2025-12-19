from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    # Paths
    project_name: str = "sf20k"
    run_name: str = "test"
    output_dir: str = "./results/"
    weights_dir: str = "/geovic/ghermi/weights/"
    train_data_path: str = "/users/ghermi/code/sf20k/data/train.csv"
    test_data_path: str = "/users/ghermi/code/sf20k/data/train.csv"
    video_dir: str = "/geovic/geovic/SF20K/videos/"
    train_subtitles_path: str = "/geovic/geovic/SF20K/subtitles.parquet"
    test_subtitles_path: str = "/users/ghermi/code/sf20k/data/test_subtitles.csv"

    # Dataset Parameters
    n_subsample_train: int = 64
    n_subsample_test: int = 8

    # Model Parameters
    model_name: str = "qwen3-vl-2b"
    adapter_path: str = None
    load_in_4bit: bool = False
    modality: str = "vision_language"
    fps: float = 1.0
    max_frames: int = 8

    # LoRA Configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training Arguments
    num_train_epochs: int = 1
    batch_size: int = 8
    optim: str = "adamw_torch_fused"
    learning_rate: float = 0.001
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    max_length: Optional[int] = None
    max_completion_length: int = 1024
    max_prompt_length: int = 2048
    
    generation_log_steps: int = 64
    num_gens_to_log: int = 5
    logging_steps: int = 1
    eval_steps: int = 64
    save_steps: int = 64
    seed: int = 42
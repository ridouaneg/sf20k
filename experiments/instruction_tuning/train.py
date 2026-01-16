import os
import argparse
from transformers import TrainerCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import wandb
import yaml
import random
import torch.distributed as dist
from datetime import datetime

from sf20k.utils import convert_to_hf_dataset, set_seed, load_config, get_num_gpus
from sf20k.prompts import OEQAPrompt

from sf20k_dataset import SF20KDataset
from models import QwenVLModel
from callbacks import GenerationCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a QwenVL model with SFT.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/test.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main(args):
    # Prepare env
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    config = load_config(args.config)
    set_seed(config.seed)

    # Prepare model
    module = QwenVLModel(
        model_name=config.model_name,
        weights_dir=config.weights_dir,
        load_in_4bit=config.load_in_4bit,
        modality=config.modality,
        fps=config.fps,
        max_frames=config.max_frames,
        device_map=None,
    )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    print(f"Model {config.model_name} loaded.")

    # Prepare data
    prompt = OEQAPrompt()

    train_dataset = SF20KDataset(
        prompt=prompt,
        data_path=config.train_data_path,
        video_dir=config.video_dir,
        subtitles_path=config.train_subtitles_path,
        n_subsample=config.n_subsample_train,
        seed=config.seed,
    )

    test_dataset = SF20KDataset(
        prompt=prompt,
        data_path=config.test_data_path,
        video_dir=config.video_dir,
        #subtitles_path=config.test_subtitles_path,
        subtitles_path=config.train_subtitles_path,
        n_subsample=config.n_subsample_test,
        seed=config.seed,
    )

    train_dataset = convert_to_hf_dataset(train_dataset)
    test_dataset = convert_to_hf_dataset(test_dataset)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Prepare trainer
    if args.output_dir is not None:
        output_folder = args.output_dir
    else:
        date = datetime.now().strftime("%H%M%S-%d%m%Y")
        output_folder = os.path.join(config.output_dir, f"{config.run_name}_{date}")
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    config_path = os.path.join(output_folder, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    num_gpus = get_num_gpus()
    gradient_accumulation_steps = config.batch_size // (config.per_device_train_batch_size * num_gpus)

    training_args = SFTConfig(
        run_name=config.run_name,
        output_dir=output_folder,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=config.max_length,
        optim=config.optim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=config.save_steps,
        bf16=True,
        tf32=True,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        remove_unused_columns=False, # Important for custom data collators
    )

    if training_args.process_index == 0:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config,
        )

    generation_callback = GenerationCallback(
        module=module,
        log_steps=config.generation_log_steps,
        eval_dataset=test_dataset,
        num_generations=config.num_gens_to_log,
    )

    trainer = SFTTrainer(
        model=module.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        data_collator=module.collate_fn,
        callbacks=[generation_callback],
    )

    if training_args.process_index == 0:
        trainer.model.print_trainable_parameters()
    
    # Train
    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    # Save
    if training_args.process_index == 0:
        final_checkpoint_dir = os.path.join(output_folder, "checkpoint-final")
        trainer.save_model(final_checkpoint_dir)
        print(f"Final model saved to {final_checkpoint_dir}")
        wandb.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
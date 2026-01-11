import random
from transformers import TrainerCallback
import torch


class GenerationCallback(TrainerCallback):

    def __init__(
        self,
        module,
        log_steps,
        eval_dataset,
        num_generations: int = 5,
        max_new_tokens: int = 1024,
    ):
        self.module = module
        self.log_steps = log_steps
        self.eval_dataset = eval_dataset.select(random.sample(range(len(eval_dataset)), num_generations))
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def on_step_end(self, args, state, control, **kwargs):
        gts = []
        preds = []

        self.module.model.eval()
        if state.is_world_process_zero and state.global_step > 0 and state.global_step % self.log_steps == 0:
            print(f"\n--- Logging Generations at Step {state.global_step} ---")
            for i, sample in enumerate(self.eval_dataset):
                gt = sample['response']
                pred = self.module.generate(
                    sample=sample,
                    max_new_tokens=self.max_new_tokens,
                )

                print(f"Ground truth: {gt}")
                print(f"Model prediction: {pred}")
                print('-' * 40)

                gts.append(gt)
                preds.append(pred)

            self.module.model.train()
            print("--- Generations logged to WandB ---")
            torch.cuda.empty_cache()
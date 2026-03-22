import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Gemma3ForCausalLM,
)

try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    import anthropic
except:
    anthropic = None


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", None)
ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY", None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/movieqa.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="gpt-5-nano", choices=[
        # OpenAI
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5.4",
        # Qwen
        "qwen3-0.6b",
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
        # Gemma
        "gemma-3-270m",
        "gemma-3-1b",
        "gemma-3-4b",
        "gemma-3-12b",
        "gemma-3-27b",
        # Llama
        "llama-3.2-1b",
        "llama-3.2-3b",
        "llama-3.1-8b",
        "llama-3.1-70b",
        # Mistral
        "mistral-7b",
        "mistral-small",
        # SmolLM2
        "smollm2-135m",
        "smollm2-360m",
        "smollm2-1.7b",
        # SmolLM3
        "smollm3-3b",
        # OLMo2
        "olmo2-7b",
        "olmo2-13b",
        # Claude
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ])
    parser.add_argument("--weights_dir", type=str, default="/geovic/ghermi/weights",
                        help="Directory containing HuggingFace model weights")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--no_title", action="store_true", help="Omit movie title from the prompt")
    parser.add_argument("--mcqa", action="store_true", help="Use multiple-choice QA format (only for movieqa, tvqa, cinepile)")
    return parser.parse_args()


MCQA_DATASETS = {"movieqa", "tvqa", "cinepile"}
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


class Prompt:

    def __init__(self, no_title: bool = False):
        self.no_title = no_title
        if no_title:
            self.template = "Here is a question about a movie: {question}\nAnswer shortly and directly without repeating the question. If you don't know the movie, try to guess the answer."
        else:
            self.template = "In the movie '{movie_title}', {question}\nAnswer shortly and directly without repeating the question. If you don't know the movie, try to guess the answer."

    def get_query(self, sample):
        question = sample['question']
        question = question[0].lower() + question[1:] # remove first upper case
        if self.no_title:
            return self.template.format(question=question)
        return self.template.format(
            movie_title=sample["movie_title"],
            question=question,
        )

    def postprocess_response(self, response):
        return response.strip() if response is not None else None


class MCQAPrompt:

    def __init__(self, no_title: bool = False):
        self.no_title = no_title
        if no_title:
            self.template = "Here is a multiple-choice question about a movie: {question}\n{choices}\nAnswer with only the letter of the correct option (e.g. A). If you don't know the movie, try to guess the answer."
        else:
            self.template = "In the movie '{movie_title}', {question}\n{choices}\nAnswer with only the letter of the correct option (e.g. A). If you don't know the movie, try to guess the answer."

    def get_query(self, sample):
        import ast
        question = sample['question']
        question = question[0].lower() + question[1:]
        options = sample['options']
        if isinstance(options, str):
            options = ast.literal_eval(options)
        choices = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        if self.no_title:
            return self.template.format(question=question, choices=choices)
        return self.template.format(
            movie_title=sample["movie_title"],
            question=question,
            choices=choices,
        )

    def postprocess_response(self, response):
        if response is None:
            return None
        response = response.strip()
        # Extract the first A-E letter found in the response
        for ch in response:
            if ch in OPTION_LETTERS:
                return ch
        return response


class Dataset:

    def __init__(self, input_path: str, prompt, n_subsample: int = -1):
        self.df = pd.read_csv(input_path)
        self.prompt = prompt

        assert "question_id" in self.df.columns
        assert "movie_title" in self.df.columns
        assert "question" in self.df.columns
        assert "answer" in self.df.columns

        if n_subsample > 0:
            self.df = self.df.sample(n=n_subsample, random_state=42)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        sample['query'] = self.prompt.get_query(sample)
        return sample


class GPTModel:
    
    def __init__(
        self,
        model_name: str,
    ):
        assert model_name in [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
            "gpt-5.4-nano",
            "gpt-5.4-mini",
            "gpt-5.4",
        ]
        
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID)

    def generate(
        self,
        query: str,
        system_prompt: str = None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": [{"type": "input_text", "text": query}]})

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages
            )
            return response.output_text
        except Exception as e:
            print(f"Error getting response: {e}")
            return None


class QwenModel:

    def __init__(
        self,
        model_name: str = "qwen3-1.7b",
        weights_dir: str = "/geovic/ghermi/weights",
    ):
        name_to_id = {
            'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
            'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
            'qwen3-4b': 'Qwen/Qwen3-4B',
            'qwen3-8b': 'Qwen/Qwen3-8B',
            'qwen3-14b': 'Qwen/Qwen3-14B',
            'qwen3-32b': 'Qwen/Qwen3-32B',
        }
        model_id = name_to_id[model_name]
        model_path = os.path.join(weights_dir, model_id)

        self.model_name = model_name
        self.model_id = model_id        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="bfloat16",
            device_map="auto",
        )

    def generate(
        self,
        query: str,
        system_prompt: str = None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        response = self.tokenizer.decode(
            output_ids, 
            skip_special_tokens=True,
        )

        return response


class GemmaModel:

    def __init__(
        self,
        model_name: str = "gemma-3-1b",
        weights_dir: str = "/geovic/ghermi/weights",
    ):
        name_to_id = {
            'gemma-3-270m': 'google/gemma-3-270m-it',
            'gemma-3-1b': 'google/gemma-3-1b-it',
            'gemma-3-4b': 'google/gemma-3-4b-it',
            'gemma-3-12b': 'google/gemma-3-12b-it',
            'gemma-3-27b': 'google/gemma-3-27b-it',
        }
        model_id = name_to_id[model_name]
        model_path = os.path.join(weights_dir, model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype="bfloat16",
        )

        self.model_name = model_name
        self.model_id = model_id 

    def generate(
        self,
        query: str,
        system_prompt: str = None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self.model.device,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=256,
            )

        trimmed_ids = generated_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
        
        return response


class HFModel:
    """Generic HuggingFace causal LM (Llama, Mistral, etc.)"""

    MODEL_IDS = {
        # Llama
        'llama-3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
        'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama-3.1-70b': 'meta-llama/Llama-3.1-70B-Instruct',
        # Mistral
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
        'mistral-small': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
        # SmolLM2
        'smollm2-135m': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'smollm2-360m': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'smollm2-1.7b': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        # SmolLM3
        'smollm3-3b': 'HuggingFaceTB/SmolLM3-3B',
        # OLMo2
        'olmo2-7b': 'allenai/OLMo-2-1124-7B-Instruct',
        'olmo2-13b': 'allenai/OLMo-2-1124-13B-Instruct',
    }

    def __init__(
        self,
        model_name: str,
        weights_dir: str = "/geovic/ghermi/weights",
    ):
        model_id = self.MODEL_IDS[model_name]
        model_path = os.path.join(weights_dir, model_id)

        self.model_name = model_name
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="bfloat16",
            device_map="auto",
        )

    def generate(
        self,
        query: str,
        system_prompt: str = None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


class ClaudeModel:

    def __init__(
        self,
        model_name: str,
    ):
        assert model_name in [
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ]
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(
        self,
        query: str,
        system_prompt: str = None,
    ):
        kwargs = {
            "model": self.model_name,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": query}],
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        try:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            print(f"Error getting response: {e}")
            return None


def get_model(model_name: str, weights_dir: str = "/geovic/ghermi/weights"):
    if model_name in [
        "qwen3-0.6b",
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
    ]:
        return QwenModel(model_name=model_name, weights_dir=weights_dir)
    elif model_name in [
        "gemma-3-270m",
        "gemma-3-1b",
        "gemma-3-4b",
        "gemma-3-12b",
        "gemma-3-27b",
    ]:
        return GemmaModel(model_name=model_name, weights_dir=weights_dir)
    elif model_name in [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5.4",
    ]:
        return GPTModel(model_name=model_name)
    elif model_name in HFModel.MODEL_IDS:
        return HFModel(model_name=model_name, weights_dir=weights_dir)
    elif model_name in [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ]:
        return ClaudeModel(model_name=model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main(args):
    # Validate --mcqa flag
    dataset_name = Path(args.input_path).stem
    if args.mcqa and dataset_name not in MCQA_DATASETS:
        raise ValueError(f"--mcqa is only supported for {MCQA_DATASETS}, got dataset '{dataset_name}'")

    # Setup output file
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = "no_title_" if args.no_title else ""
    suffix = "_mcqa" if args.mcqa else ""
    output_filename = f"{prefix}dataset_{dataset_name}_model_{args.model_name}{suffix}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Results will be saved to {output_path}")

    # Initialize prompt
    prompt = MCQAPrompt(no_title=args.no_title) if args.mcqa else Prompt(no_title=args.no_title)

    # Initialize dataset
    dataset = Dataset(
        input_path=args.input_path,
        prompt=prompt,
        n_subsample=args.n_subsample,
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    # Initialize model
    print(f"Loading model {args.model_name}...")
    model = get_model(model_name=args.model_name, weights_dir=args.weights_dir)
    print("Model loaded successfully")

    # Generation loop
    results = json.load(open(output_path, "r")) if os.path.exists(output_path) and not args.force_rerun else {}
    existing_ids = set(results.keys())
    print(f"{len(existing_ids)} samples loaded")
    A = set(existing_ids)
    B = set([x['question_id'] for x in dataset])
    print(len(A), len(B), len(A.intersection(B)))
    #import pdb; pdb.set_trace()

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        question_id = str(sample["question_id"])
        if question_id in existing_ids and not args.force_rerun:
            continue

        #try:
        response = model.generate(
            query=sample["query"],
            #system_prompt="You are a helpful assistant.",
        )

        prediction = prompt.postprocess_response(response)

        result = {
            "question_id": question_id,
            "movie_title": sample["movie_title"],
            "question": sample["question"],
            "answer": sample["answer"],
            "response": response,
            "prediction": prediction,
            "model": args.model_name,
        }
        if args.mcqa:
            import ast as _ast
            options = sample["options"]
            if isinstance(options, str):
                options = _ast.literal_eval(options)
            result["options"] = options
            result["correct_idx"] = int(sample["correct_idx"])
        results[question_id] = result

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        if args.mcqa:
            mcqa_results = [r for r in results.values() if "correct_idx" in r]
            n_correct = sum(
                r["prediction"] == OPTION_LETTERS[r["correct_idx"]]
                for r in mcqa_results
            )
            print(f"Accuracy: {n_correct}/{len(mcqa_results)} ({100*n_correct/len(mcqa_results):.1f}%)")

        #except Exception as e:
        #    print(f"Error processing sample {question_id}: {e}")
        #    continue

    print(f"Saved results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    if args.mcqa:
        mcqa_results = [r for r in results.values() if "correct_idx" in r]
        n_correct = sum(
            r["prediction"] == OPTION_LETTERS[r["correct_idx"]]
            for r in mcqa_results
        )
        print(f"Final accuracy: {n_correct}/{len(mcqa_results)} ({100*n_correct/len(mcqa_results):.1f}%)")
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
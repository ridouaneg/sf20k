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


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="movieqa.csv")
    parser.add_argument("--title_source_path", type=str, default=None, help="Optional: A second dataset to steal titles from.")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="gpt-5-nano")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--n_subsample", type=int, default=-1)
    return parser.parse_args()


class Prompt:

    def __init__(self):
        self.template = "In the movie '{movie_title}', {question}\nAnswer shortly and directly without repeating the question. If you don't know the movie, try to guess the answer."

    def get_query(self, sample):
        question = sample['question']
        question = question[0].lower() + question[1:] # remove first upper case
        return self.template.format(
            movie_title=sample["movie_title"],
            question=question,
        )

    def postprocess_response(self, response):
        return response.strip() if response is not None else None


class Dataset:

    def __init__(
        self, 
        input_path: str, 
        prompt, 
        title_source_path: str = None,
        n_subsample: int = -1,
    ):
        self.df = pd.read_csv(input_path)
        self.prompt = prompt

        assert "question_id" in self.df.columns
        assert "movie_title" in self.df.columns
        assert "question" in self.df.columns
        assert "answer" in self.df.columns

        if n_subsample > 0:
            self.df = self.df.sample(n=n_subsample, random_state=42)

        if title_source_path:
            # Load the second dataset purely for its titles
            print(f"Swapping titles: replacing titles in {Path(input_path).name} with titles from {Path(title_source_path).name}")
            df_source = pd.read_csv(title_source_path)
            df_source = df_source.sample(n=n_subsample, random_state=42)
            self.df['movie_title'] = np.random.permutation(df_source['movie_title'].values)
        else:
            # Original behavior: shuffle in-domain
            print(f"Shuffling titles within {Path(input_path).name}")
            self.df['movie_title'] = np.random.permutation(self.df['movie_title'].values)

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
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
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


def get_model(model_name: str):
    if model_name in [
        "qwen3-0.6b",
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
    ]:
        return QwenModel(model_name=model_name)
    elif model_name in [
        "gemma-3-270m",
        "gemma-3-1b",
        "gemma-3-4b",
        "gemma-3-12b",
        "gemma-3-27b",
    ]:
        return GemmaModel(model_name=model_name)
    elif model_name in [
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
    ]:
        return GPTModel(model_name=model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main(args):
    # Setup output file
    os.makedirs(args.output_dir, exist_ok=True)

    input_stem = Path(args.input_path).stem
    if args.title_source_path:
        source_stem = Path(args.title_source_path).stem
        output_filename = f"title_swap_TARGET_{input_stem}_SOURCE_{source_stem}_model_{args.model_name}.json"
    else:
        output_filename = f"title_shuffling_dataset_{input_stem}_model_{args.model_name}.json"
    
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Results will be saved to {output_path}")

    # Initialize prompt
    prompt = Prompt()

    # Initialize dataset
    dataset = Dataset(
        input_path=args.input_path,
        prompt=prompt,
        title_source_path=args.title_source_path,
        n_subsample=args.n_subsample,
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    # Initialize model
    print(f"Loading model {args.model_name}...")
    model = get_model(model_name=args.model_name)
    print("Model loaded successfully")

    # Generation loop
    results = json.load(open(output_path, "r")) if os.path.exists(output_path) and not args.force_rerun else {}
    existing_ids = set(results.keys())

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        question_id = sample["question_id"]
        if question_id in existing_ids and not args.force_rerun:
            continue

        try:
            response = model.generate(
                query=sample["query"],
            )
            
            prediction = prompt.postprocess_response(response)

            results[question_id] = {
                "question_id": question_id,
                "movie_title": sample["movie_title"],
                "question": sample["question"],
                "answer": sample["answer"],
                "response": response,
                "prediction": prediction,
                "model": args.model_name,
            }
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            
        except Exception as e:
            print(f"Error processing sample {question_id}: {e}")
            continue

    print(f"Saved results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
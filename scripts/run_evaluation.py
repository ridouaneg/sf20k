import argparse
import ast
import os
import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)


TEMPLATE_PROMPT = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {prediction}\n\n"
    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    return parser.parse_args()


def main(args):
    # Prepare data
    df = pd.read_csv(args.pred_path)

    # Prepare model
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_path = os.path.join(args.model_dir, model_id)
    model = pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
    )

    # Evaluate
    outputs = []
    for i, sample in tqdm(df.iterrows(), total=len(df)):
        USER_PROMPT = TEMPLATE_PROMPT.format(question=sample['question'], answer=sample['answer'], prediction=sample['prediction'])
        prompt = f"<<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n[INST] {USER_PROMPT} [/INST]"
        output = model(prompt, max_new_tokens=16, do_sample=False)
        output = output[0]["generated_text"].removeprefix(prompt).removesuffix('[/INST]').strip()
        outputs.append(output)

    scores = []
    for output in outputs:
        try:
            score = ast.literal_eval(output)["score"]
        except:
            score = 0
        scores.append(score)

    df['score'] = scores
    df.to_csv(args.output_path, index=False)

    print(f"Average score: {np.mean(scores)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

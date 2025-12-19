import ast
import os
import numpy as np
import pandas as pd
import openai
import argparse
import json
from tqdm import tqdm


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", None)


SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)


PROMPT_TEMPLATE = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {prediction}\n\n"
    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {{'pred': 'yes', 'score': 4}}."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="results.json")
    parser.add_argument("--force_rerun", action="store_true")
    return parser.parse_args()


def main(args):
    # Prepare client
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID,
    )

    # Prepare data
    data = json.load(open(args.pred_path))
    
    # Evaluate
    output_path = args.pred_path.replace(".json", "_eval.json")
    results = json.load(open(output_path)) if os.path.exists(output_path) and not args.force_rerun else {}
    existing_ids = set(results.keys())

    for question_id, sample in tqdm(data.items(), total=len(data)):
        if question_id in existing_ids and not args.force_rerun:
            continue
        question = sample['question']
        answer = sample['answer']
        prediction = sample['prediction']

        if prediction is None:
            continue
        
        USER_PROMPT = PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            prediction=prediction,
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                max_tokens=16,
                temperature=0.0, # Set to 0 for deterministic output
            )
            output = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred with the OpenAI API call: {e}")
            output = None

        try:
            score = int(ast.literal_eval(output)["score"])
            pred = 1 * (str(ast.literal_eval(output)["pred"]).lower() == 'yes')
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing the output: {e}\nOutput was: {output}")
            score = 0
            pred = 0

        results[question_id] = {
            "question_id": sample['question_id'],
            "question": sample['question'],
            "answer": sample['answer'],
            "response": sample['response'],
            "prediction": sample['prediction'],
            "model": sample['model'],
            "score": score,
            "pred": pred,
        }

    # Compute metrics
    df = pd.DataFrame(results.values())
    accuracy = df['pred'].mean()
    score = df['score'].mean()
    print(f"Accuracy: {accuracy}")
    print(f"Score: {score}")

    # Save results
    json.dump(results, open(output_path, "w"), indent=4)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
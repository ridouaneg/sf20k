import ast
import numpy as np
import pandas as pd
import openai
import argparse
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
    parser.add_argument("--pred_path", type=str, default="submission.csv")
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_org_id", type=str, default=None)
    return parser.parse_args()


def main(args):
    # Prepare client
    if args.openai_org_id is None:
        client = openai.OpenAI(
            api_key=args.openai_api_key,
        )
    else:
        client = openai.OpenAI(
            api_key=args.openai_api_key,
            organization=args.openai_org_id,
        )

    # Prepare data
    submission_df = pd.read_csv(args.pred_path)

    # Prepare labels
    #df = load_dataset("rghermi/sf20k-private", split="test_expert", token=os.environ.get("HF_TOKEN")).to_pandas()
    df = pd.read_csv("../data/gt_samples.csv")
    df = pd.merge(df[['question_id', 'video_id', 'question', 'answer']], submission_df[['question_id', 'prediction']], on='question_id', how='inner')
    
    # Evaluate
    outputs = []
    for _, sample in tqdm(df.iterrows(), total=len(df)):
        question = sample['question']
        answer = sample['answer']
        prediction = sample['prediction']

        if prediction is None:
            outputs.append(None)
            continue
        
        USER_PROMPT = PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            prediction=prediction,
        )
        
        try:
            response = client.chat.completions.create(
                #model="gpt-3.5-turbo-0125",
                model="gpt-4.1-nano-2025-04-14",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                max_tokens=16,
                temperature=0.0, # Set to 0 for deterministic output
            )
            output = response.choices[0].message.content
            outputs.append(output)
        except Exception as e:
            print(f"An error occurred with the OpenAI API call: {e}")
            outputs.append(None) # Append None or a default error indicator

    scores = []
    for output in outputs:
        if output is None:
            scores.append(0)
            continue
        try:
            score = ast.literal_eval(output)["score"]
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing the output: {e}\nOutput was: {output}")
            score = 0
        scores.append(score)

    score = np.sum(scores) / 538. if scores else 0.

    print(scores)
    print(f"Score: {score}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

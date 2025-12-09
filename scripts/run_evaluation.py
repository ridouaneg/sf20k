import ast
import numpy as np
import pandas as pd
import openai
import argparse
from tqdm import tqdm

from sf20k.metrics import LLMQAEval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="submission.csv")
    return parser.parse_args()


def main(args):
    # Prepare metric
    metric = LLMQAEval()

    # Prepare data
    df = pd.read_csv(args.pred_path)
    # df.columns should include ['question_id', 'question', 'answer', 'prediction']
    
    # Evaluate
    scores, preds = [], []
    for _, sample in tqdm(df.iterrows(), total=len(df)):
        score, pred = metric.compute(
            sample['question'],
            sample['answer'],
            sample['prediction'],
        )
        scores.append(score)
        preds.append(pred)
    
    df['score'] = scores
    df['prediction'] = preds

    output_path = args.pred_path.replace(".csv", "_eval.csv")
    df.to_csv(output_path, index=False)
    
    score = np.mean(scores)
    pred = np.mean(preds)

    print(f"Score: {score}")
    print(f"Prediction: {pred}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
import argparse
import json
import re
import os
import openai
import logging
import ast
import random
import numpy as np
import torch
from tqdm import tqdm

OPENAI_API_KEY = ""
OPENAI_ORG_ID = ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--n_subsample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def calculate_score_and_accuracy(results):
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for question_id in results.keys():
        sample = results[question_id]
        try:
            pred, score = sample['pred'], sample['score']
            score_sum += score
            count += 1
            if pred == 'yes':
                yes_count += 1
            else:
                no_count += 1
        except Exception as e:
            logging.info(f"Error in sample {question_id}: {e}")
    
    average_score = score_sum / (count + 1e-5)
    accuracy = yes_count / (yes_count + no_count + 1e-5)

    return count, yes_count, no_count, accuracy, average_score

def main(args):
    # Set seed and logging
    setup_logging()
    set_seed(args.seed)

    # Log config
    logging.info(args)

    # Prepare data
    data = json.load(open(args.input_file))
    if args.n_subsample > 0:
        data = dict(random.sample(data.items(), args.n_subsample))

    output_path = args.input_file.replace(".json", "_eval.json")
    results = json.load(open(output_path)) if os.path.exists(output_path) else {}

    question_mode = 'mcqa' if 'mcqa' in args.input_file else 'oeqa'
    if question_mode == "mcqa":
        pattern = r"\((\b[1-5]\b)\)"
        for key, value in tqdm(data.items(), total=len(data)):
            question_id = key
            video_id = value['video_id']
            question = value['question']
            response = value['prediction']
            correct_answer = value['correct_answer']

            if question_id in results:
                continue
            
            match = re.search(pattern, response)
            if match:
                result = int(match.group(1)) - 1
            else:
                result = None

            acc = 1 * (result == correct_answer)
            
            results[question_id] = {
                'question_id': question_id,
                'video_id': video_id,
                'question': question,
                'response': response,
                'correct_answer': correct_answer,
                'pred': result,
                'acc': acc,
            }
            json.dump(results, open(output_path, "w"), indent=4)

        cnt = 0
        for key, value in results.items():
            if value['acc']:
                cnt += 1
        
        #logging.info(f"Accuracy: {cnt / len(results) * 100:.2f}%")
        #logging.info(f"Total: {len(results)}")
    elif question_mode == 'oeqa':
        # Set OpenAI API key
        openai.api_key = OPENAI_API_KEY
        openai.organization = OPENAI_ORG_ID
        model_name = 'gpt-3.5-turbo-0125'

        for key, value in tqdm(data.items(), total=len(data)):
            question_id = key
            video_id = value['video_id']
            question = value['question']
            answer = value[f"option_{value['correct_answer']}"]
            prediction = value['prediction']

            if question_id in results:
                continue

            try:
                # Compute the correctness score
                completion = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {prediction}\n\n"
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
                )

                # Convert response to a Python dictionary.
                response_message = completion["choices"][0]["message"]["content"]
                response_dict = ast.literal_eval(response_message)
                pred, score = response_dict['pred'], response_dict['score']
                
                # Save response
                results[question_id] = {
                    'video_id': video_id,
                    'question_id': question_id,
                    'question': question,
                    'answer': answer,
                    'response': prediction,
                    'pred': pred,
                    'score': score,
                }
                #_, _, _, accuracy, average_score = calculate_score_and_accuracy(results)
                #logging.info(f"Accuracy: {accuracy * 100:.2f}%")
                #logging.info(f"Average score: {average_score}")
                #logging.info(f"Total: {len(results)}")

                # Save results
                json.dump(results, open(output_path, "w"), indent=4)
            except Exception as e:
                logging.info(f"Error in sample {question_id}: {e}")
            
        _, _, _, accuracy, average_score = calculate_score_and_accuracy(results)
        logging.info(f"Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"Average score: {average_score}")
        logging.info(f"Total: {len(results)}")
    else:
        raise ValueError(f"Invalid question_mode: {args.question_mode}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

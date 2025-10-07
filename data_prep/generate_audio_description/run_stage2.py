import json
import argparse
from tqdm import tqdm
import os
from openai import OpenAI


STAGE2_PROMPT = (
    "Please summarise the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
    "Description: {text_pred}\n\n"
    "Focus on the most attractive characters and their actions (focus on point 2., supplemented by point 3.).\n"
    #"For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
    "For characters, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
    "For actions, avoid mentioning the camera, and do not focus on 'talking' or position-related ones such as 'sitting' and 'standing'.\n"
    "Do not mention characters' mood.\n"
    "Do not hallucinate information that is not mentioned in the input.\n"
    "Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
    "Provide the AD from a narrator perspective and adjust the length of the output according to the duration.\n"
    "Duration of the video clip: {duration}s\n\n"
    #"For example, a response of duration 0.8s could be: {'summarised_AD': 'She looks at Riker.'}.\n"
    #"Another example response of duration 1.4s is: {'summarised_AD': 'Paul looks at his wife lovingly.'}.\n"
    #"An example response of duration 2.6s is: {'summarised_AD': 'He watches Tasha calmly battle with the figure.'}.\n"
    "For example, a response of duration 0.8s could be: {'summarised_AD': 'She looks at him.'}.\n"
    "Another example response of duration 1.4s is: {'summarised_AD': 'Hi looks at his wife lovingly.'}.\n"
    "An example response of duration 2.6s is: {'summarised_AD': 'He watches her calmly battle with the figure.'}.\n"
)


class GPTModel:
    
    def __init__(self, model_name: str, openai_api_key: str, openai_org_id: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=openai_api_key, organization=openai_org_id)

    def generate(self, query: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
        )
        return response.choices[0].message.content


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="stage1.json")
    parser.add_argument('--output_path', type=str, default="stage2.json")
    parser.add_argument('--model_name', type=str, default="gpt-4.1-mini", choices=[
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
    ])
    parser.add_argument('--openai_api_key', type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument('--openai_org_id', type=str, default=os.getenv("OPENAI_ORG_ID"))
    return parser.parse_args()


def main(args):
    # Prepare dataset
    data = json.load(open(args.input_path))
    verb_list = ['look', 'turn', 'take', 'hold', 'pull', 'walk', 'run', 'watch', 'stare', 'grab', 'fall', 'get', 'go', 'open', 'smile']

    # Prepare model
    model = GPTModel(args.model_name, args.openai_api_key, args.openai_org_id)
    
    # Run inference
    results = []
    for row in tqdm(data, total=len(data)):
        text_pred = row['prediction']
        duration = round(row['end_time']-row['start_time'], 2)

        query = STAGE2_PROMPT.format(
            text_pred=text_pred,
            duration=duration,
            verb_list=verb_list,
        )

        prediction = model.generate(query=query)
        row['stage2_prediction'] = prediction
        results.append(row)

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
import pandas as pd
import random
import json

df = pd.read_csv('../data/test_expert.csv')
all_answers = df.answer.tolist()

results = {}
for i, sample in df.iterrows():
    prediction = random.choice(all_answers)
    question_id = sample["question_id"]
    results[question_id] = {
        "question_id": question_id,
        "video_id": sample["video_id"],
        "question": sample["question"],
        "answer": sample["answer"], # Ground truth
        "response": prediction,
        "prediction": prediction,
        "model": "random",
        #"modality": args.modality,
        #"num_frames": args.num_frames,
    }

json.dump(results, open('./results/random_baseline.json', 'w'))

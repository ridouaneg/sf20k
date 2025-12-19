import pandas as pd
import json

# movieqa
movies = json.load(open('/geovic/ghermi/data/MovieQA/movies.json'))
qa = json.load(open('/geovic/ghermi/data/MovieQA/qa.json'))
splits = json.load(open('/geovic/ghermi/data/MovieQA/splits.json'))
df = pd.DataFrame(qa)
df_val = df[df.imdb_key.isin(splits['val'])]
df_val['answer'] = df_val.apply(lambda row: row['answers'][int(row.correct_index)], axis=1)
df_val['question_id'] = df_val['qid']
movie_titles = {x['imdb_key']: x['name'] for x in movies}
df_val['movie_title'] = df_val['imdb_key'].map(movie_titles)
df_val = df_val[['question_id', 'question', 'answer', 'movie_title']]
df_val.to_csv('data/movieqa.csv', index=False)

print(f"MovieQA: {len(df_val)}")

# sf20k
df = pd.read_csv('../../data/test_expert.csv')
df = df[['video_id', 'question_id', 'question', 'answer']]
metadata = pd.read_csv("data/sf20k_metadata.csv")
df = pd.merge(df, metadata, on="video_id", how='inner')
df = df[['question_id', 'question', 'answer', 'movie_title']]
df.to_csv('data/sf20k.csv', index=False)

print(f"SF20K: {len(df)}")
import os
import ast
import pandas as pd
from torch.utils.data import Dataset


class SF20KDataset(Dataset):

    def __init__(
        self,
        prompt = None,
        data_path: str = None,
        video_dir: str = None,
        subtitles_path: str = None,
        n_subsample: int = -1,
        seed: int = 42,
    ):
        # Load data
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        # Prepare video paths
        #df['video_path'] = df['video_id'].apply(lambda x: os.path.join(video_dir, f"{x}.mp4"))
        df['video_path'] = df['video_id'].apply(lambda x: os.path.join(video_dir, f"{x}.mkv"))
        df = df[df.video_path.apply(os.path.exists)]
        #video_files = {row['video_id']: row['video_path'] for _, row in df.iterrows()}

        # Load subtitles
        df_subs = pd.read_csv(subtitles_path) if subtitles_path.endswith('.csv') else pd.read_parquet(subtitles_path)
        df_subs = df_subs[df_subs.video_id.isin(df.video_id.unique())]
        subtitles_series = df_subs.groupby('video_id')['text'].apply(lambda x: '\n'.join(x.fillna('').astype(str)).strip())
        subtitles_series = subtitles_series.replace('', 'No subtitles.')
        subtitles_dict = subtitles_series.to_dict()
        for video_id in df.video_id.unique():
            if video_id not in subtitles_dict:
                subtitles_dict[video_id] = 'No subtitles.'

        self.df = df
        self.subtitles_dict = subtitles_dict
        self.n_subsample = n_subsample
        self.seed = seed
        self.prompt = prompt
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].copy()
        video_id = sample['video_id']
        video_path = sample['video_path']
        subtitles = self.subtitles_dict[video_id]
        
        query = self.prompt.get_query(sample, subtitles=subtitles)
        response = self.prompt.get_response(sample)
        
        return {
            'question_id': sample['question_id'],
            'video_id': sample['video_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'system_prompt': None,
            'query': query,
            'video_path': video_path,
            'response': response,
            'text': "",
        }
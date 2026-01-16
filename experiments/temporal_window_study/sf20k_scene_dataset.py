import os
import pandas as pd
from torch.utils.data import Dataset
import ast
import cv2


class SF20KSceneDataset(Dataset):

    def __init__(
        self,
        prompt = None,
        data_path: str = None,
        video_dir: str = None,
        subtitles_path: str = None,
        n_scenes: int = 1,
        n_subsample: int = -1,
        seed: int = 42,
    ):
        df = pd.read_csv(data_path)
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        # Prepare video paths
        #df['video_path'] = df['video_id'].apply(lambda x: os.path.join(video_dir, f"{x}.mp4"))
        df['video_path'] = df['video_id'].apply(lambda x: os.path.join(video_dir, f"{x}.mkv"))
        df = df[df.video_path.apply(os.path.exists)]

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
        self.n_scenes = n_scenes
        
    def __len__(self):
        return len(self.df) * self.n_scenes

    def __getitem__(self, idx):
        sample = self.df.iloc[idx // self.n_scenes].copy()

        video_id = sample['video_id']
        video_path = sample['video_path']
        subtitles = self.subtitles_dict[video_id]

        duration = sample['duration']
        scene_nb = idx % self.n_scenes
        video_start = scene_nb * duration / self.n_scenes
        video_end = (scene_nb + 1) * duration / self.n_scenes

        query = self.prompt.get_query(sample, subtitles=subtitles)
        response = self.prompt.get_response(sample)

        return {
            'question_id': f"{sample['question_id']}_{scene_nb:04d}",
            'video_id': video_id,
            'video_path': video_path,
            'video_start': video_start,
            'video_end': video_end,
            'question': sample['question'],
            'answer': sample['answer'],
            'options': [sample[f'option_{i}'] for i in range(5)] if 'option_0' in sample else None,
            'answer_id': sample['answer_id'] if 'answer_id' in sample else None,
            'query': query,
            'response': response,
        }
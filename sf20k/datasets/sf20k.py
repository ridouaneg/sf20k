import os
import pandas as pd
from torch.utils.data import Dataset

from .templates import TEMPLATE


class SF20KDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        subtitles_path: str,
        video_dir: str,
        n_subsample: int = -1,
        seed: int = 42,
    ):
        df = pd.read_csv(data_path)
        df = df[['question_id', 'video_id', 'question', 'answer']]
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        df_subs = pd.read_csv(subtitles_path)
        subtitles_dict = {}
        for video_id in df.video_id.unique():
            subtitles = '\n'.join(df_subs[(df_subs.video_id == video_id)].text.tolist())
            subtitles_dict[video_id] = subtitles

        self.df = df
        self.subtitles_dict = subtitles_dict
        self.video_dir = video_dir
        self.n_subsample = n_subsample
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        video_id = sample['video_id']
        subtitles = self.subtitles_dict[video_id]
        query = TEMPLATE.format(subtitles=subtitles, question=sample['question'])
        return {
            'video_id': video_id,
            'query': query,
            'video_path': os.path.join(self.video_dir, f"{video_id}.mkv"),
            'question': sample['question'],
            'answer': sample['answer']
        }
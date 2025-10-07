import os
import pandas as pd
from torch.utils.data import Dataset

from .templates import STAGE1_AD_PROMPT, STAGE2_AD_PROMPT


class SF20KDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        shots_path: str,
        video_dir: str,
        n_subsample: int = -1,
        seed: int = 42,
    ):
        df = pd.read_csv(data_path)
        df = df[['video_id']]
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        df_shots = pd.read_csv(shots_path)
        df_shots = df_shots[['shot_id', 'video_id', 'start', 'end']]

        all_clips = []
        for _, shot_row in df_shots.iterrows():
            video_path = os.path.join(video_dir, f"{shot_row['video_id']}.mkv")
            if not os.path.exists(video_path):
                continue
                
            all_clips.append(
                {
                    'video_id': shot_row['video_id'],
                    'video_path': video_path,
                    'start': shot_row['start'],
                    'end': shot_row['end'],
                }
            )

        self.all_clips = all_clips
        self.n_subsample = n_subsample
        self.seed = seed

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        sample = self.all_clips[idx]
        video_id = sample['video_id']
        video_path = sample['video_path']
        start_time = sample['start']
        end_time = sample['end']
        query = STAGE1_AD_PROMPT.format()

        return {
            'video_id': video_id,
            'query': query,
            'video_path': video_path,
            'start_time': start_time,
            'end_time': end_time,
        }
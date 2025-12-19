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
        df = pd.read_csv(data_path)
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        # Load subtitles
        df_subs = pd.read_csv(subtitles_path) if subtitles_path.endswith('.csv') else pd.read_parquet(subtitles_path)
        subtitles_dict = {}
        for video_id in df.video_id.unique():
            subtitles = '\n'.join(df_subs[(df_subs.video_id == video_id)].text.fillna('').astype(str).tolist())
            subtitles = 'No subtitles.' if subtitles.strip() == '' else subtitles.strip()
            subtitles_dict[video_id] = subtitles

        # Prepare video paths
        video_files = {}
        for video_id in df.video_id.unique():
            #video_path = os.path.join(video_dir, f"{video_id}.mp4")
            video_path = os.path.join(video_dir, f"{video_id}.mkv")
            if os.path.exists(video_path):
                video_files[video_id] = video_path
        
        # Filter out videos that don't exist
        df = df[df.video_id.isin(video_files.keys())]

        self.df = df
        self.subtitles_dict = subtitles_dict
        self.video_files = video_files
        self.n_subsample = n_subsample
        self.seed = seed
        self.prompt = prompt
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].copy()
        video_id = sample['video_id']
        video_path = self.video_files[video_id]
        sample['subtitles'] = self.subtitles_dict[video_id]
        query = self.prompt.get_query(sample)
        response = self.prompt.get_response(sample)
        return {
            #'question_id': sample['question_id'],
            #'video_id': video_id,
            #'question': sample['question'],
            #'answer': sample['answer'],
            #'options': [sample[f'option_{i}'] for i in range(5)] if 'option_0' in sample else None,
            #'answer_id': sample['answer_id'] if 'answer_id' in sample else None,
            'system_prompt': None,
            'video_path': video_path,
            'query': query,
            'response': response,
            'text': "",
        }
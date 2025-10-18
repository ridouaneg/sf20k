import os
import pandas as pd
from torch.utils.data import Dataset

from ..templates import TEMPLATE_VL, TEMPLATE_L, TEMPLATE_V


class SF20KDataset(Dataset):

    def __init__(
        self,
        modality: str = "vision_language",
        data_path: str,
        subtitles_path: str,
        video_dir: str,
        n_subsample: int = -1,
        seed: int = 42,
    ):
        assert modality in ["vision", "language", "vision_language"]

        df = pd.read_csv(data_path)
        df = df[['question_id', 'video_id', 'question', 'answer']]
        if n_subsample > -1:
            df = df.sample(n=n_subsample, random_state=seed)

        df_subs = pd.read_csv(subtitles_path)
        subtitles_dict = {}
        for video_id in df.video_id.unique():
            subtitles = '\n'.join(df_subs[(df_subs.video_id == video_id)].text.tolist())
            subtitles_dict[video_id] = subtitles

        video_files = {}
        for video_id in df.video_id.unique():
            video_path = os.path.join(video_dir, f"{video_id}.mkv")
            if os.path.exists(video_path):
                video_files[video_id] = video_path
        
        df = df[df.video_id.isin(video_files.keys())]

        self.df = df
        self.subtitles_dict = subtitles_dict
        self.video_files = video_files
        self.n_subsample = n_subsample
        self.seed = seed
        self.modality = modality
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        video_id = sample['video_id']
        subtitles = self.subtitles_dict[video_id]
        question = sample['question']

        if self.modality == "vision_language":
            query = TEMPLATE_VL.format(question=question, subtitles=subtitles)
        elif self.modality == "language":
            query = TEMPLATE_L.format(question=question, subtitles=subtitles)
        elif self.modality == "vision":
            query = TEMPLATE_V.format(question=question)
        else:
            raise ValueError(f"Invalid modality: {self.modality}")
        
        return {
            'question_id': sample['question_id'],
            'video_id': video_id,
            'query': query,
            'video_path': self.video_files[video_id],
            'question': question,
            'answer': sample['answer'],
            'system_prompt': None,
        }
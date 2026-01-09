import os
import pandas as pd
from torch.utils.data import Dataset
import ast


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

        # Load subtitles
        df_subs = pd.read_csv(subtitles_path)
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

        # Get video durations
        def get_video_duration(video_path):
            import cv2
            cap = cv2.VideoCapture(video_path)
            return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        df['duration'] = df.video_id.apply(lambda x: get_video_duration(video_files[x]))

        self.df = df
        self.subtitles_dict = subtitles_dict
        self.video_files = video_files
        self.n_scenes = n_scenes
        self.n_subsample = n_subsample
        self.seed = seed
        self.prompt = prompt
        
    def __len__(self):
        return len(self.df) * self.n_scenes

    def __getitem__(self, idx):
        sample = self.df.iloc[idx // self.n_scenes].copy()

        video_id = sample['video_id']
        video_path = self.video_files[video_id]

        duration = sample['duration']
        scene_nb = idx % self.n_scenes
        video_start = scene_nb * duration / self.n_scenes
        video_end = (scene_nb + 1) * duration / self.n_scenes

        subtitles = self.subtitles_dict[video_id]
        query = self.prompt.get_query(sample, subtitles=subtitles)
        response = self.prompt.get_response(sample)

        return {
            'question_id': f"{sample['question_id']}_{scene_nb}",
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
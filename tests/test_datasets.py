
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

from sf20k.datasets.sf20k import SF20KDataset


class TestSF20KDataset(unittest.TestCase):

    def test_dataset(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/test_expert.csv')
        subtitles_path = os.path.join(base_dir, '../data/test_subtitles.csv')
        video_dir = '/geovic/geovic/SF20K/videos'
        
        if not os.path.exists(data_path) or not os.path.exists(subtitles_path) or not os.path.exists(video_dir):
            self.skipTest("Real data not found")

        dataset = SF20KDataset(
            data_path=data_path,
            subtitles_path=subtitles_path,
            video_dir=video_dir,
            task='oeqa',
            modality='vision_language'
        )
        
        self.assertGreater(len(dataset), 0)
        item = dataset[0]

        self.assertIn('question_id', item)
        self.assertIn('video_id', item)
        self.assertIn('video_path', item)
        self.assertTrue(os.path.exists(item['video_path']))
        self.assertIn('question', item)
        self.assertIn('answer', item)
        self.assertIn('answer_id', item)
        self.assertIn('options', item)
        self.assertTrue(len(item['options']) == 5)
        self.assertIn('query', item)
        self.assertIn('response', item)


if __name__ == '__main__':
    unittest.main()
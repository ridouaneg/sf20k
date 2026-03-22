import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json
import tempfile
import pandas as pd

# Add scripts to path to import run_baselines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
import run_baselines

class TestRunBaselines(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "data.csv")
        self.subtitles_path = os.path.join(self.test_dir, "subtitles.csv")
        self.video_dir = os.path.join(self.test_dir, "videos")
        self.output_dir = os.path.join(self.test_dir, "results")
        os.makedirs(self.video_dir)

        # Create dummy data
        df = pd.DataFrame({
            "question_id": [1],
            "video_id": ["vid1"],
            "question": ["What is happening?"],
            "answer": ["A"],
            "answer_id": [0],
            "option_0": ["A"],
            "option_1": ["B"],
            "option_2": ["C"],
            "option_3": ["D"],
            "option_4": ["E"],
        })
        df.to_csv(self.data_path, index=False)

        # Create dummy subtitles
        df_subs = pd.DataFrame({
            "video_id": ["vid1"],
            "text": ["Hello world"]
        })
        df_subs.to_csv(self.subtitles_path, index=False)

        # Create dummy video
        with open(os.path.join(self.video_dir, "vid1.mkv"), "w") as f:
            f.write("dummy video content")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    @patch("run_baselines.get_model")
    def test_run_baselines_mcqa(self, mock_get_model):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = "A"
        mock_get_model.return_value = mock_model

        # Setup args
        test_args = [
            "run_baselines.py",
            "--model_name", "test_model",
            "--data_path", self.data_path,
            "--video_dir", self.video_dir,
            "--subtitles_path", self.subtitles_path,
            "--output_dir", self.output_dir,
            "--task_type", "mcqa",
            "--modality", "vision_language"
        ]

        with patch.object(sys, 'argv', test_args):
            run_baselines.main()

        # Verify output
        output_file = os.path.join(self.output_dir, "test_model_mcqa_vision_language_8f.json")
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, "r") as f:
            results = json.load(f)
            self.assertEqual(len(results), 1)
            result = results["1"]
            self.assertEqual(result["question_id"], 1)
            self.assertEqual(result["response"], "A")
            self.assertEqual(result["prediction"], 0) # A -> 0

    @patch("run_baselines.get_model")
    def test_run_baselines_oeqa(self, mock_get_model):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = "Some answer"
        mock_get_model.return_value = mock_model

        # Setup args
        test_args = [
            "run_baselines.py",
            "--model_name", "test_model",
            "--data_path", self.data_path,
            "--video_dir", self.video_dir,
            "--subtitles_path", self.subtitles_path,
            "--output_dir", self.output_dir,
            "--task_type", "oeqa",
            "--modality", "vision"
        ]

        with patch.object(sys, 'argv', test_args):
            run_baselines.main()

        # Verify output
        output_file = os.path.join(self.output_dir, "test_model_oeqa_vision_8f.json")
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, "r") as f:
            results = json.load(f)
            self.assertEqual(len(results), 1)
            result = results["1"]
            self.assertEqual(result["question_id"], 1)
            self.assertEqual(result["prediction"], "Some answer")

if __name__ == "__main__":
    unittest.main()

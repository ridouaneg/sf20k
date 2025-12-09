import unittest
import os
from sf20k.constants import DATA_DIR, VIDEO_DIR, WEIGHTS_DIR

class TestEnvironmentSetup(unittest.TestCase):
    """
    Checks that the local environment is correctly set up with 
    required directories and model weights.
    """

    def test_root_directories_exist(self):
        """Verify that the main data, video, and weights directories exist."""
        directories = {
            "DATA_DIR": DATA_DIR,
            "VIDEO_DIR": VIDEO_DIR,
            "WEIGHTS_DIR": WEIGHTS_DIR
        }

        for name, path in directories.items():
            with self.subTest(directory=name):
                self.assertTrue(
                    os.path.exists(path), 
                    f"{name} folder not found at: {path}"
                )

    def test_baseline_weights_exist(self):
        """Verify that all required baseline model weights exist in WEIGHTS_DIR."""
        
        # List of relative paths to check inside WEIGHTS_DIR
        model_paths = [
            # Qwen Models
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            
            # OpenGVLab Models
            "OpenGVLab/InternVL3_5-2B",
            "OpenGVLab/InternVL3_5-4B",
            "OpenGVLab/InternVL3_5-8B",
            
            # Other Baselines
            "lmms-lab/LongVA-7B-DPO",
            "Vision-CAIR/LongVU_Qwen2_7B",
            
            # Note: You listed these in comments but did not have assertions for them yet.
            # You can uncomment these lines once you have the paths ready:
            # "Llava-Video", 
            # "MovieChat", 
            # "video-salmonn-2+", 
            # "ma-lmm"

            # Metric Models
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

        for relative_path in model_paths:
            full_path = os.path.join(WEIGHTS_DIR, relative_path)
            
            # subTest ensures that if one weight is missing, 
            # the test continues checking the others before finishing.
            with self.subTest(model=relative_path):
                self.assertTrue(
                    os.path.exists(full_path), 
                    f"Model weights missing: {relative_path}\nExpected at: {full_path}"
                )

if __name__ == '__main__':
    unittest.main()
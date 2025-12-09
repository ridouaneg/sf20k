
import unittest
import sys
import os

from sf20k.prompts.mcqa_prompt import MCQAPrompt
from sf20k.prompts.oeqa_prompt import OEQAPrompt


class TestPrompts(unittest.TestCase):

    def test_oeqa_prompt(self):
        prompt = OEQAPrompt(modality="vision_language")
        sample = {'question': 'What happens?', 'answer': 'Something happens.'}
        sample['subtitles'] = "Subtitle text"
        
        query = prompt.get_query(sample)
        self.assertIn('What happens?', query)
        self.assertIn('Subtitle text', query)
        
        response = prompt.get_response(sample)
        self.assertEqual(response, 'Something happens.')
        
        processed = prompt.postprocess_response("  Response  ")
        self.assertEqual(processed, "Response")

    def test_mcqa_prompt(self):
        prompt = MCQAPrompt(modality="vision_language")
        sample = {
            'question': 'What happens?', 
            'options': ['A', 'B', 'C', 'D', 'E'],
            'answer_id': 1,
            'answer': 'B'
        }
        sample['subtitles'] = "Subtitle text"
        
        query = prompt.get_query(sample)
        self.assertIn('What happens?', query)
        self.assertIn('B. B', query) # Option B
        self.assertIn('Subtitle text', query)
        
        response = prompt.get_response(sample)
        self.assertEqual(response, 'B')
        
        processed = prompt.postprocess_response("  B. option text  ")
        self.assertEqual(processed, sample['answer_id'])


if __name__ == '__main__':
    unittest.main()
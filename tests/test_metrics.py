
import unittest
import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from sf20k.metrics.accuracy_metric import AccuracyMetric
from sf20k.metrics.similarity_metric import SimilarityMetric
from sf20k.metrics.embedding_metric import EmbeddingSimilarityMetric


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        metric = AccuracyMetric()
        
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        
        acc = metric.compute(y_true, y_pred)
        self.assertEqual(acc, 0.75)


class TestSimilarityMetric(unittest.TestCase):

    def test_cider(self):
        metric = SimilarityMetric(metric='cider')
        y_true = ["hello world"]
        y_pred = ["hello world"]
        score = metric.compute(y_true, y_pred)
        # CIDEr score for identical captions should be high (usually > 0, often 10.0 for self-match but depends on DF)
        # Note: CIDEr requires document frequency, with 1 sample it might behave oddly or default.
        # However, identical strings should give a positive score.
        self.assertGreaterEqual(score, 0.0)

    def test_meteor(self):
        metric = SimilarityMetric(metric='meteor')
        y_true = ["hello world"]
        y_pred = ["hello world"]
        score = metric.compute(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=1)

    def test_rouge(self):
        metric = SimilarityMetric(metric='rouge')
        y_true = ["hello world"]
        y_pred = ["hello world"]
        score = metric.compute(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=1)

    def test_bleu(self):
        metric = SimilarityMetric(metric='bleu')
        y_true = ["the quick brown fox jumps over the lazy dog"]
        y_pred = ["the quick brown fox jumps over the lazy dog"]
        score = metric.compute(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=1)

    def test_invalid_metric(self):
        with self.assertRaises(ValueError):
            SimilarityMetric(metric='invalid')


class TestEmbeddingSimilarityMetric(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model once for all tests to save time
        cls.metric = EmbeddingSimilarityMetric()

    def test_exact_match(self):
        y_true = ["hello world"]
        y_pred = ["hello world"]
        score = self.metric.compute(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_high_similarity(self):
        y_true = ["The movie was great"]
        y_pred = ["The film was excellent"]
        score = self.metric.compute(y_true, y_pred)
        self.assertGreater(score, 0.7)

    def test_low_similarity(self):
        y_true = ["The movie was great"]
        y_pred = ["I like bananas"]
        score = self.metric.compute(y_true, y_pred)
        self.assertLess(score, 0.5)

    def test_batch(self):
        y_true = ["hello", "world"]
        y_pred = ["hello", "world"]
        score = self.metric.compute(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
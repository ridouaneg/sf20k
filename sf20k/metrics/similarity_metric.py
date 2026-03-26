from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu


class SimilarityMetric:

    def __init__(self, metric='cider'):
        """
        Initialize the similarity metric.
        
        Args:
            metric (str): The metric to use. One of ['cider', 'meteor', 'rouge', 'bleu'].
                          Default is 'cider'.
        """
        self.metric_name = metric.lower()
        if self.metric_name == 'cider':
            self.scorer = Cider()
        elif self.metric_name == 'meteor':
            self.scorer = Meteor()
        elif self.metric_name == 'rouge':
            self.scorer = Rouge()
        elif self.metric_name == 'bleu':
            self.scorer = Bleu(4)
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported metrics: ['cider', 'meteor', 'rouge', 'bleu']")

    def compute(self, y_true, y_pred):
        """
        Compute the score between predictions and ground truths.
        
        Args:
            y_true (list[str]): List of ground truth strings.
            y_pred (list[str]): List of predicted strings.
            
        Returns:
            float: The computed score.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
            
        if len(y_true) == 0:
            return 0.0

        # Format for pycocoevalcap: {id: [caption]}
        gts = {i: [t] for i, t in enumerate(y_true)}
        res = {i: [p] for i, p in enumerate(y_pred)}

        score, scores = self.scorer.compute_score(gts, res)

        if self.metric_name == 'bleu':
            # Bleu returns a list of scores for N=1..4. We return Bleu-1.
            if isinstance(score, list):
                return score[0]
            return score
        
        return score
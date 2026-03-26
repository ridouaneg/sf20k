import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

from ..constants import WEIGHTS_DIR


class EmbeddingSimilarityMetric:

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(WEIGHTS_DIR, model_name))
        self.model = AutoModel.from_pretrained(os.path.join(WEIGHTS_DIR, model_name))
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute(self, y_true, y_pred):
        """
        Compute the average cosine similarity between predictions and ground truths.
        
        Args:
            y_true (list[str]): List of ground truth strings.
            y_pred (list[str]): List of predicted strings.
            
        Returns:
            float: Average cosine similarity.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
            
        if len(y_true) == 0:
            return 0.0

        # Tokenize sentences
        encoded_input_true = self.tokenizer(y_true, padding=True, truncation=True, return_tensors='pt')
        encoded_input_pred = self.tokenizer(y_pred, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output_true = self.model(**encoded_input_true)
            model_output_pred = self.model(**encoded_input_pred)

        # Perform pooling
        sentence_embeddings_true = self._mean_pooling(model_output_true, encoded_input_true['attention_mask'])
        sentence_embeddings_pred = self._mean_pooling(model_output_pred, encoded_input_pred['attention_mask'])

        # Normalize embeddings
        sentence_embeddings_true = torch.nn.functional.normalize(sentence_embeddings_true, p=2, dim=1)
        sentence_embeddings_pred = torch.nn.functional.normalize(sentence_embeddings_pred, p=2, dim=1)

        # Compute cosine similarity
        cosine_scores = torch.nn.functional.cosine_similarity(sentence_embeddings_true, sentence_embeddings_pred)
        
        return cosine_scores.mean().item()
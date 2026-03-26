class AccuracyMetric:
    
    def __init__(self):
        pass

    def compute(self, answer_id: int, pred_id: int):
        pred = 1 * (answer_id == pred_id)
        return pred
from torcheval.metrics import BinaryAccuracy, BinaryF1Score

ACCURACY = 'Accuracy'
F1SCORE = 'F1-Score'


class MetricsManager:

    def __init__(self):
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()

    def update(self, preds, labels):
        self.accuracy.update(preds, labels)
        self.f1_score.update(preds, labels)

    def compute(self):
        return {
            ACCURACY: self.accuracy.compute().item(),
            F1SCORE: self.f1_score.compute().item(),
        }

    def reset(self):
        self.accuracy.reset()
        self.f1_score.reset()

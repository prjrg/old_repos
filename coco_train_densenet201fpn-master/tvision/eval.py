"""
@Copyright 2019 VOID SOFTWARE, S.A.
"""
import numpy as np


class GlassEvalClasses:
    def __init__(self, score_tresh, n_classes):
        self.correct = {x: 0 for x in range(1, n_classes + 1)}
        self.false_positives = {x: 0 for x in range(1, n_classes + 1)}
        self.false_negatives = {x: 0 for x in range(1, n_classes + 1)}
        self.total = 0
        self.cutoff = score_tresh
        self.n_classes = n_classes

    def reset(self):
        self.correct = {x: 0 for x in range(1, self.n_classes + 1)}
        self.false_positives = {x: 0 for x in range(1, self.n_classes + 1)}
        self.false_negatives = {x: 0 for x in range(1, self.n_classes + 1)}
        self.total = 0

    def process(self, outputs, targets):
        self.total = self.total + 1
        relevant_annotations = targets["labels"].cpu().tolist()

        instances = outputs
        pred_scores = instances["scores"].cpu()
        pred_classes = instances["labels"].cpu()

        # predictions with a confidence score < cutoff
        to_keep = np.where(pred_scores >= self.cutoff)

        unique_gt = set(np.unique(relevant_annotations).tolist())
        unique_predictions = np.unique(pred_classes[to_keep])
        unique_predictions = set(unique_predictions.tolist())


        for i in range(1, self.n_classes+1):
            if i in unique_gt and i not in unique_predictions:
                self.false_negatives[i] = self.false_negatives[i] + 1
                continue
            elif i in unique_predictions and i not in unique_gt:
                self.false_positives[i] = self.false_positives[i] + 1
                continue
            self.correct[i] = self.correct[i] + 1

    def evaluate(self):
        for i in range(1, self.n_classes + 1):
            result = self.correct[i] / self.total

            print("Glasscan Class " + str(i) + " Accuracy Evaluation: {}".format(result))
            print("Glasscan Class " + str(i) + " False-Positives: {}".format(self.false_positives[i]))
            print("Glasscan Class " + str(i) + " False-Negatives: {}".format(self.false_negatives[i]))

        return {"average_per_class": 0}

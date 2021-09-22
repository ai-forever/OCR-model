import numpy as np


def get_accuracy(y_true, y_pred):
    """Calc accuracy between two list of strings."""
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score = np.mean(scores)
    return avg_score

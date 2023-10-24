import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(model_name)
    print(f"Sensitivity: {tp / (tp + fn) * 100:.1f}%")
    print(f"Specificity: {tn / (tn + fp) * 100:.1f}%")
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.1f}%")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred) * 100:.1f}%")
    print()

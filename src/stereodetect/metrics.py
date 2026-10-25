"""Task metrics, including invalid JSON predictions."""

from stereodetect.labels import LABEL_NAMES, LABELS


def metrics(truth, predicted):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = list(range(5))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(
            precision_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)
        ),
    }


def detailed_metrics(truth, predicted):
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )

    return {
        **metrics(truth, predicted),
        "classification_report": classification_report(
            truth, predicted, labels=list(range(5)), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(5))).tolist(),
    }


label_metrics = metrics


def detailed_report(truth, predicted):
    import numpy as np
    from sklearn.metrics import (
        classification_report,
    )

    matrix = np.zeros((5, 6), dtype=int)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess if guess != -1 else 5] += 1
    return {
        **label_metrics(truth, predicted),
        "classification_report": classification_report(
            truth, predicted, labels=list(range(5)), output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=gold labels 0-4; columns=predicted labels 0-4, invalid",
        "invalid_predictions": sum(p == -1 for p in predicted),
    }


def calculate_metrics(truth, predictions):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(truth, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predictions, labels=LABELS, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=LABELS,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
    }

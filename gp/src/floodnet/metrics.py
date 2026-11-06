"""One scoring implementation for both practical pipelines."""
import numpy as np
from .common import CLASSES, ProjectError


def confusion(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ProjectError("Metric shapes differ")
    valid = truth != 255
    actual, predicted = truth[valid], prediction[valid]
    if np.any((actual < 0) | (actual > 2)) or np.any((predicted < 0) | (predicted > 2)):
        raise ProjectError("Metrics require class IDs 0..2 (truth padding=255)")
    return np.bincount(actual.astype(np.int64) * 3 + predicted.astype(np.int64), minlength=9).reshape(3, 3).astype(np.int64)


def score(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (3, 3) or np.any(matrix < 0):
        raise ProjectError("Expected a nonnegative 3x3 confusion matrix")
    tp = np.diag(matrix)
    support, predicted = matrix.sum(1), matrix.sum(0)
    def ratio(a, b):
        return float(a / b) if b else None
    per_class = []
    for c, name in enumerate(CLASSES):
        per_class.append({"id": c, "name": name, "support": int(support[c]), "predicted_pixels": int(predicted[c]),
                          "iou": ratio(tp[c], support[c] + predicted[c] - tp[c]),
                          "dice": ratio(2 * tp[c], support[c] + predicted[c]), "recall": ratio(tp[c], support[c])})
    defined = [p["iou"] for p in per_class if p["iou"] is not None]
    flooded = [p["iou"] for p in per_class[1:] if p["iou"] is not None]
    return {"confusion_matrix": matrix.tolist(), "valid_pixels": int(matrix.sum()), "per_class": per_class,
            "miou": float(np.mean(defined)) if defined else None,
            "flooded_miou": float(np.mean(flooded)) if flooded else None,
            "defined_flooded_classes": len(flooded)}

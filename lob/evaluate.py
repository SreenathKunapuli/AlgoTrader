"""Classification evaluation for LOB models."""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["down", "flat", "up"]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    report = {"confusion_matrix": cm.tolist(), "per_class": {}}
    f1s = []
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1s.append(f1)
        report["per_class"][name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }
    report["accuracy"] = round(float((y_true == y_pred).mean()), 4)
    report["macro_f1"] = round(float(np.mean(f1s)), 4)
    return report


def format_report(report: dict) -> str:
    lines = [f"{'class':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'support':>8}"]
    for name, m in report["per_class"].items():
        lines.append(
            f"{name:>6}  {m['precision']:6.3f}  {m['recall']:6.3f}  "
            f"{m['f1']:6.3f}  {m['support']:8d}"
        )
    lines.append(f"accuracy {report['accuracy']:.4f}   macro-F1 {report['macro_f1']:.4f}")
    cm = np.array(report["confusion_matrix"])
    lines.append("confusion (rows=true, cols=pred):")
    for row in cm:
        lines.append("   " + "  ".join(f"{v:8d}" for v in row))
    return "\n".join(lines)

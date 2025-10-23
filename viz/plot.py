from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(labels: Iterable[str], out_path: str | Path | None = None):
    s = pd.Series(list(labels)).value_counts()
    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def plot_text_length_distribution(texts: Iterable[str], out_path: str | Path | None = None):
    lengths = [len(str(t)) for t in texts]
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=30)
    ax.set_title("Text length distribution")
    ax.set_xlabel("length")
    ax.set_ylabel("count")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, out_path: str | Path | None = None):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


if __name__ == "__main__":
    # quick smoke test
    plot_class_distribution(["spam", "ham", "ham", "spam"], out_path="../viz/out/class_dist.png")
    plot_text_length_distribution(["hello", "this is a longer sms", "ok"], out_path="../viz/out/len.png")

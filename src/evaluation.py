"""Shared evaluation and plotting utilities for HMEQ model scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def print_section(title: str) -> None:
    """Print a readable terminal section heading."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_table(table: pd.DataFrame, title: str | None = None) -> None:
    """Print a DataFrame in a terminal-friendly format."""
    if title:
        print(f"\n{title}")
    print(table.to_string())


def calculate_metrics(y_true, y_pred, y_proba, *, include_average_precision: bool = False) -> dict[str, float]:
    """Calculate default-focused binary-classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_BAD_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_BAD_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_BAD_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }
    if include_average_precision:
        metrics["average_precision"] = average_precision_score(y_true, y_proba)
    return metrics


def evaluate_pipeline(
    model_name: str,
    pipeline,
    X,
    y,
    split: str,
    threshold: float = 0.5,
    *,
    include_average_precision: bool = False,
) -> dict[str, float | str]:
    """Evaluate a fitted pipeline at a specified probability threshold."""
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        **calculate_metrics(y, y_pred, y_proba, include_average_precision=include_average_precision),
    }


def threshold_analysis(
    y_true,
    y_proba,
    thresholds: np.ndarray | None = None,
    *,
    include_average_precision: bool = False,
) -> pd.DataFrame:
    """Evaluate threshold trade-offs using training data only."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.005), 3)

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        row = calculate_metrics(
            y_true,
            y_pred,
            y_proba,
            include_average_precision=include_average_precision,
        )
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def select_threshold(threshold_table: pd.DataFrame, criterion: str = "f1_BAD_1") -> float:
    """Select the threshold that maximises a training-set criterion."""
    best_idx = threshold_table[criterion].idxmax()
    return float(threshold_table.loc[best_idx, "threshold"])


def save_table(table: pd.DataFrame, output_dir: Path, filename: str, index: bool = True) -> None:
    """Save a table as CSV."""
    path = output_dir / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def finish_plot(filename: str, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    """Apply layout, optionally save, optionally show, and close the current figure."""
    plt.tight_layout()
    if save_outputs and output_dir is not None:
        path = output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    if show_plots:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    filename: str,
    output_dir: Path | None,
    save_outputs: bool,
    show_plots: bool,
) -> None:
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    finish_plot(filename, output_dir, save_outputs, show_plots)

"""Expected-loss and lending decision trade-off analysis for HMEQ models.

This script compares the tuned project models using an economic decision-cost
framework. BAD = 1 is default/severe delinquency. A false negative is therefore
an approved borrower who defaults; a false positive is a rejected borrower who
would have repaid.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from .preprocessing import resolve_data_path, resolve_output_dir
except ImportError:
    from preprocessing import resolve_data_path, resolve_output_dir


TARGET = "BAD"
EAD_COL = "LOAN"
DEFAULT_RECOVERY_RATE = 0.40
DEFAULT_INTEREST_RATE = 0.04


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_table(table: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
    print(table.to_string(index=False))


def save_table(table: pd.DataFrame, output_dir: Path, filename: str, index: bool = False) -> None:
    path = output_dir / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def finish_plot(filename: str, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    plt.tight_layout()
    if save_outputs and output_dir is not None:
        path = output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    if show_plots:
        plt.show()
    plt.close()


def load_data_split(data_path: Path, test_size: float, random_state: int):
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found in {data_path}")
    if EAD_COL not in df.columns:
        raise ValueError(f"Exposure column '{EAD_COL}' was not found in {data_path}")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def collect_model_probabilities(
    data_path: Path,
    X_test: pd.DataFrame,
    test_size: float,
    random_state: int,
    suppress_output: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    """Fit existing tuned model scripts one at a time and collect test probabilities."""
    try:
        from .project_logit import run_logistic_regression_analysis
        from .project_tree import run_decision_tree_analysis
        from .project_rf import run_random_forest_analysis
        from .project_gb import run_gradient_boosting_analysis
        from .project_ada import run_adaboost_analysis
        from .project_xg import run_xgboost_analysis
    except ImportError:
        from project_logit import run_logistic_regression_analysis
        from project_tree import run_decision_tree_analysis
        from project_rf import run_random_forest_analysis
        from project_gb import run_gradient_boosting_analysis
        from project_ada import run_adaboost_analysis
        from project_xg import run_xgboost_analysis

    runners = [
        ("Logistic Regression", run_logistic_regression_analysis, "improved_pipeline"),
        ("Decision Tree", run_decision_tree_analysis, "tuned_pipeline"),
        ("Random Forest", run_random_forest_analysis, "tuned_pipeline"),
        ("Gradient Boosting", run_gradient_boosting_analysis, "tuned_pipeline"),
        ("AdaBoost", run_adaboost_analysis, "tuned_pipeline"),
        ("XGBoost", run_xgboost_analysis, "tuned_pipeline"),
    ]

    probability_by_model: dict[str, np.ndarray] = {}
    selected_thresholds: dict[str, float] = {}
    model_metadata: dict[str, Any] = {}
    for model_name, runner, pipeline_key in runners:
        print(f"Fitting/loading {model_name}...")
        kwargs = {
            "data_path": data_path,
            "test_size": test_size,
            "random_state": random_state,
            "save_outputs": False,
            "show_plots": False,
        }
        if suppress_output:
            buffer = StringIO()
            with redirect_stdout(buffer):
                result = runner(**kwargs)
        else:
            result = runner(**kwargs)

        if result.get("status") == "missing_dependency":
            print(f"Skipping {model_name}: missing dependency {result.get('required_package')}.")
            continue

        pipeline = result.get(pipeline_key)
        if pipeline is None:
            print(f"Skipping {model_name}: fitted pipeline was not returned.")
            continue

        probability_by_model[model_name] = pipeline.predict_proba(X_test)[:, 1]
        selected_thresholds[model_name] = float(result.get("selected_threshold", 0.5))
        model_metadata[model_name] = {
            "model_selected_threshold": selected_thresholds[model_name],
            "best_params": result.get("best_params") or result.get("best_parameters"),
            "best_cv_score": result.get("best_cv_score")
            or result.get("best_cv_roc_auc")
            or result.get("best_score"),
            "status": result.get("status", "success"),
        }

        del pipeline
        del result
        gc.collect()

    return probability_by_model, selected_thresholds, model_metadata


def expected_loss_components(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    exposure: pd.Series | np.ndarray,
    recovery_rate: float,
    interest_rate: float,
) -> dict[str, float]:
    """Compute decision costs: FN default loss and FP missed interest."""
    y_true_arr = np.asarray(y_true)
    exposure_arr = np.asarray(exposure, dtype=float)

    false_negative_mask = (y_true_arr == 1) & (y_pred == 0)
    false_positive_mask = (y_true_arr == 0) & (y_pred == 1)

    loss_given_default = 1.0 - recovery_rate
    false_negative_loss = float((exposure_arr[false_negative_mask] * loss_given_default).sum())
    false_positive_loss = float((exposure_arr[false_positive_mask] * interest_rate).sum())
    total_loss = false_negative_loss + false_positive_loss

    return {
        "false_negative_loss": false_negative_loss,
        "false_positive_loss": false_positive_loss,
        "total_expected_loss": total_loss,
        "average_expected_loss": total_loss / len(y_true_arr),
        "loss_given_default": loss_given_default,
        "false_negative_loan_value": float(exposure_arr[false_negative_mask].sum()),
        "false_positive_loan_value": float(exposure_arr[false_positive_mask].sum()),
    }


def predictive_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_BAD_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_BAD_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_BAD_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_model_at_threshold(
    model_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    exposure: pd.Series,
    threshold: float,
    recovery_rate: float,
    interest_rate: float,
    threshold_type: str,
) -> dict[str, float | str]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "recovery_rate": recovery_rate,
        "interest_rate": interest_rate,
        **predictive_metrics(y_true, y_pred, y_proba),
        **expected_loss_components(y_true, y_pred, exposure, recovery_rate, interest_rate),
    }


def build_threshold_grid(
    model_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    exposure: pd.Series,
    recovery_rate: float,
    interest_rate: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.991, 0.01), 2)
    rows = [
        evaluate_model_at_threshold(
            model_name,
            y_true,
            y_proba,
            exposure,
            float(threshold),
            recovery_rate,
            interest_rate,
            "threshold_grid",
        )
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def make_confusion_matrix_table(y_true, y_proba, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_name, threshold in thresholds.items():
        y_pred = (y_proba[model_name] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "model": model_name,
            "threshold": threshold,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })
    return pd.DataFrame(rows)


def robustness_analysis(
    y_true: pd.Series,
    probability_by_model: dict[str, np.ndarray],
    exposure: pd.Series,
    threshold_by_model: dict[str, float],
    recovery_rates: np.ndarray | None = None,
    interest_rates: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recovery_rates is None:
        recovery_rates = np.round(np.arange(0.20, 0.601, 0.10), 2)
    if interest_rates is None:
        interest_rates = np.round(np.arange(0.02, 0.081, 0.01), 2)

    rows = []
    for recovery_rate in recovery_rates:
        for interest_rate in interest_rates:
            for model_name, y_proba in probability_by_model.items():
                threshold = threshold_by_model[model_name]
                y_pred = (y_proba >= threshold).astype(int)
                row = {
                    "model": model_name,
                    "threshold": threshold,
                    "recovery_rate": recovery_rate,
                    "interest_rate": interest_rate,
                    **expected_loss_components(y_true, y_pred, exposure, recovery_rate, interest_rate),
                }
                rows.append(row)

    robustness = pd.DataFrame(rows)
    winners = (
        robustness.loc[robustness.groupby(["recovery_rate", "interest_rate"])["total_expected_loss"].idxmin()]
        .sort_values(["recovery_rate", "interest_rate"])
        .reset_index(drop=True)
    )
    return robustness, winners


def plot_expected_loss_bars(table: pd.DataFrame, title: str, filename: str, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    plot_table = table.sort_values("total_expected_loss")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_table, x="total_expected_loss", y="model", color="#4472c4", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Total expected loss")
    ax.set_ylabel("Model")
    ax.ticklabel_format(axis="x", style="plain")
    finish_plot(filename, output_dir, save_outputs, show_plots)


def plot_threshold_loss(threshold_table: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(data=threshold_table, x="threshold", y="total_expected_loss", hue="model", ax=ax)
    ax.set_title("Expected Loss Across Classification Thresholds")
    ax.set_xlabel("Threshold for predicting BAD = 1")
    ax.set_ylabel("Total expected loss")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_plot("loss_by_threshold.png", output_dir, save_outputs, show_plots)


def plot_winner_heatmap(winners: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    pivot = winners.pivot(index="recovery_rate", columns="interest_rate", values="model")
    pivot = pivot.sort_index().sort_index(axis=1)
    codes, labels = pd.factorize(pivot.values.ravel(), sort=True)
    matrix = codes.reshape(pivot.shape)
    cmap = plt.get_cmap("tab20", len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(len(labels) - 1, 0), aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{value:.0%}" for value in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{value:.0%}" for value in pivot.index])
    ax.set_xlabel("Opportunity cost / interest rate")
    ax.set_ylabel("Recovery rate")
    ax.set_title("Model With Minimum Expected Loss")

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i), label=str(label)) for i, label in enumerate(labels)]
    ax.legend(handles=handles, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_plot("robustness_best_model_heatmap.png", output_dir, save_outputs, show_plots)


def run_expected_loss_analysis(
    data_path=None,
    output_dir=None,
    test_size=0.30,
    random_state=42,
    recovery_rate=DEFAULT_RECOVERY_RATE,
    interest_rate=DEFAULT_INTEREST_RATE,
    save_outputs=True,
    show_plots=True,
    suppress_model_output=True,
) -> dict[str, Any]:
    """Run expected-loss analysis for all available tuned project models."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    resolved_data_path = resolve_data_path(data_path)
    resolved_output_dir = resolve_output_dir(output_dir, "outputs_loss")

    if save_outputs:
        resolved_output_dir.mkdir(exist_ok=True)
    else:
        resolved_output_dir = None

    sns.set_theme(style="whitegrid", context="notebook")
    pd.options.display.float_format = "{:,.4f}".format

    print_section("Expected-Loss Framework")
    print(
        "BAD = 1 is the default class. A false negative means the bank approves a borrower who "
        "defaults, costing the unrecovered share of the loan. A false positive means the bank "
        "rejects a borrower who would have repaid, costing missed interest income."
    )
    print(
        f"Baseline assumptions: recovery rate = {recovery_rate:.0%}, "
        f"loss-given-default = {1 - recovery_rate:.0%}, opportunity cost = {interest_rate:.0%}."
    )

    X_train, X_test, y_train, y_test = load_data_split(resolved_data_path, test_size, random_state)
    exposure_test = X_test[EAD_COL]
    split_info = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "test_default_rate": y_test.mean(),
        "total_test_loan_value": float(exposure_test.sum()),
    }
    print_table(pd.DataFrame([split_info]).round(4), "Shared test split")

    print_section("Fit Existing Tuned Models")
    probability_by_model, model_selected_thresholds, model_metadata = collect_model_probabilities(
        resolved_data_path,
        X_test,
        test_size=test_size,
        random_state=random_state,
        suppress_output=suppress_model_output,
    )
    if not probability_by_model:
        raise RuntimeError("No fitted models were available for expected-loss analysis.")

    print_section("Baseline Expected Loss at 0.50 Threshold")
    baseline_rows = [
        evaluate_model_at_threshold(
            model_name,
            y_test,
            y_proba,
            exposure_test,
            threshold=0.50,
            recovery_rate=recovery_rate,
            interest_rate=interest_rate,
            threshold_type="default_0.50",
        )
        for model_name, y_proba in probability_by_model.items()
    ]
    baseline_loss = pd.DataFrame(baseline_rows).sort_values("total_expected_loss").reset_index(drop=True)
    best_baseline_loss = baseline_loss["total_expected_loss"].min()
    baseline_loss["relative_to_best"] = baseline_loss["total_expected_loss"] / best_baseline_loss
    print_table(baseline_loss.round(4), "Baseline expected-loss comparison")

    print_section("Threshold Optimisation")
    threshold_tables = []
    threshold_summary_rows = []
    optimal_thresholds = {}
    for model_name, y_proba in probability_by_model.items():
        table = build_threshold_grid(model_name, y_test, y_proba, exposure_test, recovery_rate, interest_rate)
        threshold_tables.append(table)
        best_row = table.loc[table["total_expected_loss"].idxmin()].to_dict()
        default_row = baseline_loss.loc[baseline_loss["model"].eq(model_name)].iloc[0].to_dict()
        optimal_thresholds[model_name] = float(best_row["threshold"])
        threshold_summary_rows.append({
            "model": model_name,
            "default_threshold": 0.50,
            "best_threshold": best_row["threshold"],
            "default_expected_loss": default_row["total_expected_loss"],
            "optimised_expected_loss": best_row["total_expected_loss"],
            "improvement": default_row["total_expected_loss"] - best_row["total_expected_loss"],
            "improvement_pct": (default_row["total_expected_loss"] - best_row["total_expected_loss"]) / default_row["total_expected_loss"],
            "false_negatives_at_best_threshold": best_row["false_negatives"],
            "false_positives_at_best_threshold": best_row["false_positives"],
            "precision_BAD_1_at_best_threshold": best_row["precision_BAD_1"],
            "recall_BAD_1_at_best_threshold": best_row["recall_BAD_1"],
            "f1_BAD_1_at_best_threshold": best_row["f1_BAD_1"],
        })

    threshold_loss = pd.concat(threshold_tables, ignore_index=True)
    threshold_summary = pd.DataFrame(threshold_summary_rows).sort_values("optimised_expected_loss").reset_index(drop=True)
    best_optimised_loss = threshold_summary["optimised_expected_loss"].min()
    threshold_summary["relative_to_best_optimised"] = threshold_summary["optimised_expected_loss"] / best_optimised_loss
    print_table(threshold_summary.round(4), "Threshold optimisation summary")

    optimised_rows = [
        evaluate_model_at_threshold(
            model_name,
            y_test,
            probability_by_model[model_name],
            exposure_test,
            threshold=optimal_thresholds[model_name],
            recovery_rate=recovery_rate,
            interest_rate=interest_rate,
            threshold_type="loss_minimising_threshold",
        )
        for model_name in optimal_thresholds
    ]
    optimised_loss = pd.DataFrame(optimised_rows).sort_values("total_expected_loss").reset_index(drop=True)
    optimised_loss["relative_to_best"] = optimised_loss["total_expected_loss"] / optimised_loss["total_expected_loss"].min()
    print_table(optimised_loss.round(4), "Expected loss at loss-minimising thresholds")

    default_thresholds = {model_name: 0.50 for model_name in probability_by_model}
    baseline_confusion = make_confusion_matrix_table(y_test, probability_by_model, default_thresholds)
    optimised_confusion = make_confusion_matrix_table(y_test, probability_by_model, optimal_thresholds)

    print_section("Robustness Analysis")
    robustness, winners = robustness_analysis(y_test, probability_by_model, exposure_test, optimal_thresholds)
    winner_counts = winners["model"].value_counts().rename_axis("model").reset_index(name="winning_grid_cells")
    print_table(winner_counts, "Best-model frequency across recovery/interest grid")
    print_table(winners.round(4), "Best model under each assumption combination")

    if save_outputs and resolved_output_dir is not None:
        save_table(pd.DataFrame([split_info]), resolved_output_dir, "loss_data_split_info.csv")
        save_table(baseline_loss, resolved_output_dir, "loss_baseline_expected_loss.csv")
        save_table(optimised_loss, resolved_output_dir, "loss_optimised_expected_loss.csv")
        save_table(threshold_loss, resolved_output_dir, "loss_threshold_grid.csv")
        save_table(threshold_summary, resolved_output_dir, "loss_threshold_optimisation_summary.csv")
        save_table(baseline_confusion, resolved_output_dir, "loss_baseline_confusion_matrices.csv")
        save_table(optimised_confusion, resolved_output_dir, "loss_optimised_confusion_matrices.csv")
        save_table(robustness, resolved_output_dir, "loss_robustness_table.csv")
        save_table(winners, resolved_output_dir, "loss_robustness_best_model_by_assumption.csv")
        save_table(winner_counts, resolved_output_dir, "loss_robustness_winner_counts.csv")

    print_section("Plots")
    plot_expected_loss_bars(
        baseline_loss,
        "Expected Loss at Default 0.50 Threshold",
        "loss_baseline_expected_loss.png",
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_expected_loss_bars(
        optimised_loss,
        "Expected Loss After Threshold Optimisation",
        "loss_optimised_expected_loss.png",
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_threshold_loss(threshold_loss, resolved_output_dir, save_outputs, show_plots)
    plot_winner_heatmap(winners, resolved_output_dir, save_outputs, show_plots)

    if save_outputs and resolved_output_dir is not None:
        print(f"\nAll expected-loss outputs saved to: {resolved_output_dir}")

    return {
        "model_metadata": model_metadata,
        "model_selected_thresholds": model_selected_thresholds,
        "probability_by_model": probability_by_model,
        "baseline_loss": baseline_loss,
        "optimised_loss": optimised_loss,
        "threshold_grid": threshold_loss,
        "threshold_summary": threshold_summary,
        "baseline_confusion_matrices": baseline_confusion,
        "optimised_confusion_matrices": optimised_confusion,
        "robustness_table": robustness,
        "robustness_winners": winners,
        "winner_counts": winner_counts,
        "split_info": split_info,
        "output_dir": resolved_output_dir,
    }


if __name__ == "__main__":
    run_expected_loss_analysis()


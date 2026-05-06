"""Gradient Boosting model for HMEQ loan default prediction.

This module builds a GradientBoostingClassifier for BAD while keeping all
preprocessing leakage-free through sklearn Pipeline and ColumnTransformer.
It intentionally excludes Random Forest, XGBoost, neural networks, and all
other models.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "BAD"
CATEGORICAL_PREDICTORS = ["REASON", "JOB"]
MISSING_INDICATOR_COLUMNS = ["VALUE", "MORTDUE", "DEBTINC"]

try:
    from .preprocessing import build_preprocessor, clean_feature_name, resolve_data_path, resolve_output_dir, get_feature_names, resolve_data_path, resolve_output_dir
except ImportError:
    from preprocessing import build_preprocessor, clean_feature_name, resolve_data_path, resolve_output_dir, get_feature_names, resolve_data_path, resolve_output_dir

try:
    from .evaluation import (
        calculate_metrics,
        evaluate_pipeline,
        finish_plot,
        plot_confusion_matrix,
        print_section,
        print_table,
        save_table,
        select_threshold,
        threshold_analysis,
    )
except ImportError:
    from evaluation import (
        calculate_metrics,
        evaluate_pipeline,
        finish_plot,
        plot_confusion_matrix,
        print_section,
        print_table,
        save_table,
        select_threshold,
        threshold_analysis,
    )


def build_gb_pipeline(
    numerical_predictors: list[str],
    categorical_predictors: list[str],
    model: GradientBoostingClassifier,
) -> Pipeline:
    """Combine preprocessing and a Gradient Boosting classifier."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(numerical_predictors, categorical_predictors)),
        ("gb", model),
    ])


def extract_feature_importance_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Extract Gradient Boosting impurity-based feature importances."""
    feature_names = get_feature_names(fitted_pipeline)
    importances = fitted_pipeline.named_steps["gb"].feature_importances_
    table = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_permutation_importance_table(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Compute permutation importance on the held-out test set."""
    result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": X_test.columns,
        "permutation_importance_mean": result.importances_mean,
        "permutation_importance_std": result.importances_std,
    })
    return table.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def plot_roc_curve(y_true, model_probabilities: dict[str, np.ndarray], output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    """Plot ROC curves for baseline and tuned Gradient Boosting models."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, y_proba in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random baseline")
    ax.set_title("Gradient Boosting ROC Curve - Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / recall")
    ax.legend(loc="lower right")
    finish_plot("gb_roc_curve.png", output_dir, save_outputs, show_plots)


def plot_precision_recall_curve(y_true, model_probabilities: dict[str, np.ndarray], output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    """Plot precision-recall curves for baseline and tuned Gradient Boosting models."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    baseline_rate = float(np.mean(y_true))
    for label, y_proba in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(recall, precision, label=label)
    ax.axhline(baseline_rate, linestyle="--", color="#888888", label=f"Default rate = {baseline_rate:.3f}")
    ax.set_title("Gradient Boosting Precision-Recall Curve - Test Set")
    ax.set_xlabel("Recall for BAD = 1")
    ax.set_ylabel("Precision for BAD = 1")
    ax.legend(loc="best")
    finish_plot("gb_precision_recall_curve.png", output_dir, save_outputs, show_plots)


def plot_feature_importances(feature_importances: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool, top_n: int = 20) -> None:
    """Plot top impurity-based feature importances."""
    plot_data = feature_importances.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_data["feature"], plot_data["importance"], color="#4C78A8")
    ax.set_title(f"Top {top_n} Gradient Boosting Feature Importances")
    ax.set_xlabel("Impurity-based feature importance")
    ax.set_ylabel("Feature")
    finish_plot("tuned_gb_feature_importances.png", output_dir, save_outputs, show_plots)


def plot_permutation_importances(permutation_table: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool, top_n: int = 12) -> None:
    """Plot top permutation importances computed on raw input features."""
    plot_data = permutation_table.head(top_n).sort_values("permutation_importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_data["feature"], plot_data["permutation_importance_mean"], color="#F58518")
    ax.set_title(f"Top {top_n} Gradient Boosting Permutation Importances")
    ax.set_xlabel("Mean ROC-AUC decrease after permutation")
    ax.set_ylabel("Raw feature")
    finish_plot("tuned_gb_permutation_importances.png", output_dir, save_outputs, show_plots)


def load_previous_model_summaries(project_dir: Path) -> pd.DataFrame:
    """Load existing model summary CSVs when available for high-level comparison."""
    rows = []

    logit_path = project_dir / "outputs" / "outputs_logit" / "improved_model_metrics.csv"
    if logit_path.exists():
        logit = pd.read_csv(logit_path, index_col=0)
        if "test" in logit.index:
            row = logit.loc["test"].to_dict()
            row.update({"model": "Logistic Regression"})
            rows.append(row)

    tree_path = project_dir / "outputs" / "outputs_tree" / "tree_model_comparison.csv"
    if tree_path.exists():
        tree = pd.read_csv(tree_path)
        selected = tree[(tree["model"] == "tuned_tree_selected_threshold") & (tree["split"] == "test")]
        if not selected.empty:
            row = selected.iloc[0].to_dict()
            row.update({"model": "Decision Tree"})
            rows.append(row)

    rf_path = project_dir / "outputs" / "outputs_rf" / "rf_model_comparison.csv"
    if rf_path.exists():
        rf = pd.read_csv(rf_path)
        selected = rf[(rf["model"] == "tuned_rf_selected_threshold") & (rf["split"] == "test")]
        if not selected.empty:
            row = selected.iloc[0].to_dict()
            row.update({"model": "Random Forest"})
            rows.append(row)

    return pd.DataFrame(rows)


def run_gradient_boosting_analysis(
    data_path=None,
    output_dir=None,
    test_size=0.30,
    random_state=42,
    save_outputs=True,
    show_plots=True,
):
    """Run baseline and tuned Gradient Boosting analyses for HMEQ default prediction."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    resolved_data_path = resolve_data_path(data_path)
    resolved_output_dir = resolve_output_dir(output_dir, "outputs_gb")

    if save_outputs:
        resolved_output_dir.mkdir(exist_ok=True)
    else:
        resolved_output_dir = None

    pd.options.display.float_format = "{:,.4f}".format
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", None)
    sns.set_theme(style="whitegrid", context="notebook")

    if not resolved_data_path.exists():
        raise FileNotFoundError(f"Could not find hmeq.csv at {resolved_data_path}")

    print_section("Original Notebook Reference")
    print(
        "project.py was not found in this folder, so project.ipynb was used as the reference. "
        "The original notebook used GradientBoostingClassifier with n_estimators=200, "
        "learning_rate=0.05, max_depth=3, subsample=0.8, then RandomizedSearchCV scored by ROC-AUC "
        "over n_estimators, learning_rate, max_depth, subsample, and min_samples_leaf."
    )
    print(
        "This script keeps the same modelling logic but moves imputation, one-hot encoding, and "
        "missingness indicators into a leakage-free Pipeline fitted only on the training data. "
        "Missing REASON/JOB values are retained as an explicit 'Missing' category because missing "
        "application information may be predictively informative."
    )

    print_section("Load Data")
    raw_data = pd.read_csv(resolved_data_path)
    y = raw_data[TARGET].astype(int)
    X = raw_data.drop(columns=[TARGET])
    categorical_predictors = [col for col in CATEGORICAL_PREDICTORS if col in X.columns]
    numerical_predictors = [col for col in X.columns if col not in categorical_predictors]

    print(f"Loaded dataset: {raw_data.shape[0]:,} rows x {raw_data.shape[1]:,} columns")
    print("Numerical predictors:", numerical_predictors)
    print("Categorical predictors:", categorical_predictors)

    print_section("Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    split_info = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "train_default_rate": float(y_train.mean()),
        "test_default_rate": float(y_test.mean()),
    }
    print_table(pd.DataFrame([split_info]), "Split information")

    print_section("Fit Baseline Gradient Boosting")
    baseline_gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
    )
    baseline_pipeline = build_gb_pipeline(numerical_predictors, categorical_predictors, baseline_gb)
    baseline_pipeline.fit(X_train, y_train)

    print_section("Tune Gradient Boosting")
    gb_base = GradientBoostingClassifier(random_state=random_state)
    tuned_template = build_gb_pipeline(numerical_predictors, categorical_predictors, gb_base)
    param_dist = {
        "gb__n_estimators": [150, 200, 250, 300, 400],
        "gb__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.10, 0.15],
        "gb__max_depth": [2, 3, 4, 5],
        "gb__min_samples_split": [2, 10, 25],
        "gb__min_samples_leaf": [1, 3, 5, 10],
        "gb__subsample": [0.7, 0.8, 0.9, 1.0],
        "gb__max_features": [None, "sqrt", "log2", 0.5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        tuned_template,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    tuned_pipeline = search.best_estimator_

    print("Tuning metric: ROC-AUC, chosen to evaluate ranking ability across thresholds on training folds.")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    print_section("Threshold Analysis")
    oof_train_proba = cross_val_predict(
        tuned_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold_table = threshold_analysis(y_train, oof_train_proba, include_average_precision=True)
    selected_threshold = select_threshold(threshold_table, criterion="f1_BAD_1")
    selected_row = threshold_table.loc[threshold_table["threshold"].eq(selected_threshold)].iloc[0]
    print_table(threshold_table.sort_values("f1_BAD_1", ascending=False).head(10).round(4), "Top out-of-fold training thresholds by F1 for BAD = 1")
    print(
        f"\nSelected threshold: {selected_threshold:.3f}. "
        "It was chosen by maximising out-of-fold training F1 for BAD = 1, not by optimising the test set."
    )
    print(
        f"At this out-of-fold threshold, precision={selected_row['precision_BAD_1']:.3f}, "
        f"recall={selected_row['recall_BAD_1']:.3f}, false negatives={int(selected_row['false_negatives'])}, "
        f"false positives={int(selected_row['false_positives'])}."
    )

    print_section("Model Evaluation")
    rows = [
        evaluate_pipeline("baseline_gradient_boosting", baseline_pipeline, X_train, y_train, "train", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("baseline_gradient_boosting", baseline_pipeline, X_test, y_test, "test", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("tuned_gb_default_threshold", tuned_pipeline, X_train, y_train, "train", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("tuned_gb_default_threshold", tuned_pipeline, X_test, y_test, "test", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("tuned_gb_selected_threshold", tuned_pipeline, X_train, y_train, "train", threshold=selected_threshold, include_average_precision=True),
        evaluate_pipeline("tuned_gb_selected_threshold", tuned_pipeline, X_test, y_test, "test", threshold=selected_threshold, include_average_precision=True),
    ]
    comparison_table = pd.DataFrame(rows)
    metric_cols = ["accuracy", "precision_BAD_1", "recall_BAD_1", "f1_BAD_1", "roc_auc", "average_precision"]
    comparison_table[metric_cols] = comparison_table[metric_cols].round(4)
    print_table(comparison_table, "Gradient Boosting model comparison")
    print(
        "\nAccuracy alone is not sufficient because BAD = 1 is the minority class. "
        "Recall for BAD = 1 and false negatives matter because missed likely defaulters are costly. "
        "The train-test gap helps flag overfitting."
    )

    baseline_test_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_test_pred = (baseline_test_proba >= 0.5).astype(int)
    tuned_test_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
    tuned_test_pred = (tuned_test_proba >= selected_threshold).astype(int)

    baseline_cm = pd.DataFrame(
        confusion_matrix(y_test, baseline_test_pred, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    tuned_cm = pd.DataFrame(
        confusion_matrix(y_test, tuned_test_pred, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    print_table(baseline_cm, "Baseline test confusion matrix")
    print_table(tuned_cm, "Tuned test confusion matrix at selected threshold")

    baseline_train_metrics = comparison_table[(comparison_table["model"] == "baseline_gradient_boosting") & (comparison_table["split"] == "train")].iloc[0]
    baseline_test_metrics = comparison_table[(comparison_table["model"] == "baseline_gradient_boosting") & (comparison_table["split"] == "test")].iloc[0]
    tuned_train_metrics = comparison_table[(comparison_table["model"] == "tuned_gb_selected_threshold") & (comparison_table["split"] == "train")].iloc[0]
    tuned_test_metrics = comparison_table[(comparison_table["model"] == "tuned_gb_selected_threshold") & (comparison_table["split"] == "test")].iloc[0]
    print(
        f"\nOverfitting check: baseline train/test F1 gap = "
        f"{baseline_train_metrics['f1_BAD_1'] - baseline_test_metrics['f1_BAD_1']:.3f}; "
        f"tuned train/test F1 gap = {tuned_train_metrics['f1_BAD_1'] - tuned_test_metrics['f1_BAD_1']:.3f}."
    )

    train_report = pd.DataFrame(
        classification_report(
            y_train,
            (tuned_pipeline.predict_proba(X_train)[:, 1] >= selected_threshold).astype(int),
            output_dict=True,
            zero_division=0,
        )
    ).T
    test_report = pd.DataFrame(classification_report(y_test, tuned_test_pred, output_dict=True, zero_division=0)).T
    print_table(test_report.round(4), "Tuned Gradient Boosting test classification report")

    print_section("Comparison With Previous Models")
    previous_models = load_previous_model_summaries(project_dir)
    gb_test_row = comparison_table[(comparison_table["model"] == "tuned_gb_selected_threshold") & (comparison_table["split"] == "test")].copy()
    gb_test_row.loc[:, "model"] = "Gradient Boosting"
    model_summary = pd.concat([previous_models, gb_test_row], ignore_index=True, sort=False)
    if not model_summary.empty:
        keep_cols = [
            "model", "accuracy", "precision_BAD_1", "recall_BAD_1",
            "f1_BAD_1", "roc_auc", "average_precision", "false_negatives", "false_positives", "source"
        ]
        model_summary = model_summary.reindex(columns=[col for col in keep_cols if col in model_summary.columns])
        print_table(model_summary.round(4), "Available model comparison")
    else:
        print("No previous model output files were found for comparison.")

    print_section("Interpretability")
    feature_importances = extract_feature_importance_table(tuned_pipeline)
    print_table(feature_importances.head(20).round(4), "Top 20 impurity-based feature importances")
    print(
        "\nFeature importances describe predictive associations, not causal effects. "
        "Impurity-based importances can be biased toward continuous or high-cardinality variables, "
        "so permutation importance is also computed on the held-out test set."
    )
    permutation_table = compute_permutation_importance_table(tuned_pipeline, X_test, y_test, random_state=random_state)
    print_table(permutation_table.round(4), "Permutation importances on the test set")

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best_params = search.best_params_

    if save_outputs and resolved_output_dir is not None:
        save_table(comparison_table, resolved_output_dir, "gb_model_comparison.csv", index=False)
        save_table(model_summary, resolved_output_dir, "gb_comparison_with_previous_models.csv", index=False)
        save_table(cv_results, resolved_output_dir, "gb_hyperparameter_results.csv", index=False)
        with (resolved_output_dir / "gb_best_params.json").open("w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved table: {resolved_output_dir / 'gb_best_params.json'}")
        save_table(threshold_table, resolved_output_dir, "gb_threshold_analysis.csv", index=False)
        save_table(baseline_cm, resolved_output_dir, "baseline_gb_confusion_matrix.csv")
        save_table(tuned_cm, resolved_output_dir, "tuned_gb_confusion_matrix.csv")
        save_table(feature_importances, resolved_output_dir, "tuned_gb_feature_importances.csv", index=False)
        save_table(permutation_table, resolved_output_dir, "tuned_gb_permutation_importances.csv", index=False)
        save_table(train_report, resolved_output_dir, "tuned_gb_classification_report_train.csv")
        save_table(test_report, resolved_output_dir, "tuned_gb_classification_report_test.csv")
        pd.DataFrame([split_info]).to_csv(resolved_output_dir / "gb_data_split_info.csv", index=False)
        print(f"Saved table: {resolved_output_dir / 'gb_data_split_info.csv'}")

    print_section("Plots")
    plot_confusion_matrix(
        baseline_cm.values,
        "Baseline Gradient Boosting Confusion Matrix - Test Set",
        "baseline_gb_confusion_matrix.png",
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_confusion_matrix(
        tuned_cm.values,
        "Tuned Gradient Boosting Confusion Matrix - Test Set",
        "tuned_gb_confusion_matrix.png",
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_roc_curve(
        y_test,
        {"Baseline GB": baseline_test_proba, "Tuned GB": tuned_test_proba},
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_precision_recall_curve(
        y_test,
        {"Baseline GB": baseline_test_proba, "Tuned GB": tuned_test_proba},
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_feature_importances(feature_importances, resolved_output_dir, save_outputs, show_plots, top_n=20)
    plot_permutation_importances(permutation_table, resolved_output_dir, save_outputs, show_plots, top_n=12)

    if save_outputs and resolved_output_dir is not None:
        print(f"\nAll Gradient Boosting outputs saved to: {resolved_output_dir}")

    return {
        "baseline_pipeline": baseline_pipeline,
        "tuned_pipeline": tuned_pipeline,
        "best_parameters": best_params,
        "cv_results": cv_results,
        "model_comparison": comparison_table,
        "comparison_with_previous_models": model_summary,
        "train_metrics": {
            "baseline": rows[0],
            "tuned_default_threshold": rows[2],
            "tuned_selected_threshold": rows[4],
        },
        "test_metrics": {
            "baseline": rows[1],
            "tuned_default_threshold": rows[3],
            "tuned_selected_threshold": rows[5],
        },
        "threshold_comparison_table": threshold_table,
        "selected_threshold": selected_threshold,
        "baseline_confusion_matrix": baseline_cm,
        "tuned_confusion_matrix": tuned_cm,
        "feature_importance_table": feature_importances,
        "permutation_importance_table": permutation_table,
        "data_split_info": split_info,
        "output_dir": resolved_output_dir,
    }


if __name__ == "__main__":
    run_gradient_boosting_analysis()


"""XGBoost model for HMEQ loan default prediction.

This module builds a tuned XGBoost classifier for BAD while keeping all
preprocessing leakage-free through sklearn Pipeline and ColumnTransformer.
It does not replace XGBoost with another model if the optional xgboost package
is unavailable.
"""

from pathlib import Path
import importlib.util
import json
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
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


def get_xgb_classifier_class():
    """Import XGBClassifier only when the optional xgboost dependency is available."""
    if importlib.util.find_spec("xgboost") is None:
        return None
    from xgboost import XGBClassifier

    return XGBClassifier


def build_xgb_pipeline(numerical_predictors: list[str], categorical_predictors: list[str], model: Any) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(numerical_predictors, categorical_predictors)),
        ("xgb", model),
    ])


def extract_feature_importance_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    feature_names = get_feature_names(fitted_pipeline)
    model = fitted_pipeline.named_steps["xgb"]

    weight_importance = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    table = pd.DataFrame({
        "feature": feature_names,
        "importance": weight_importance,
    })

    try:
        booster = model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        gain_by_feature = {}
        for key, value in gain_scores.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(feature_names):
                    gain_by_feature[feature_names[idx]] = value
            else:
                gain_by_feature[key] = value
        table["gain"] = table["feature"].map(gain_by_feature).fillna(0.0)
    except Exception:
        table["gain"] = np.nan

    return table.sort_values(["gain", "importance"], ascending=False).reset_index(drop=True)


def compute_permutation_importance_table(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
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
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, y_proba in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No skill")
    ax.set_title("XGBoost ROC Curve - Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    finish_plot("xgb_roc_curve.png", output_dir, save_outputs, show_plots)


def plot_precision_recall_curve(y_true, model_probabilities: dict[str, np.ndarray], output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, y_proba in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})")
    ax.set_title("XGBoost Precision-Recall Curve - Test Set")
    ax.set_xlabel("Recall for BAD = 1")
    ax.set_ylabel("Precision for BAD = 1")
    ax.legend(loc="lower left")
    finish_plot("xgb_precision_recall_curve.png", output_dir, save_outputs, show_plots)


def plot_feature_importances(feature_importances: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool, top_n: int = 20) -> None:
    plot_table = feature_importances.head(top_n).sort_values("gain")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=plot_table, x="gain", y="feature", color="#4472c4", ax=ax)
    ax.set_title(f"Top {top_n} XGBoost Feature Importances by Gain")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Feature")
    finish_plot("tuned_xgb_feature_importances.png", output_dir, save_outputs, show_plots)


def plot_permutation_importances(permutation_table: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool, top_n: int = 12) -> None:
    plot_table = permutation_table.head(top_n).sort_values("permutation_importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=plot_table, x="permutation_importance_mean", y="feature", color="#70ad47", ax=ax)
    ax.set_title(f"Top {top_n} XGBoost Permutation Importances")
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_ylabel("Original feature")
    finish_plot("tuned_xgb_permutation_importances.png", output_dir, save_outputs, show_plots)


def confusion_matrix_table(y_true, y_proba, threshold: float) -> pd.DataFrame:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"])


def load_existing_model_summaries(project_dir: Path) -> pd.DataFrame:
    sources = [
        ("Logistic Regression", project_dir / "outputs" / "outputs_logit" / "improved_model_metrics.csv", "improved_logistic", "test"),
        ("Decision Tree", project_dir / "outputs" / "outputs_tree" / "tree_model_comparison.csv", "tuned_tree_selected_threshold", "test"),
        ("Random Forest", project_dir / "outputs" / "outputs_rf" / "rf_model_comparison.csv", "tuned_rf_selected_threshold", "test"),
        ("Gradient Boosting", project_dir / "outputs" / "outputs_gb" / "gb_model_comparison.csv", "tuned_gb_selected_threshold", "test"),
        ("AdaBoost", project_dir / "outputs" / "outputs_ada" / "ada_model_comparison.csv", "tuned_ada_selected_threshold", "test"),
    ]

    rows = []
    for label, path, model_name, split in sources:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "model" in table.columns and "split" in table.columns:
            matched = table[(table["model"] == model_name) & (table["split"] == split)]
            if matched.empty:
                matched = table[table["split"] == split].tail(1)
        elif "split" in table.columns:
            matched = table[table["split"] == split].tail(1)
        else:
            matched = table.tail(1)
        if matched.empty:
            continue
        row = matched.iloc[0].to_dict()
        row["model"] = label
        rows.append(row)

    return pd.DataFrame(rows)


def write_missing_dependency_note(output_dir: Path | None, save_outputs: bool) -> None:
    message = (
        "The optional package 'xgboost' is not installed in the current Python environment.\n"
        "Install it with: pip install xgboost\n"
        "No substitute model was trained, because this script is specifically for XGBoost.\n"
    )
    print(message)
    if save_outputs and output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        note_path = output_dir / "xgboost_dependency_note.txt"
        note_path.write_text(message, encoding="utf-8")
        print(f"Saved dependency note: {note_path}")


def run_xgboost_analysis(
    data_path=None,
    output_dir=None,
    test_size=0.30,
    random_state=42,
    save_outputs=True,
    show_plots=True,
) -> dict[str, Any]:
    """Run baseline and tuned XGBoost analyses for HMEQ default prediction."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    resolved_data_path = resolve_data_path(data_path)
    resolved_output_dir = resolve_output_dir(output_dir, "outputs_xg")

    if save_outputs:
        resolved_output_dir.mkdir(exist_ok=True)
    else:
        resolved_output_dir = None

    XGBClassifier = get_xgb_classifier_class()
    if XGBClassifier is None:
        write_missing_dependency_note(resolved_output_dir, save_outputs)
        return {
            "status": "missing_dependency",
            "required_package": "xgboost",
            "install_command": "pip install xgboost",
            "output_dir": resolved_output_dir,
        }

    print_section("Original Notebook Reference")
    print(
        "The original notebook used XGBClassifier with histogram trees, eval_metric='auc', "
        "scale_pos_weight for the approximately 80/20 class imbalance, and RandomizedSearchCV "
        "scored by ROC-AUC over n_estimators, learning_rate, max_depth, subsample, and "
        "colsample_bytree. This script keeps that modelling logic but uses the same leakage-free "
        "Pipeline/ColumnTransformer convention as the revised model files. Missing REASON/JOB "
        "values are retained as an explicit 'Missing' category."
    )

    print_section("Load Data")
    df = pd.read_csv(resolved_data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found in {resolved_data_path}")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    categorical_predictors = [col for col in CATEGORICAL_PREDICTORS if col in X.columns]
    numerical_predictors = [col for col in X.columns if col not in categorical_predictors]

    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Numerical predictors: {numerical_predictors}")
    print(f"Categorical predictors: {categorical_predictors}")

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
        "train_default_rate": y_train.mean(),
        "test_default_rate": y_test.mean(),
    }
    print_table(pd.DataFrame([split_info]).round(4), "Split information")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    print_section("Fit Baseline XGBoost")
    baseline_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
    )
    baseline_pipeline = build_xgb_pipeline(numerical_predictors, categorical_predictors, baseline_model)
    baseline_pipeline.fit(X_train, y_train)

    print_section("Tune XGBoost")
    tuned_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
    )
    tuned_pipeline = build_xgb_pipeline(numerical_predictors, categorical_predictors, tuned_base)
    param_distributions = {
        "xgb__n_estimators": [200, 300, 400, 500, 600],
        "xgb__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.10, 0.15],
        "xgb__max_depth": [2, 3, 4, 5, 6],
        "xgb__min_child_weight": [1, 2, 3, 5, 8],
        "xgb__subsample": [0.70, 0.80, 0.90, 1.00],
        "xgb__colsample_bytree": [0.60, 0.70, 0.80, 0.90, 1.00],
        "xgb__gamma": [0, 0.05, 0.10, 0.25, 0.50],
        "xgb__reg_alpha": [0, 0.01, 0.05, 0.10, 0.50],
        "xgb__reg_lambda": [0.50, 1.00, 1.50, 2.00, 3.00],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        tuned_pipeline,
        param_distributions=param_distributions,
        n_iter=35,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
        return_train_score=True,
    )
    print("Tuning metric: ROC-AUC, chosen to evaluate ranking ability across thresholds on training folds.")
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    print_section("Threshold Analysis")
    oof_train_proba = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold_table = threshold_analysis(y_train, oof_train_proba, include_average_precision=True)
    selected_threshold = select_threshold(threshold_table, criterion="f1_BAD_1")
    top_thresholds = threshold_table.sort_values("f1_BAD_1", ascending=False).head(10)
    print_table(top_thresholds.round(4), "Top out-of-fold training thresholds by F1 for BAD = 1")
    selected_row = threshold_table.loc[threshold_table["threshold"] == selected_threshold].iloc[0]
    print(
        f"\nSelected threshold: {selected_threshold:.3f}. It was chosen by maximising "
        "out-of-fold training F1 for BAD = 1, not by optimising the test set."
    )
    print(
        f"At this out-of-fold threshold, precision={selected_row['precision_BAD_1']:.3f}, "
        f"recall={selected_row['recall_BAD_1']:.3f}, false negatives={int(selected_row['false_negatives'])}, "
        f"false positives={int(selected_row['false_positives'])}."
    )

    print_section("Model Evaluation")
    baseline_test_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    tuned_train_proba = best_pipeline.predict_proba(X_train)[:, 1]
    tuned_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    comparison_rows = [
        evaluate_pipeline("baseline_xgboost", baseline_pipeline, X_train, y_train, "train", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("baseline_xgboost", baseline_pipeline, X_test, y_test, "test", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("tuned_xgb_default_threshold", best_pipeline, X_train, y_train, "train", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("tuned_xgb_default_threshold", best_pipeline, X_test, y_test, "test", threshold=0.5, include_average_precision=True),
        evaluate_pipeline("tuned_xgb_selected_threshold", best_pipeline, X_train, y_train, "train", threshold=selected_threshold, include_average_precision=True),
        evaluate_pipeline("tuned_xgb_selected_threshold", best_pipeline, X_test, y_test, "test", threshold=selected_threshold, include_average_precision=True),
    ]
    comparison_table = pd.DataFrame(comparison_rows)
    print_table(comparison_table.round(4), "XGBoost model comparison")
    print(
        "\nAccuracy alone is not sufficient because BAD = 1 is the minority class. "
        "Recall for BAD = 1 and false negatives matter because missed likely defaulters are costly."
    )

    baseline_cm = confusion_matrix_table(y_test, baseline_test_proba, threshold=0.5)
    tuned_cm = confusion_matrix_table(y_test, tuned_test_proba, threshold=selected_threshold)
    print_table(baseline_cm, "Baseline test confusion matrix")
    print_table(tuned_cm, "Tuned test confusion matrix at selected threshold")

    tuned_train_pred = (tuned_train_proba >= selected_threshold).astype(int)
    tuned_test_pred = (tuned_test_proba >= selected_threshold).astype(int)
    train_report = pd.DataFrame(classification_report(y_train, tuned_train_pred, output_dict=True, zero_division=0)).T
    test_report = pd.DataFrame(classification_report(y_test, tuned_test_pred, output_dict=True, zero_division=0)).T
    print_table(test_report.round(4), "Tuned XGBoost test classification report")

    print_section("Comparison With Previous Models")
    previous_summary = load_existing_model_summaries(project_dir)
    xgb_test_row = comparison_table[
        (comparison_table["model"] == "tuned_xgb_selected_threshold") & (comparison_table["split"] == "test")
    ].copy()
    xgb_test_row.loc[:, "model"] = "XGBoost"
    model_summary = pd.concat([previous_summary, xgb_test_row], ignore_index=True, sort=False)
    if not model_summary.empty:
        columns = [
            col for col in [
                "model",
                "accuracy",
                "precision_BAD_1",
                "recall_BAD_1",
                "f1_BAD_1",
                "roc_auc",
                "average_precision",
                "false_negatives",
                "false_positives",
            ] if col in model_summary.columns
        ]
        print_table(model_summary[columns].round(4), "Available model comparison")

    print_section("Interpretability")
    feature_importances = extract_feature_importance_table(best_pipeline)
    print_table(feature_importances.head(20).round(4), "Top 20 XGBoost feature importances by gain")
    print(
        "\nFeature importances describe predictive associations, not causal effects. "
        "Gain-based XGBoost importance measures improvement in splits and should be interpreted "
        "alongside permutation importance."
    )
    permutation_table = compute_permutation_importance_table(best_pipeline, X_test, y_test, random_state)
    print_table(permutation_table.round(4), "Permutation importances on the test set")

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")

    if save_outputs and resolved_output_dir is not None:
        save_table(comparison_table, resolved_output_dir, "xgb_model_comparison.csv", index=False)
        save_table(model_summary, resolved_output_dir, "xgb_comparison_with_previous_models.csv", index=False)
        save_table(cv_results, resolved_output_dir, "xgb_hyperparameter_results.csv", index=False)
        with (resolved_output_dir / "xgb_best_params.json").open("w", encoding="utf-8") as f:
            json.dump(search.best_params_, f, indent=2)
        print(f"Saved table: {resolved_output_dir / 'xgb_best_params.json'}")
        save_table(threshold_table, resolved_output_dir, "xgb_threshold_analysis.csv", index=False)
        save_table(baseline_cm, resolved_output_dir, "baseline_xgb_confusion_matrix.csv")
        save_table(tuned_cm, resolved_output_dir, "tuned_xgb_confusion_matrix.csv")
        save_table(feature_importances, resolved_output_dir, "tuned_xgb_feature_importances.csv", index=False)
        save_table(permutation_table, resolved_output_dir, "tuned_xgb_permutation_importances.csv", index=False)
        save_table(train_report, resolved_output_dir, "tuned_xgb_classification_report_train.csv")
        save_table(test_report, resolved_output_dir, "tuned_xgb_classification_report_test.csv")
        pd.DataFrame([split_info]).to_csv(resolved_output_dir / "xgb_data_split_info.csv", index=False)
        print(f"Saved table: {resolved_output_dir / 'xgb_data_split_info.csv'}")

    print_section("Plots")
    plot_confusion_matrix(
        baseline_cm.values,
        "Baseline XGBoost Confusion Matrix - Test Set",
        "baseline_xgb_confusion_matrix.png",
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_confusion_matrix(
        tuned_cm.values,
        "Tuned XGBoost Confusion Matrix - Test Set",
        "tuned_xgb_confusion_matrix.png",
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_roc_curve(
        y_test,
        {"Baseline XGBoost": baseline_test_proba, "Tuned XGBoost": tuned_test_proba},
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_precision_recall_curve(
        y_test,
        {"Baseline XGBoost": baseline_test_proba, "Tuned XGBoost": tuned_test_proba},
        resolved_output_dir,
        save_outputs,
        show_plots,
    )
    plot_feature_importances(feature_importances, resolved_output_dir, save_outputs, show_plots, top_n=20)
    plot_permutation_importances(permutation_table, resolved_output_dir, save_outputs, show_plots, top_n=12)

    if save_outputs and resolved_output_dir is not None:
        print(f"\nAll XGBoost outputs saved to: {resolved_output_dir}")

    return {
        "status": "success",
        "baseline_pipeline": baseline_pipeline,
        "tuned_pipeline": best_pipeline,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": cv_results,
        "model_comparison": comparison_table,
        "comparison_with_previous_models": model_summary,
        "threshold_analysis": threshold_table,
        "selected_threshold": selected_threshold,
        "baseline_confusion_matrix": baseline_cm,
        "tuned_confusion_matrix": tuned_cm,
        "feature_importances": feature_importances,
        "permutation_importances": permutation_table,
        "train_classification_report": train_report,
        "test_classification_report": test_report,
        "split_info": split_info,
        "output_dir": resolved_output_dir,
    }


if __name__ == "__main__":
    run_xgboost_analysis()


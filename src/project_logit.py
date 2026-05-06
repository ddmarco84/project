"""Improved leakage-free logistic regression model for HMEQ default prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "BAD"
CATEGORICAL_PREDICTORS = ["REASON", "JOB"]
MISSING_INDICATOR_COLUMNS = ["VALUE", "MORTDUE", "DEBTINC"]

try:
    from .preprocessing import build_preprocessor, clean_feature_name, resolve_data_path, resolve_output_dir
except ImportError:
    from preprocessing import build_preprocessor, clean_feature_name, resolve_data_path, resolve_output_dir

try:
    from .evaluation import (
        calculate_metrics,
        finish_plot,
        print_section,
        print_table,
        save_table,
        select_threshold,
        threshold_analysis,
    )
except ImportError:
    from evaluation import (
        calculate_metrics,
        finish_plot,
        print_section,
        print_table,
        save_table,
        select_threshold,
        threshold_analysis,
    )


def build_improved_pipeline(
    numerical_predictors: list[str],
    categorical_predictors: list[str],
    *,
    class_weight=None,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build the recommended leakage-free logistic regression pipeline."""
    preprocessor = build_preprocessor(
        numerical_predictors,
        categorical_predictors,
        scale_numeric=True,
    )

    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=5000,
        class_weight=class_weight,
        random_state=random_state,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("logit", model),
    ])


def extract_coefficient_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Extract logistic-regression coefficients after preprocessing."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    logit = fitted_pipeline.named_steps["logit"]
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]

    coefficient_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": logit.coef_[0],
    })
    coefficient_table["odds_ratio"] = np.exp(coefficient_table["coefficient"])
    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()
    return coefficient_table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def interpret_coefficients(coefficient_table: pd.DataFrame, top_n: int = 5) -> str:
    """Create a concise non-causal interpretation of the largest coefficients."""
    positive = coefficient_table[coefficient_table["coefficient"] > 0].head(top_n)
    negative = coefficient_table[coefficient_table["coefficient"] < 0].head(top_n)

    positive_features = ", ".join(positive["feature"].tolist())
    negative_features = ", ".join(negative["feature"].tolist())

    return (
        "Positive coefficients are associated with higher predicted default risk, "
        f"with the largest positive terms including: {positive_features}. "
        "Negative coefficients are associated with lower predicted default risk, "
        f"with the largest negative terms including: {negative_features}. "
        "Because numerical variables are scaled and categorical variables are one-hot encoded, "
        "coefficient magnitudes should be interpreted as model-specific associations, not causal effects."
    )


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    """Plot the final test-set confusion matrix."""
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
    ax.set_title("Improved Logistic Regression Confusion Matrix - Test Set")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    finish_plot("improved_confusion_matrix.png", output_dir, save_outputs, show_plots)


def plot_roc_curve(y_true, y_proba, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    """Plot the final test-set ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="#4C78A8")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random baseline")
    ax.set_title("Improved Logistic Regression ROC Curve - Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / recall")
    ax.legend(loc="lower right")
    finish_plot("roc_curve_improved_logistic.png", output_dir, save_outputs, show_plots)


def plot_precision_recall_curve(y_true, y_proba, output_dir: Path | None, save_outputs: bool, show_plots: bool) -> None:
    """Plot the final test-set precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    baseline_rate = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(recall, precision, label="Improved logistic regression", color="#F58518")
    ax.axhline(baseline_rate, linestyle="--", color="#888888", label=f"Default rate = {baseline_rate:.3f}")
    ax.set_title("Improved Logistic Regression Precision-Recall Curve - Test Set")
    ax.set_xlabel("Recall for BAD = 1")
    ax.set_ylabel("Precision for BAD = 1")
    ax.legend(loc="best")
    finish_plot("precision_recall_curve_improved_logistic.png", output_dir, save_outputs, show_plots)


def plot_coefficients(coefficient_table: pd.DataFrame, output_dir: Path | None, save_outputs: bool, show_plots: bool, top_n: int = 20) -> None:
    """Plot the largest improved-model coefficients by absolute magnitude."""
    plot_data = coefficient_table.head(top_n).sort_values("coefficient")
    colors = np.where(plot_data["coefficient"] >= 0, "#D55E00", "#0072B2")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_data["feature"], plot_data["coefficient"], color=colors)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title(f"Top {top_n} Improved Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    finish_plot("improved_logistic_coefficients_top20.png", output_dir, save_outputs, show_plots)


def run_logistic_regression_analysis(
    data_path=None,
    output_dir=None,
    test_size=0.30,
    random_state=42,
    save_outputs=True,
    show_plots=True,
):
    """Run the improved leakage-free logistic regression workflow."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    resolved_data_path = resolve_data_path(data_path)
    resolved_output_dir = resolve_output_dir(output_dir, "outputs_logit")

    if save_outputs:
        resolved_output_dir.mkdir(exist_ok=True)
    else:
        resolved_output_dir = None

    pd.options.display.float_format = "{:,.4f}".format
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    sns.set_theme(style="whitegrid", context="notebook")

    if not resolved_data_path.exists():
        raise FileNotFoundError(f"Could not find hmeq.csv at {resolved_data_path}")

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

    print_section("Fit Improved Logistic Regression")
    base_pipeline = build_improved_pipeline(
        numerical_predictors,
        categorical_predictors,
        class_weight=None,
        C=1.0,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {
        "logit__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "logit__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    improved_pipeline = search.best_estimator_

    logit = improved_pipeline.named_steps["logit"]
    solver_iterations = int(np.max(logit.n_iter_))
    converged = solver_iterations < logit.max_iter

    print(f"Best parameters from train CV: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Solver iterations used: {solver_iterations} / {logit.max_iter}")
    print(f"Converged: {converged}")
    print(
        "The pipeline imputes and scales numerical variables, imputes categorical missing values as "
        "'Missing', one-hot encodes REASON and JOB, and adds missingness indicators for VALUE, "
        "MORTDUE, and DEBTINC. All preprocessing is fitted only on the training data."
    )

    print_section("Threshold Analysis")
    train_proba = improved_pipeline.predict_proba(X_train)[:, 1]
    threshold_table = threshold_analysis(y_train, train_proba)
    selected_threshold = select_threshold(threshold_table, criterion="f1_BAD_1")
    selected_row = threshold_table.loc[threshold_table["threshold"].eq(selected_threshold)].iloc[0]
    print_table(threshold_table.sort_values("f1_BAD_1", ascending=False).head(10).round(4), "Top training thresholds by F1 for BAD = 1")
    print(
        f"\nSelected threshold: {selected_threshold:.3f}. "
        "It was chosen by maximising F1 for BAD = 1 on the training set, not the test set."
    )
    print(
        f"At this training threshold, precision={selected_row['precision_BAD_1']:.3f}, "
        f"recall={selected_row['recall_BAD_1']:.3f}, false negatives={int(selected_row['false_negatives'])}, "
        f"false positives={int(selected_row['false_positives'])}."
    )

    print_section("Final Evaluation")
    test_proba = improved_pipeline.predict_proba(X_test)[:, 1]
    train_pred = (train_proba >= selected_threshold).astype(int)
    test_pred = (test_proba >= selected_threshold).astype(int)

    train_metrics = calculate_metrics(y_train, train_pred, train_proba)
    test_metrics = calculate_metrics(y_test, test_pred, test_proba)
    metrics_table = pd.DataFrame([train_metrics, test_metrics], index=["train", "test"]).round(4)
    print_table(metrics_table, "Improved logistic regression metrics")
    print(
        "\nAccuracy alone is not sufficient because defaults are the minority class. "
        "For loan-default prediction, recall for BAD = 1 and false negatives are especially important, "
        "but improving recall usually increases false positives."
    )

    confusion_matrix_table = pd.DataFrame(
        confusion_matrix(y_test, test_pred, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    print_table(confusion_matrix_table, "Final test confusion matrix")

    train_report = pd.DataFrame(classification_report(y_train, train_pred, output_dict=True, zero_division=0)).T
    test_report = pd.DataFrame(classification_report(y_test, test_pred, output_dict=True, zero_division=0)).T
    print_table(test_report.round(4), "Final test classification report")

    print_section("Coefficient Interpretation")
    coefficient_table = extract_coefficient_table(improved_pipeline)
    print_table(coefficient_table.head(20).round(4), "Top 20 improved-model coefficients")
    coefficient_interpretation = interpret_coefficients(coefficient_table)
    print("\n" + coefficient_interpretation)

    if save_outputs and resolved_output_dir is not None:
        save_table(metrics_table, resolved_output_dir, "improved_model_metrics.csv")
        save_table(threshold_table, resolved_output_dir, "threshold_analysis.csv", index=False)
        save_table(confusion_matrix_table, resolved_output_dir, "improved_confusion_matrix.csv")
        save_table(coefficient_table, resolved_output_dir, "improved_logistic_coefficients.csv", index=False)
        save_table(train_report, resolved_output_dir, "improved_classification_report_train.csv")
        save_table(test_report, resolved_output_dir, "improved_classification_report_test.csv")
        pd.DataFrame([split_info]).to_csv(resolved_output_dir / "data_split_info.csv", index=False)
        print(f"Saved table: {resolved_output_dir / 'data_split_info.csv'}")

    print_section("Plots")
    plot_confusion_matrix(confusion_matrix_table.values, resolved_output_dir, save_outputs, show_plots)
    plot_roc_curve(y_test, test_proba, resolved_output_dir, save_outputs, show_plots)
    plot_precision_recall_curve(y_test, test_proba, resolved_output_dir, save_outputs, show_plots)
    plot_coefficients(coefficient_table, resolved_output_dir, save_outputs, show_plots, top_n=20)

    if save_outputs and resolved_output_dir is not None:
        print(f"\nAll improved logistic-regression outputs saved to: {resolved_output_dir}")

    return {
        "improved_pipeline": improved_pipeline,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "metrics_table": metrics_table,
        "threshold_analysis": threshold_table,
        "selected_threshold": selected_threshold,
        "confusion_matrix": confusion_matrix_table,
        "train_classification_report": train_report,
        "test_classification_report": test_report,
        "coefficient_table": coefficient_table,
        "coefficient_interpretation": coefficient_interpretation,
        "data_split_info": split_info,
        "best_params": search.best_params_,
        "best_cv_roc_auc": search.best_score_,
        "converged": converged,
        "solver_iterations": solver_iterations,
        "output_dir": resolved_output_dir,
    }


if __name__ == "__main__":
    run_logistic_regression_analysis()


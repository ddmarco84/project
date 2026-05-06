"""SHAP analysis for the best predictive HMEQ loan-default model.

The original notebook used SHAP to explain the tuned XGBoost model. This script
keeps that intent, selects the best model by the main predictive metric
ROC-AUC, retrains the final tuned XGBoost specification, and saves SHAP tables
and plots for the leakage-free preprocessed feature matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import json

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

try:
    from .project_xg import build_xgb_pipeline, get_feature_names, get_xgb_classifier_class
except ImportError:
    from project_xg import build_xgb_pipeline, get_feature_names, get_xgb_classifier_class


TARGET = "BAD"
CATEGORICAL_PREDICTORS = ["REASON", "JOB"]
MISSING_INDICATOR_COLUMNS = ["VALUE", "MORTDUE", "DEBTINC"]
PRIMARY_SELECTION_METRIC = "roc_auc"


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_table(table: pd.DataFrame, title: str | None = None, index: bool = False) -> None:
    if title:
        print(f"\n{title}")
    print(table.to_string(index=index))


def save_table(table: pd.DataFrame, output_dir: Path, filename: str, index: bool = False) -> None:
    path = output_dir / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def finish_plot(path: Path, show_plots: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path}")
    if show_plots:
        plt.show()
    plt.close()


def check_shap_available():
    if importlib.util.find_spec("shap") is None:
        return None
    import shap

    return shap


def write_missing_dependency_note(output_dir: Path, package_name: str) -> dict[str, Any]:
    message = (
        f"The optional package '{package_name}' is not installed in the current Python environment.\n"
        f"Install it with: pip install {package_name}\n"
        "No substitute interpretability method was run.\n"
    )
    print(message)
    output_dir.mkdir(exist_ok=True)
    note_path = output_dir / f"{package_name}_dependency_note.txt"
    note_path.write_text(message, encoding="utf-8")
    print(f"Saved dependency note: {note_path}")
    return {"status": "missing_dependency", "required_package": package_name, "install_command": f"pip install {package_name}"}


def load_main_model_comparison(project_dir: Path) -> pd.DataFrame:
    """Load test-set predictive metrics from the existing model output folders."""
    sources = [
        ("Logistic Regression", project_dir / "outputs" / "outputs_logit" / "improved_model_metrics.csv", None),
        ("Decision Tree", project_dir / "outputs" / "outputs_tree" / "tree_model_comparison.csv", "tuned_tree_selected_threshold"),
        ("Random Forest", project_dir / "outputs" / "outputs_rf" / "rf_model_comparison.csv", "tuned_rf_selected_threshold"),
        ("Gradient Boosting", project_dir / "outputs" / "outputs_gb" / "gb_model_comparison.csv", "tuned_gb_selected_threshold"),
        ("AdaBoost", project_dir / "outputs" / "outputs_ada" / "ada_model_comparison.csv", "tuned_ada_selected_threshold"),
        ("XGBoost", project_dir / "outputs" / "outputs_xg" / "xgb_model_comparison.csv", "tuned_xgb_selected_threshold"),
    ]

    rows = []
    for model_name, path, target_model_row in sources:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "split" in table.columns:
            table = table[table["split"].astype(str).str.lower().eq("test")]
        if target_model_row and "model" in table.columns:
            matched = table[table["model"].eq(target_model_row)]
            if not matched.empty:
                table = matched
        if table.empty:
            continue
        row = table.tail(1).iloc[0].to_dict()
        row["model_label"] = model_name
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise FileNotFoundError("No existing model-comparison outputs were found.")
    return comparison


def select_best_predictive_model(comparison: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in comparison.columns:
        raise ValueError(f"Metric '{metric}' is not available in the model comparison table.")
    return comparison.loc[comparison[metric].idxmax()]


def load_best_xgb_params(project_dir: Path) -> dict[str, Any]:
    params_path = project_dir / "outputs" / "outputs_xg" / "xgb_best_params.json"
    if not params_path.exists():
        return {
            "n_estimators": 400,
            "learning_rate": 0.06,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "gamma": 0.05,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
        }

    with params_path.open("r", encoding="utf-8") as f:
        raw_params = json.load(f)
    return {key.replace("xgb__", ""): value for key, value in raw_params.items()}


def load_data_split(data_path: Path, test_size: float, random_state: int):
    df = pd.read_csv(data_path)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_final_xgb_pipeline(data_path: Path, project_dir: Path, test_size: float, random_state: int):
    XGBClassifier = get_xgb_classifier_class()
    if XGBClassifier is None:
        return None

    X_train, X_test, y_train, y_test = load_data_split(data_path, test_size, random_state)
    categorical_predictors = [col for col in CATEGORICAL_PREDICTORS if col in X_train.columns]
    numerical_predictors = [col for col in X_train.columns if col not in categorical_predictors]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best_params = load_best_xgb_params(project_dir)
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "n_jobs": -1,
        **best_params,
    }
    xgb_model = XGBClassifier(**model_params)
    pipeline = build_xgb_pipeline(numerical_predictors, categorical_predictors, xgb_model)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test, model_params


def evaluate_pipeline(pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_BAD_1": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_BAD_1": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_BAD_1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }


def transform_for_shap(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = get_feature_names(pipeline)
    X_transformed = pd.DataFrame(transformed, columns=feature_names, index=X.index)
    return X_transformed, pipeline.named_steps["xgb"]


def make_shap_explanation(shap_module, model, X_transformed: pd.DataFrame):
    """Create SHAP Explanation in XGBoost raw-margin/log-odds space."""
    explainer = shap_module.TreeExplainer(model)
    raw_values = explainer.shap_values(X_transformed)
    if isinstance(raw_values, list):
        values = raw_values[1]
    else:
        values = raw_values

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(np.ravel(expected_value)[-1])
    else:
        base_value = float(expected_value)

    explanation = shap_module.Explanation(
        values=values,
        base_values=np.repeat(base_value, X_transformed.shape[0]),
        data=X_transformed.values,
        feature_names=list(X_transformed.columns),
    )
    return explainer, explanation, base_value


def build_shap_importance_table(explanation) -> pd.DataFrame:
    values = np.asarray(explanation.values)
    table = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["clean_feature_label"] = table["feature"].str.replace("_", " ", regex=False)
    return table[["rank", "feature", "clean_feature_label", "mean_abs_shap", "mean_shap"]]


def save_shap_plots(shap_module, explanation, X_transformed: pd.DataFrame, importance_table: pd.DataFrame, output_dir: Path, show_plots: bool) -> list[int]:
    shap_module.plots.bar(explanation, max_display=20, show=False)
    finish_plot(output_dir / "shap_bar_mean_absolute.png", show_plots)

    shap_module.plots.beeswarm(explanation, max_display=20, show=False)
    finish_plot(output_dir / "shap_summary_beeswarm.png", show_plots)

    top_features = importance_table["feature"].head(5).tolist()
    for feature in top_features:
        shap_module.dependence_plot(
            feature,
            explanation.values,
            X_transformed,
            feature_names=list(X_transformed.columns),
            show=False,
        )
        safe_name = feature.replace("/", "_").replace("\\", "_").replace(" ", "_")
        finish_plot(output_dir / f"shap_dependence_{safe_name}.png", show_plots)

    model_scores = np.asarray(explanation.values).sum(axis=1) + np.asarray(explanation.base_values)
    low_risk_idx = int(np.argmin(model_scores))
    high_risk_idx = int(np.argmax(model_scores))
    median_risk_idx = int(np.argsort(model_scores)[len(model_scores) // 2])
    selected_indices = [low_risk_idx, median_risk_idx, high_risk_idx]

    for label, idx in zip(["low_risk", "median_risk", "high_risk"], selected_indices):
        shap_module.plots.waterfall(explanation[idx], max_display=15, show=False)
        finish_plot(output_dir / f"shap_waterfall_{label}_observation.png", show_plots)

        force_html = shap_module.plots.force(explanation[idx], matplotlib=False)
        shap_module.save_html(str(output_dir / f"shap_force_{label}_observation.html"), force_html)
        print(f"Saved HTML: {output_dir / f'shap_force_{label}_observation.html'}")

    return selected_indices


def write_interpretation(
    output_dir: Path,
    selected_metric: str,
    selected_model: str,
    model_metrics: dict[str, float],
    importance_table: pd.DataFrame,
    base_value: float,
) -> str:
    top_features = importance_table.head(8)
    lines = [
        "# SHAP Interpretation",
        "",
        f"Selected model: {selected_model}.",
        f"Selection metric: {selected_metric}. The model was selected from the main predictive metrics, not from expected-loss analysis.",
        "",
        "The SHAP values were computed for the transformed feature matrix used by the fitted XGBoost model. For binary XGBoost with the default TreeExplainer output, SHAP values are in raw-margin/log-odds space. Positive SHAP values push the prediction toward higher default risk; negative values push it toward lower default risk.",
        "",
        "## Test Metrics",
        "",
        pd.DataFrame([model_metrics]).round(4).to_markdown(index=False),
        "",
        "## Most Important Features",
        "",
        top_features[["rank", "feature", "mean_abs_shap", "mean_shap"]].round(4).to_markdown(index=False),
        "",
        "The highest-ranked features are dominated by debt burden, missing debt-to-income information, credit-history variables, and collateral or balance-sheet variables. This is consistent with the original notebook, where SHAP highlighted DEBTINC, DEBTINC_missing, DELINQ, CLAGE, VALUE, and related missingness signals.",
        "",
        "One-hot encoded categorical variables such as JOB_Sales or REASON_HomeImp should be interpreted relative to the omitted baseline category. Missing-category indicators and one-hot features are predictive associations, not causal explanations.",
        "",
        f"Base value in log-odds space: {base_value:.4f}.",
    ]
    text = "\n".join(lines)
    path = output_dir / "shap_interpretation.md"
    path.write_text(text, encoding="utf-8")
    print(f"Saved interpretation: {path}")
    return text


def run_shap_analysis(
    data_path=None,
    output_dir=None,
    test_size=0.30,
    random_state=42,
    save_outputs=True,
    show_plots=False,
) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    resolved_data_path = resolve_data_path(data_path)
    resolved_output_dir = resolve_output_dir(output_dir, "outputs_shap")
    if save_outputs:
        resolved_output_dir.mkdir(exist_ok=True)

    shap_module = check_shap_available()
    if shap_module is None:
        return write_missing_dependency_note(resolved_output_dir, "shap")
    if get_xgb_classifier_class() is None:
        return write_missing_dependency_note(resolved_output_dir, "xgboost")

    print_section("Select Best Predictive Model")
    comparison = load_main_model_comparison(project_dir)
    best_row = select_best_predictive_model(comparison, PRIMARY_SELECTION_METRIC)
    selected_model = str(best_row["model_label"])
    columns = [
        col for col in [
            "model_label",
            "accuracy",
            "precision_BAD_1",
            "recall_BAD_1",
            "f1_BAD_1",
            "roc_auc",
            "average_precision",
        ] if col in comparison.columns
    ]
    print_table(comparison[columns].sort_values(PRIMARY_SELECTION_METRIC, ascending=False).round(4), "Main predictive model comparison")
    print(
        f"\nSelected model: {selected_model}, chosen by highest {PRIMARY_SELECTION_METRIC}. "
        "ROC-AUC is the primary metric because the project tunes models by ROC-AUC and it evaluates ranking quality across thresholds."
    )
    if selected_model != "XGBoost":
        print("The current saved metrics did not select XGBoost. This script is designed around the notebook's XGBoost SHAP section.")

    if save_outputs:
        save_table(comparison, resolved_output_dir, "shap_model_selection_metrics.csv")

    print_section("Train Final Tuned XGBoost")
    trained = train_final_xgb_pipeline(resolved_data_path, project_dir, test_size, random_state)
    if trained is None:
        return write_missing_dependency_note(resolved_output_dir, "xgboost")
    pipeline, X_train, X_test, y_train, y_test, model_params = trained
    metrics = evaluate_pipeline(pipeline, X_test, y_test, threshold=0.5)
    print_table(pd.DataFrame([metrics]).round(4), "Final tuned XGBoost test metrics at 0.50 threshold")
    print(f"Final XGBoost parameters: {model_params}")

    print_section("Compute SHAP Values")
    X_shap, xgb_model = transform_for_shap(pipeline, X_test)
    explainer, explanation, base_value = make_shap_explanation(shap_module, xgb_model, X_shap)
    importance_table = build_shap_importance_table(explanation)
    print_table(importance_table.head(20).round(4), "Top 20 features by mean absolute SHAP value")

    if save_outputs:
        save_table(pd.DataFrame([model_params]), resolved_output_dir, "shap_xgb_model_params.csv")
        save_table(pd.DataFrame([metrics]), resolved_output_dir, "shap_xgb_test_metrics.csv")
        save_table(importance_table, resolved_output_dir, "shap_feature_importance.csv")

    print_section("Save SHAP Plots")
    selected_indices = save_shap_plots(shap_module, explanation, X_shap, importance_table, resolved_output_dir, show_plots)

    interpretation = write_interpretation(
        resolved_output_dir,
        PRIMARY_SELECTION_METRIC,
        selected_model,
        metrics,
        importance_table,
        base_value,
    )

    if save_outputs:
        print(f"\nAll SHAP outputs saved to: {resolved_output_dir}")

    return {
        "selected_model": selected_model,
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "model_selection_table": comparison,
        "pipeline": pipeline,
        "model_params": model_params,
        "test_metrics": metrics,
        "shap_explainer": explainer,
        "shap_explanation": explanation,
        "shap_feature_importance": importance_table,
        "selected_observation_positions": selected_indices,
        "interpretation": interpretation,
        "output_dir": resolved_output_dir,
    }


if __name__ == "__main__":
    run_shap_analysis()


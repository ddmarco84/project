"""Descriptive analysis for the HMEQ loan default dataset.

This module performs exploratory data analysis only. It keeps REASON and JOB
as raw categorical variables and does not perform modelling preprocessing,
train-test splitting, dummy encoding, scaling, model fitting, or evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .preprocessing import resolve_data_path, resolve_output_dir
except ImportError:
    from preprocessing import resolve_data_path, resolve_output_dir


TARGET = "BAD"
CATEGORICAL_PREDICTORS = ["REASON", "JOB"]
DATA_DICTIONARY = pd.DataFrame({
    "Variable": [
        "BAD", "LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG",
        "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"
    ],
    "Description": [
        "Default indicator: 1 = defaulted or severely delinquent; 0 = repaid",
        "Amount of loan approved",
        "Amount due on existing mortgage",
        "Current property value",
        "Reason for loan request",
        "Applicant job type",
        "Years at present job",
        "Number of major derogatory reports",
        "Number of delinquent credit lines",
        "Age of oldest credit line in months",
        "Number of recent credit inquiries",
        "Number of existing credit lines",
        "Debt-to-income ratio",
    ],
})


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


def save_table(table: pd.DataFrame, output_dir: Path | None, filename: str) -> None:
    """Save a table as CSV when an output directory is provided."""
    if output_dir is None:
        return
    path = output_dir / filename
    table.to_csv(path, index=True)
    print(f"Saved table: {path}")


def finish_plot(filename: str, output_dir: Path | None, show_plots: bool, save_plots: bool) -> None:
    """Apply final layout, optionally save, optionally show, then close the figure."""
    plt.tight_layout()
    if save_plots and output_dir is not None:
        path = output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {path}")
    if show_plots:
        plt.show()
    plt.close()


def categorical_summary_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return counts and percentages for a categorical variable, including missing values."""
    counts = data[column].fillna("Missing").value_counts(dropna=False)
    summary = counts.rename_axis(column).reset_index(name="count")
    summary["percent"] = (summary["count"] / len(data) * 100).round(2)
    return summary


def default_rate_table(data: pd.DataFrame, column: str, target: str = TARGET) -> pd.DataFrame:
    """Return default rates by category, including missing values as an explicit category."""
    temp = data[[column, target]].copy()
    temp[column] = temp[column].fillna("Missing")
    summary = (
        temp.groupby(column, dropna=False)[target]
        .agg(count="size", default_rate="mean")
        .reset_index()
    )
    summary["default_rate_%"] = summary["default_rate"].mul(100).round(2)
    return summary.sort_values("default_rate_%", ascending=False)


def plot_target_distribution(
    target_balance: pd.DataFrame,
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
) -> None:
    """Plot the target-class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_data = target_balance.copy()
    plot_data[TARGET] = plot_data[TARGET].astype(str)
    sns.barplot(data=plot_data, x=TARGET, y="count", ax=ax, color="#4C78A8")
    ax.set_title("Distribution of Loan Outcomes")
    ax.set_xlabel("BAD (0 = repaid, 1 = defaulted/severely delinquent)")
    ax.set_ylabel("Number of clients")
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{p:.1f}%" for p in plot_data["percent"]], padding=3)
    finish_plot("target_distribution_bad.png", output_dir, show_plots, save_plots)


def plot_hist_grid(
    data: pd.DataFrame,
    columns: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
    bins: int = 30,
    cols: int = 3,
) -> None:
    """Plot histograms for selected numerical variables."""
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax, column in zip(axes.ravel(), columns):
        sns.histplot(data=data, x=column, bins=bins, kde=True, ax=ax, color="#4C78A8")
        ax.axvline(data[column].median(), color="#222222", linestyle="--", linewidth=1, label="Median")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.legend()

    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)

    finish_plot("numerical_distributions.png", output_dir, show_plots, save_plots)


def plot_box_grid(
    data: pd.DataFrame,
    columns: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
    cols: int = 3,
) -> None:
    """Plot boxplots for selected numerical variables to inspect outliers."""
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax, column in zip(axes.ravel(), columns):
        sns.boxplot(data=data, x=column, ax=ax, color="#72B7B2")
        ax.set_title(f"Boxplot of {column}")
        ax.set_xlabel(column)

    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)

    finish_plot("numerical_boxplots.png", output_dir, show_plots, save_plots)


def plot_categorical_distributions(
    data: pd.DataFrame,
    categorical_predictors: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
) -> None:
    """Plot raw categorical-variable distributions."""
    fig, axes = plt.subplots(1, len(categorical_predictors), figsize=(13, 4.5))
    axes = np.array(axes).reshape(-1)

    for ax, column in zip(axes, categorical_predictors):
        cat_data = categorical_summary_table(data, column)
        sns.barplot(data=cat_data, x=column, y="count", ax=ax, color="#4C78A8")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of clients")
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, labels=[f"{p:.1f}%" for p in cat_data["percent"]], padding=3, fontsize=9)

    finish_plot("categorical_distributions.png", output_dir, show_plots, save_plots)


def plot_by_target_box_grid(
    data: pd.DataFrame,
    columns: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
    target: str = TARGET,
    cols: int = 3,
) -> None:
    """Plot selected numerical variables by target class."""
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax, column in zip(axes.ravel(), columns):
        sns.boxplot(data=data, x=target, y=column, hue=target, ax=ax, palette="Set2", legend=False)
        ax.set_title(f"{column} by default status")
        ax.set_xlabel("BAD (0 = repaid, 1 = defaulted)")
        ax.set_ylabel(column)

    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)

    finish_plot("numerical_boxplots_by_bad.png", output_dir, show_plots, save_plots)


def plot_target_hist_grid(
    data: pd.DataFrame,
    columns: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
    target: str = TARGET,
    bins: int = 30,
    cols: int = 3,
) -> None:
    """Plot numerical distributions split by target class."""
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    plot_data = data.copy()
    plot_data[target] = plot_data[target].map({0: "Repaid", 1: "Defaulted"})

    for ax, column in zip(axes.ravel(), columns):
        sns.histplot(
            data=plot_data,
            x=column,
            hue=target,
            bins=bins,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title(f"{column} distribution by loan outcome")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")

    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)

    finish_plot("numerical_distributions_by_bad.png", output_dir, show_plots, save_plots)


def plot_default_rates(
    data: pd.DataFrame,
    categorical_predictors: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
    target: str = TARGET,
) -> None:
    """Plot default rates for categorical predictors."""
    fig, axes = plt.subplots(1, len(categorical_predictors), figsize=(13, 4.5))
    axes = np.array(axes).reshape(-1)

    for ax, column in zip(axes, categorical_predictors):
        rate_data = default_rate_table(data, column, target=target)
        sns.barplot(data=rate_data, x=column, y="default_rate_%", ax=ax, color="#F58518")
        ax.set_title(f"Default rate by {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Default rate (%)")
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)

    finish_plot("categorical_default_rates.png", output_dir, show_plots, save_plots)


def plot_correlation_heatmap(
    data: pd.DataFrame,
    columns: list[str],
    output_dir: Path | None,
    show_plots: bool,
    save_plots: bool,
) -> None:
    """Plot correlations among numerical predictors only."""
    corr = data[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap for Numerical Predictors")
    finish_plot("numerical_correlation_heatmap.png", output_dir, show_plots, save_plots)


def run_descriptive_analysis(
    data_path=None,
    output_dir=None,
    show_plots=True,
    save_plots=True,
):
    """Run descriptive analysis for the HMEQ dataset.

    Parameters
    ----------
    data_path : str or pathlib.Path, optional
        Path to hmeq.csv. Defaults to hmeq.csv in the same folder as this script.
    output_dir : str or pathlib.Path, optional
        Folder for saved tables and plots. Defaults to outputs/outputs_descr in the project root.
    show_plots : bool, default=True
        Whether to display plots interactively.
    save_plots : bool, default=True
        Whether to save plots and summary tables to output_dir.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The raw dataframe copy and a dictionary of summary tables/metadata.
    """
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    resolved_data_path = resolve_data_path(data_path)
    resolved_output_dir = resolve_output_dir(output_dir, "outputs_descr")

    if save_plots:
        resolved_output_dir.mkdir(exist_ok=True)
    else:
        resolved_output_dir = None

    pd.options.display.float_format = "{:,.2f}".format
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    sns.set_theme(style="whitegrid", context="notebook")

    if not resolved_data_path.exists():
        raise FileNotFoundError(f"Could not find hmeq.csv at {resolved_data_path}")

    df_raw = pd.read_csv(resolved_data_path)
    df = df_raw.copy()
    numerical_predictors = [col for col in df.columns if col not in CATEGORICAL_PREDICTORS + [TARGET]]

    first_rows = df.head()
    column_overview = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null_count": df.notna().sum().values,
        "missing_count": df.isna().sum().values,
        "missing_%": df.isna().mean().mul(100).round(2).values,
    })
    target_balance = (
        df[TARGET]
        .value_counts(dropna=False)
        .rename_axis(TARGET)
        .reset_index(name="count")
    )
    target_balance["percent"] = (target_balance["count"] / len(df) * 100).round(2)
    target_balance["class_label"] = target_balance[TARGET].map({0: "Repaid", 1: "Defaulted / severely delinquent"})
    target_balance = target_balance[[TARGET, "class_label", "count", "percent"]]

    missing_summary = (
        df.isna()
        .agg(["sum", "mean"])
        .T
        .rename(columns={"sum": "missing_count", "mean": "missing_%"})
    )
    missing_summary["missing_%"] = missing_summary["missing_%"].mul(100).round(2)
    missing_summary = missing_summary.sort_values("missing_%", ascending=False)

    numerical_summary = df[numerical_predictors].describe().T
    numerical_summary["missing_%"] = df[numerical_predictors].isna().mean().mul(100).round(2)
    numerical_summary["skew"] = df[numerical_predictors].skew(numeric_only=True).round(2)

    categorical_summaries = {
        column: categorical_summary_table(df, column)
        for column in CATEGORICAL_PREDICTORS
    }
    categorical_default_rates = {
        column: default_rate_table(df, column, target=TARGET)
        for column in CATEGORICAL_PREDICTORS
    }

    target_correlations = (
        df[numerical_predictors + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .to_frame("correlation_with_BAD")
    )
    target_correlations["abs_correlation"] = target_correlations["correlation_with_BAD"].abs()
    target_correlations = (
        target_correlations
        .sort_values("abs_correlation", ascending=False)
        .drop(columns="abs_correlation")
    )
    numerical_correlations = df[numerical_predictors].corr(numeric_only=True)

    outputs = {
        "data_path": resolved_data_path,
        "output_dir": resolved_output_dir,
        "shape": df.shape,
        "first_rows": first_rows,
        "column_overview": column_overview,
        "data_dictionary": DATA_DICTIONARY.copy(),
        "target_balance": target_balance,
        "missing_summary": missing_summary,
        "numerical_predictors": numerical_predictors,
        "categorical_predictors": CATEGORICAL_PREDICTORS.copy(),
        "numerical_summary": numerical_summary,
        "categorical_summaries": categorical_summaries,
        "categorical_default_rates": categorical_default_rates,
        "target_correlations": target_correlations,
        "numerical_correlations": numerical_correlations,
    }

    print_section("Basic Data Inspection")
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print_table(first_rows, "First five rows")
    print("\nColumn names:")
    print(list(df.columns))
    print_table(column_overview, "Column overview")
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print_section("Data Dictionary")
    print_table(DATA_DICTIONARY)

    print_section("Target Variable Analysis")
    print_table(target_balance, "BAD class balance")
    print("\nComment: Defaults represent about one fifth of the observations, so later model evaluation should account for class imbalance.")

    print_section("Missing-Value Analysis")
    print_table(missing_summary)
    print("\nComment: DEBTINC has the highest missingness and should receive special attention during later modelling.")

    print_section("Numerical Predictor Analysis")
    print("Target variable:", TARGET)
    print("Categorical predictors:", CATEGORICAL_PREDICTORS)
    print("Numerical predictors:", numerical_predictors)
    print_table(numerical_summary, "Numerical summary statistics")
    print("\nComment: Several monetary and credit-history variables are right-skewed and contain high-end outliers.")

    print_section("Categorical Predictor Analysis")
    for column, summary in categorical_summaries.items():
        print_table(summary, f"{column} counts and percentages")
    print("\nComment: Categorical variables are kept in raw form; no dummy encoding is performed in this script.")

    print_section("Bivariate Analysis With BAD")
    for column, summary in categorical_default_rates.items():
        print_table(summary, f"Default rate by {column}")
    print_table(target_correlations, "Numerical correlations with BAD")
    print("\nComment: Credit-history variables such as DEROG, DELINQ, NINQ, and DEBTINC appear associated with default status.")

    if save_plots and resolved_output_dir is not None:
        save_table(column_overview, resolved_output_dir, "column_overview.csv")
        save_table(DATA_DICTIONARY, resolved_output_dir, "data_dictionary.csv")
        save_table(target_balance, resolved_output_dir, "target_balance.csv")
        save_table(missing_summary, resolved_output_dir, "missing_summary.csv")
        save_table(numerical_summary, resolved_output_dir, "numerical_summary.csv")
        save_table(target_correlations, resolved_output_dir, "target_correlations.csv")
        save_table(numerical_correlations, resolved_output_dir, "numerical_correlations.csv")
        for column, summary in categorical_summaries.items():
            save_table(summary, resolved_output_dir, f"{column.lower()}_summary.csv")
        for column, summary in categorical_default_rates.items():
            save_table(summary, resolved_output_dir, f"{column.lower()}_default_rates.csv")

    print_section("Plot Generation")
    key_numeric = ["LOAN", "MORTDUE", "VALUE", "DEBTINC", "YOJ", "CLAGE"]
    risk_numeric = ["LOAN", "MORTDUE", "VALUE", "DEBTINC", "DEROG", "DELINQ", "NINQ", "CLNO", "CLAGE"]

    plot_target_distribution(target_balance, resolved_output_dir, show_plots, save_plots)
    plot_hist_grid(df, key_numeric, resolved_output_dir, show_plots, save_plots, bins=35, cols=3)
    plot_box_grid(df, key_numeric, resolved_output_dir, show_plots, save_plots, cols=3)
    plot_categorical_distributions(df, CATEGORICAL_PREDICTORS, resolved_output_dir, show_plots, save_plots)
    plot_by_target_box_grid(df, risk_numeric, resolved_output_dir, show_plots, save_plots, target=TARGET, cols=3)
    plot_target_hist_grid(
        df,
        ["DEBTINC", "DEROG", "DELINQ", "NINQ", "CLAGE", "YOJ"],
        resolved_output_dir,
        show_plots,
        save_plots,
        target=TARGET,
        bins=30,
        cols=3,
    )
    plot_default_rates(df, CATEGORICAL_PREDICTORS, resolved_output_dir, show_plots, save_plots, target=TARGET)
    plot_correlation_heatmap(df, numerical_predictors, resolved_output_dir, show_plots, save_plots)

    print_section("EDA Summary")
    print(
        "The descriptive analysis confirms class imbalance, substantial missingness in DEBTINC, "
        "right-skewness and high-end outliers in several numerical variables, and visible "
        "associations between BAD and credit-history variables. Categorical predictors REASON "
        "and JOB were analysed in their original form."
    )
    if save_plots and resolved_output_dir is not None:
        print(f"\nAll tables and plots saved to: {resolved_output_dir}")

    return df, outputs


if __name__ == "__main__":
    run_descriptive_analysis()


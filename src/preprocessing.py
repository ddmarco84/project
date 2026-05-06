"""Shared preprocessing utilities for the HMEQ loan-default project."""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "BAD"
CATEGORICAL_PREDICTORS = ["REASON", "JOB"]
MISSING_INDICATOR_COLUMNS = ["VALUE", "MORTDUE", "DEBTINC"]


def get_project_dir() -> Path:
    """Return the project root when called from modules inside src."""
    return Path(__file__).resolve().parent.parent


def resolve_data_path(data_path=None) -> Path:
    """Resolve the HMEQ data path, preferring data/hmeq.csv in the project root."""
    if data_path is not None:
        return Path(data_path)

    project_dir = get_project_dir()
    default_data_path = project_dir / "data" / "hmeq.csv"
    if default_data_path.exists():
        return default_data_path
    return project_dir / "hmeq.csv"


def resolve_output_dir(output_dir=None, folder_name: str = "outputs") -> Path:
    """Resolve an output folder inside the consolidated outputs directory."""
    if output_dir is not None:
        return Path(output_dir)
    return get_project_dir() / "outputs" / folder_name


def load_hmeq_data(data_path=None) -> pd.DataFrame:
    """Load the raw HMEQ dataset."""
    return pd.read_csv(resolve_data_path(data_path))


def split_predictors_target(
    df: pd.DataFrame,
    target: str = TARGET,
    categorical_predictors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Separate predictors, target, numerical predictors, and categorical predictors."""
    categorical_predictors = categorical_predictors or CATEGORICAL_PREDICTORS
    X = df.drop(columns=[target])
    y = df[target]
    numerical_predictors = [col for col in X.columns if col not in categorical_predictors]
    return X, y, numerical_predictors, categorical_predictors


def make_one_hot_encoder(drop: str | None = "first") -> OneHotEncoder:
    """Create a OneHotEncoder compatible with recent and older sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop=drop, sparse=False)


def build_preprocessor(
    numerical_predictors: list[str],
    categorical_predictors: list[str],
    *,
    scale_numeric: bool = False,
    one_hot_drop: str | None = "first",
    missing_indicator_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build leakage-free preprocessing for model pipelines."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", make_one_hot_encoder(drop=one_hot_drop)),
    ])

    indicator_candidates = missing_indicator_columns or MISSING_INDICATOR_COLUMNS
    indicator_columns = [col for col in indicator_candidates if col in numerical_predictors]

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_predictors),
            ("cat", categorical_pipeline, categorical_predictors),
            ("miss", MissingIndicator(features="missing-only"), indicator_columns),
        ],
        remainder="drop",
    )


def clean_feature_name(name: str) -> str:
    """Make transformed feature names easier to read."""
    is_missing_indicator = "missingindicator_" in name
    name = name.replace("num__", "").replace("cat__", "").replace("miss__", "")
    name = name.replace("missingindicator_", "")
    if is_missing_indicator:
        return f"{name}_missing"
    return name


def get_feature_names(fitted_pipeline) -> list[str]:
    """Recover transformed feature names from a fitted preprocessing pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]

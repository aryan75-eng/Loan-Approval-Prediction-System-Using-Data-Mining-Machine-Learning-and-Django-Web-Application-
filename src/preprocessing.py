"""
preprocessing.py — Defines the ColumnTransformer used in both training and serving.

IMPORTANT: This module only *defines* the preprocessor. It must be fit inside
a Pipeline in model_training.py so the fitted transformer is serialised with
the model. Never re-create the transformer at prediction time.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that:
      - numeric  → median impute → StandardScaler
      - categorical → most_frequent impute → OneHotEncoder

    The transformer is NOT fit here. Fit happens in model_training.py on the
    training split only. The same fitted object is serialised inside the Pipeline.

    Returns
    -------
    ColumnTransformer
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",   # unseen categories → all-zero row
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",          # silently drop any extra columns
        verbose_feature_names_out=True,
    )

    return preprocessor

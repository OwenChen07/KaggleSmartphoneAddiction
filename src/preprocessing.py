"""Leakage-safe preprocessing.

Every transformer here is returned *unfitted*, to be composed into a Pipeline
and fit inside a CV fold by `validation.run_cv`. Nothing in this module ever
touches the full training set, and nothing is fit on data that includes the
rows it will be evaluated on.

Two variants exist because the model families genuinely need different things:

- `tree_preprocessor` leaves numerics untouched, NaN included.
  HistGradientBoostingClassifier learns a default split direction for missing
  values natively, so imputing would *discard* information rather than add it.
- `linear_preprocessor` must impute and scale, because logistic regression
  cannot consume NaN and is sensitive to feature scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)

from .data import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS

MISSING_TOKEN = "__MISSING__"


def _isna_flags(X: pd.DataFrame) -> np.ndarray:
    """Is-missing indicators for every column passed in.

    Hand-rolled rather than `sklearn.impute.MissingIndicator` because that
    transformer defaults to emitting columns only for features that had
    missing values *in the fold it was fit on* — which lets the feature count
    drift between folds. This always emits one column per input column.
    """
    return X.isna().to_numpy(dtype=np.int8)


def _isna_names(transformer, input_features) -> np.ndarray:
    """Suffix the indicator columns so they do not collide with the names of
    the source columns in `get_feature_names_out`."""
    return np.asarray([f"{c}__isna" for c in input_features], dtype=object)


def missing_indicator_block() -> FunctionTransformer:
    return FunctionTransformer(_isna_flags, feature_names_out=_isna_names)


def _categorical_pipeline(encoder) -> Pipeline:
    """Missingness in the categoricals (4-8% per column) is treated as its own
    level rather than imputed to the mode — if it is informative the model can
    use it, and if it is not the split simply never gets chosen."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("encode", encoder),
        ]
    )


def tree_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    missing_indicators: bool = False,
) -> ColumnTransformer:
    """Preprocessing for tree models: numerics passed through with NaN intact."""
    numeric_cols = NUMERIC_COLS if numeric_cols is None else numeric_cols
    categorical_cols = CATEGORICAL_COLS if categorical_cols is None else categorical_cols

    blocks = [("num", "passthrough", numeric_cols)]
    if categorical_cols:
        blocks.append(
            (
                "cat",
                _categorical_pipeline(
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                ),
                categorical_cols,
            )
        )
    if missing_indicators:
        blocks.append(("isna", missing_indicator_block(), numeric_cols))

    return ColumnTransformer(blocks, verbose_feature_names_out=False)


def linear_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    missing_indicators: bool = True,
) -> ColumnTransformer:
    """Preprocessing for linear models: median-impute + scale, one-hot cats.

    Missing indicators default to on here: median imputation destroys the
    missingness signal, so without them a linear model cannot represent it at
    all.

    They cover the *numeric* columns only. A categorical is imputed to an
    explicit `__MISSING__` level which one-hot encoding then gives its own
    column, so an is-missing indicator for the same column would be an exact
    duplicate of it — perfectly collinear, and enough to make the lbfgs solver
    emit overflow warnings during line search.
    """
    numeric_cols = NUMERIC_COLS if numeric_cols is None else numeric_cols
    categorical_cols = CATEGORICAL_COLS if categorical_cols is None else categorical_cols

    blocks = [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_cols,
        )
    ]
    if categorical_cols:
        blocks.append(
            (
                "cat",
                _categorical_pipeline(
                    # drop="first" avoids the dummy-variable trap: without it
                    # each one-hot group sums to the all-ones vector, leaving
                    # the design matrix rank-deficient by (n_groups - 1).
                    # Unknown categories cannot arise here — missing values are
                    # already imputed to an explicit level and the source
                    # columns are closed 3-level sets.
                    OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
                ),
                categorical_cols,
            )
        )
    if missing_indicators:
        blocks.append(("isna", missing_indicator_block(), numeric_cols))

    return ColumnTransformer(blocks, verbose_feature_names_out=False)


def missingness_only_preprocessor() -> ColumnTransformer:
    """Discards the feature values entirely and keeps only the is-missing
    pattern. Used to test whether missingness alone predicts the target."""
    return ColumnTransformer(
        [("isna", missing_indicator_block(), FEATURE_COLS)], verbose_feature_names_out=False
    )


def nn_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    missing_indicators: bool = True,
) -> ColumnTransformer:
    """Preprocessing for a neural net: rank-normalise, then one-hot.

    Not `linear_preprocessor`, for one reason: `StandardScaler`. Many of the
    engineered columns are *ratios*, so their denominators approach zero and
    the distributions carry heavy tails. Standardising divides by a standard
    deviation those tails inflate, which leaves the bulk of the mass squeezed
    into a narrow band near zero. A linear model does not care — it only ever
    forms a weighted sum, and a scale change is absorbed by its coefficient.
    A neural net does care: the first layer's activations saturate, and the
    gradient through a saturated unit is close to zero, so the tails decide
    the learning rate for every row.

    `QuantileTransformer` maps each column onto its own rank and then through
    the normal quantile function, so the output is Gaussian by construction no
    matter how skewed the input. It is a monotone, per-column transform, which
    is exactly the class of transform ROC AUC cannot see; it therefore changes
    what the *network* can fit without changing what the metric would reward
    from any single column on its own.

    Rank transforms are fit-dependent, so this is only safe inside the fold --
    which is what returning it unfitted enforces.
    """
    numeric_cols = NUMERIC_COLS if numeric_cols is None else numeric_cols
    categorical_cols = CATEGORICAL_COLS if categorical_cols is None else categorical_cols

    blocks = [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "scale",
                        QuantileTransformer(
                            n_quantiles=1000,
                            output_distribution="normal",
                            subsample=200_000,
                            random_state=0,
                        ),
                    ),
                ]
            ),
            numeric_cols,
        )
    ]
    if categorical_cols:
        blocks.append(
            (
                "cat",
                _categorical_pipeline(
                    OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
                ),
                categorical_cols,
            )
        )
    if missing_indicators:
        blocks.append(("isna", missing_indicator_block(), numeric_cols))

    return ColumnTransformer(blocks, verbose_feature_names_out=False)

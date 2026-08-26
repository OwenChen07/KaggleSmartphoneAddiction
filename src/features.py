"""Engineered features for the screen-time columns.

Three columns carry nearly all the univariate signal (`daily_screen_time_hours`
0.890, `weekend_screen_time` 0.881, `social_media_hours` 0.858). A gradient
boosted tree can already split on each of them, so a new feature only earns its
place if it expresses something splitting on the raw columns *cannot* reach.

That constraint rules more out than it first appears:

- A tree splits on `x <= t`. Any strictly monotonic function of a single
  column therefore yields exactly the same partition of the rows, just with
  relabelled thresholds. `8 - sleep_hours` and `24 - daily_screen_time_hours`
  cannot add anything to a tree, however meaningful "sleep deficit" is to a
  human reading the column list. `sleep_deficit` is included below anyway, as
  a deliberate control: it is the one feature we can predict will score zero
  on permutation importance before running anything, and checking that it does
  is a test of the measurement, not of the feature.
- Reciprocals are monotonic too, on positive data. `daily_screen / app_opens`
  and `app_opens / daily_screen` are the same feature to a tree, so only one
  of each such pair appears here.

What a tree genuinely cannot reach cheaply is a *ratio of two columns*. To
approximate `a / b <= t` with axis-aligned splits it must stair-step the
boundary with many splits, spending depth and rows at every step. Handing it
the quotient directly turns that staircase into one split. Every feature below
is either a ratio, a difference, or a product of two or more columns.

Everything here is stateless: the transformer computes row-wise arithmetic and
learns nothing from the data it is fit on. `fit` genuinely has nothing to do.
That is the strongest possible position on leakage — there is no fitted
quantity that *could* carry information across a fold boundary — but the class
is still a proper transformer so it sits inside the Pipeline and is composed,
cloned and fit per fold like everything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import CATEGORICAL_COLS, NUMERIC_COLS

#: Engineered column names, in the order `EngineeredFeatures` emits them.
ENGINEERED_COLS = [
    "social_share",
    "gaming_share",
    "work_share",
    "weekend_ratio",
    "weekend_gap",
    "session_length",
    "notif_per_open",
    "notif_per_screen_hour",
    "tracked_screen",
    "residual_screen",
    "waking_load",
    "screen_sleep_ratio",
    "age_x_screen",
    "screen_per_age",
    "sleep_deficit",
]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN, never inf, on a zero denominator.

    On this data no denominator used below actually reaches zero — the minima
    are 0.5 hours of daily screen time, 15 app opens and age 18 — so this guard
    should never fire. It is here because "should never fire" is an assumption
    about the *training* data, and an inf reaching a HistGBM bin mapper would
    be a silent corruption rather than a loud failure. Explicitly NaN instead:
    NaN is a value the model already knows how to route.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """Append engineered columns to the raw feature frame.

    Missing inputs propagate: a ratio whose numerator or denominator is NaN is
    NaN, and is left that way rather than imputed. Missing rates run 4-19% per
    column, so a two-column ratio is missing for roughly a quarter to a third
    of rows. Imputing would invent a value for a third of the column;
    HistGradientBoosting instead learns a default routing direction for NaN at
    every split, which is strictly more information than a filled-in median.
    """

    def __init__(self, columns: list[str] | None = None):
        # `columns` selects a subset of ENGINEERED_COLS. Stored unmodified and
        # unvalidated here: sklearn's clone() contract requires __init__ to
        # assign its arguments through untouched, so validation happens in fit.
        self.columns = columns

    def _selected(self) -> list[str]:
        return list(ENGINEERED_COLS) if self.columns is None else list(self.columns)

    def fit(self, X: pd.DataFrame, y=None):
        unknown = set(self._selected()) - set(ENGINEERED_COLS)
        if unknown:
            raise ValueError(f"unknown engineered columns: {sorted(unknown)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        screen = X["daily_screen_time_hours"]
        opens = X["app_opens_per_day"]

        # How the day's screen time is composed. The raw hour counts and the
        # shares are different questions: two hours of social media means
        # something different at three hours of total screen time than at ten.
        built = {
            "social_share": _safe_divide(X["social_media_hours"], screen),
            "gaming_share": _safe_divide(X["gaming_hours"], screen),
            "work_share": _safe_divide(X["work_study_hours"], screen),
            # Weekend vs weekday, as a ratio and as a difference. They are not
            # monotonic in one another (the ratio normalises by weekday level,
            # the gap does not), so a tree can use both.
            "weekend_ratio": _safe_divide(X["weekend_screen_time"], screen),
            "weekend_gap": X["weekend_screen_time"] - screen,
            # Usage texture: hours per app open is average session length,
            # notifications per open is how much prompting each visit needs.
            "session_length": _safe_divide(screen, opens),
            "notif_per_open": _safe_divide(X["notifications_per_day"], opens),
            "notif_per_screen_hour": _safe_divide(X["notifications_per_day"], screen),
            # Named activities against the total. `residual_screen` is the
            # screen time the itemised columns do not explain.
            "tracked_screen": X["social_media_hours"] + X["gaming_hours"],
            "residual_screen": screen - X["social_media_hours"] - X["gaming_hours"],
            # Time budget against a 24h day. Only the sum is kept: `24 - sum`
            # is monotonic in it and so is the same feature to a tree.
            "waking_load": screen + X["sleep_hours"] + X["work_study_hours"],
            "screen_sleep_ratio": _safe_divide(screen, X["sleep_hours"]),
            # Age interactions with the strongest column, as a product and as a
            # ratio — the same hours mean something different at 18 than at 35.
            "age_x_screen": X["age"] * screen,
            "screen_per_age": _safe_divide(screen, X["age"]),
            # Control feature, expected to be worthless: strictly decreasing in
            # sleep_hours alone, so it induces the identical partition.
            "sleep_deficit": 8.0 - X["sleep_hours"],
        }

        for name in self._selected():
            X[name] = built[name].astype(np.float32)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.asarray([*base, *self._selected()], dtype=object)


def engineered_numeric_cols(columns: list[str] | None = None) -> list[str]:
    """Numeric column list for a `tree_preprocessor` sitting downstream of
    `EngineeredFeatures` — the raw numerics plus whichever engineered ones the
    transformer was asked to emit."""
    selected = list(ENGINEERED_COLS) if columns is None else list(columns)
    return [*NUMERIC_COLS, *selected]


__all__ = [
    "ENGINEERED_COLS",
    "EngineeredFeatures",
    "engineered_numeric_cols",
    "CATEGORICAL_COLS",
]


# --------------------------------------------------------------------------
# Phase 8 — imputing the recoverable columns
# --------------------------------------------------------------------------

#: Columns worth reconstructing, from the Stage 0 predictability gate
#: (reports/imputation.md). Each is predictable from the other 11 features
#: with R2 >= 0.50 on held-out complete rows.
RECOVERABLE_COLS = [
    "daily_screen_time_hours",  # R2 0.8716
    "weekend_screen_time",      # R2 0.6648
    "social_media_hours",       # R2 0.6068
    "work_study_hours",         # R2 0.5448
    "gaming_hours",             # R2 0.5013
]

#: Deliberately NOT imputed: age (R2 0.0038), sleep_hours (0.0245),
#: notifications_per_day (0.0228), app_opens_per_day (0.0198). These four are
#: mutually independent of everything else — the largest correlation among all
#: remaining column pairs is 0.086 — so a "prediction" for them is the column
#: mean wearing a disguise. Filling them would replace an honest NaN, which
#: HistGBM routes deliberately, with a fabricated constant it cannot tell apart
#: from a real measurement. That is strictly worse than leaving the hole.


class RecoverableImputer(BaseEstimator, TransformerMixin):
    """Reconstruct missing values in the mutually-predictable screen-time columns.

    Unlike `EngineeredFeatures`, this transformer is **stateful** — it fits one
    regressor per target column and carries them into `transform`. That makes
    fold discipline load-bearing rather than decorative: fitted on the full
    training set before splitting, it would leak validation rows into the
    imputations used to score those same rows, and every OOF number downstream
    would be fiction. It is therefore a step *inside* the Pipeline, cloned and
    refit per fold by `run_cv`, exactly like the estimator itself.

    Each column's regressor is trained on the rows where that column is
    **present**, predicting it from the other 11 raw features. The predictors
    themselves may be missing; HistGradientBoostingRegressor routes NaN
    natively, so no cascading imputation is needed.

    Training on present-rows and applying to missing-rows assumes the two
    groups are comparable. Phase 3 established exactly that: a model trained on
    nothing but the is-missing flags scores 0.50038, and target rates between
    missing and present rows differ by at most 0.0042. The values are missing
    at random, so the regressor is not being asked to extrapolate.

    `mode="replace"` fills the holes in place, so the ratios built downstream
    by `EngineeredFeatures` become computable for rows that previously produced
    NaN. `mode="augment"` keeps the original NaN-bearing columns and appends
    the reconstructions alongside, letting the model use both and keeping the
    original missingness visible.
    """

    def __init__(self, columns: list[str] | None = None, mode: str = "replace",
                 max_iter: int = 200, random_state: int = 42):
        self.columns = columns
        self.mode = mode
        self.max_iter = max_iter
        self.random_state = random_state

    def _targets(self) -> list[str]:
        return list(RECOVERABLE_COLS) if self.columns is None else list(self.columns)

    def fit(self, X: pd.DataFrame, y=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline as _Pipeline

        from .preprocessing import tree_preprocessor

        if self.mode not in {"replace", "augment"}:
            raise ValueError("mode must be 'replace' or 'augment'")
        unknown = set(self._targets()) - set(NUMERIC_COLS)
        if unknown:
            raise ValueError(f"not raw numeric columns: {sorted(unknown)}")

        X = pd.DataFrame(X)
        self.models_ = {}
        for col in self._targets():
            predictors = [c for c in NUMERIC_COLS if c != col]
            present = X[col].notna().to_numpy()
            model = _Pipeline([
                ("prep", tree_preprocessor(
                    numeric_cols=predictors, categorical_cols=list(CATEGORICAL_COLS))),
                ("reg", HistGradientBoostingRegressor(
                    random_state=self.random_state, max_iter=self.max_iter)),
            ])
            model.fit(X.loc[present], X.loc[present, col])
            self.models_[col] = model
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col, model in self.models_.items():
            missing = X[col].isna().to_numpy()
            if not missing.any():
                if self.mode == "augment":
                    X[f"{col}__imp"] = X[col].astype(np.float32)
                continue
            pred = model.predict(X.loc[missing]).astype(np.float32)
            if self.mode == "replace":
                X.loc[missing, col] = pred
            else:
                filled = X[col].astype(np.float32).copy()
                filled.loc[missing] = pred
                X[f"{col}__imp"] = filled
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        extra = [] if self.mode == "replace" else [f"{c}__imp" for c in self._targets()]
        return np.asarray([*base, *extra], dtype=object)


def imputed_numeric_cols(mode: str = "replace", columns: list[str] | None = None,
                         engineered: list[str] | None = None) -> list[str]:
    """Numeric column list for a `tree_preprocessor` downstream of both
    `RecoverableImputer` and `EngineeredFeatures`."""
    targets = list(RECOVERABLE_COLS) if columns is None else list(columns)
    extra = [] if mode == "replace" else [f"{c}__imp" for c in targets]
    return [*engineered_numeric_cols(engineered), *extra]

"""CatBoost with the lookup keys handed over as raw categorical levels.

Phase 9 established that several columns are lookup keys rather than
quantities, and that reading them as such is worth +0.0039. `src/encoding.py`
does that with a hand-rolled encoder: a smoothed target mean per level, one
global smoothing constant, nested folds to keep it honest.

CatBoost has computed **ordered target statistics** internally since it
existed. Given a column as `cat_features`, it builds the same kind of
statistic — but under a random permutation, using only the rows that precede
each row in that permutation. Two consequences:

- the shrinkage is **adaptive per row** rather than one constant for every
  level: early rows in the permutation get heavily shrunk estimates because
  little history precedes them, later rows get sharper ones;
- the leakage protection is structural rather than something we implement,
  since a row's own target is never in its own statistic by construction.

So this is not a second opinion from a different library. It is the same idea
with a better estimator, and the reason to try it is that a hand-rolled
encoder with a single smoothing constant is the crude version of what CatBoost
already does properly.

The numeric block still carries the engineered ratios, the decimal lattice and
the accounting residual — CatBoost's categorical handling replaces
`TargetFrequencyEncoder`, not `EngineeredFeatures` or `GeneratorFeatures`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .data import CATEGORICAL_COLS, NUMERIC_COLS
from .encoding import ENCODE_COLS, level_frame
from .features import (
    ENGINEERED_COLS,
    GENERATOR_COLS,
    EngineeredFeatures,
    GeneratorFeatures,
)
from .validation import SEED

CAT_DEFAULTS = dict(
    iterations=2000,
    learning_rate=0.06,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False,
)


class CatBoostLookup(BaseEstimator, ClassifierMixin):
    """Numeric features plus every column repeated as a raw string level.

    Deliberately a single estimator rather than a Pipeline: CatBoost needs the
    categorical columns identified by position at `fit` time, and a
    ColumnTransformer in front would convert the frame to a numeric array and
    destroy exactly the information being handed over.

    Each of the twelve columns appears **twice** — once as its numeric value in
    the numeric block, once as a string level in the categorical block. That is
    not redundancy: the first supports ordering questions (`screen time above
    9.8 hours`), the second supports identity questions (`this exact value`).
    Phase 9 showed the second is where most of the signal is, and the first is
    still what carries the genuinely continuous structure.
    """

    def __init__(self, params: dict | None = None, use_lattice: bool = True):
        self.params = params
        self.use_lattice = use_lattice

    def _frame(self, X: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
        X = pd.DataFrame(X).reset_index(drop=True)
        feats = EngineeredFeatures().fit(X).transform(X)
        numeric = [*NUMERIC_COLS, *ENGINEERED_COLS]
        if self.use_lattice:
            feats = GeneratorFeatures().fit(feats).transform(feats)
            numeric = [*numeric, *GENERATOR_COLS]
        num = feats[numeric].replace([np.inf, -np.inf], np.nan)
        lvl = level_frame(X, ENCODE_COLS).add_prefix("lvl_")
        out = pd.concat([num.reset_index(drop=True), lvl.reset_index(drop=True)], axis=1)
        cat_idx = [out.columns.get_loc(c) for c in lvl.columns]
        return out, cat_idx

    def fit(self, X: pd.DataFrame, y):
        from catboost import CatBoostClassifier

        frame, cat_idx = self._frame(X)
        self.classes_ = np.unique(y)
        self.cat_idx_ = cat_idx
        params = {**CAT_DEFAULTS, **(self.params or {})}
        self.model_ = CatBoostClassifier(**params)
        self.model_.fit(frame, np.asarray(y), cat_features=cat_idx)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame, _ = self._frame(X)
        return self.model_.predict_proba(frame)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def catboost_lookup() -> CatBoostLookup:
    return CatBoostLookup()

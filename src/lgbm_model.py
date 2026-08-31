"""LightGBM as a third family, for blend diversity rather than solo strength.

The blend gained +0.00044 from adding one diverse family (CatBoost beside
HistGBM, Spearman 0.9865). Phase 6 established the condition under which that
pays: members must be **comparably strong** *and* **genuinely different**.
This tests whether a third distinct mechanism adds again.

The three families now attack the lookup-key problem three different ways:

- `src/encoding.py` — a hand-rolled nested target encoder, one global
  smoothing constant applied to every level;
- CatBoost — ordered target statistics, shrinkage adapted per row under a
  random permutation;
- LightGBM here — no target statistics at all. Its categorical splits sort
  levels by accumulated gradient statistics within the node and then cut that
  ordering, which finds arbitrary level subsets without ever estimating a
  per-level target mean.

That third mechanism is the reason to expect decorrelation. It is also the
reason to expect it might be *weaker*: sorting by gradient is a heuristic,
where the other two estimate the quantity directly.

Phase 8's finding decides the numeric block: imputed columns are appended
alongside the originals rather than replacing them, because LightGBM routes
NaN natively and an imputed point estimate is strictly less expressive than a
learned default direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .data import NUMERIC_COLS
from .encoding import ENCODE_COLS, level_frame
from .features import (
    ENGINEERED_COLS,
    GENERATOR_COLS,
    EngineeredFeatures,
    GeneratorFeatures,
)
from .validation import SEED

LGBM_DEFAULTS = dict(
    n_estimators=2000,
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=200,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=3.0,
    objective="binary",
    n_jobs=4,
    verbose=-1,
    random_state=SEED,
)


class LGBMLookup(BaseEstimator, ClassifierMixin):
    """Numeric block plus every column repeated as a pandas `category`.

    Handing the levels over as `category` dtype is what switches LightGBM into
    its categorical split finder; passing the same values as integers would
    get ordered splits and defeat the point.

    As with the CatBoost member, each of the twelve columns appears twice —
    once numeric, supporting ordering questions, once categorical, supporting
    identity questions. Phase 9 showed the second is where the signal lives.
    """

    def __init__(self, params: dict | None = None, use_lattice: bool = True):
        self.params = params
        self.use_lattice = use_lattice

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).reset_index(drop=True)
        feats = EngineeredFeatures().fit(X).transform(X)
        numeric = [*NUMERIC_COLS, *ENGINEERED_COLS]
        if self.use_lattice:
            feats = GeneratorFeatures().fit(feats).transform(feats)
            numeric = [*numeric, *GENERATOR_COLS]
        num = feats[numeric].replace([np.inf, -np.inf], np.nan)
        lvl = level_frame(X, ENCODE_COLS).add_prefix("lvl_")
        for c in lvl.columns:
            # Categories are fixed from the union seen at fit time so that a
            # level absent from one fold does not shift the encoding of others.
            lvl[c] = pd.Categorical(lvl[c], categories=self.categories_[c])
        return pd.concat([num.reset_index(drop=True), lvl.reset_index(drop=True)], axis=1)

    def fit(self, X: pd.DataFrame, y):
        from lightgbm import LGBMClassifier

        raw = level_frame(pd.DataFrame(X), ENCODE_COLS).add_prefix("lvl_")
        self.categories_ = {c: pd.Index(sorted(raw[c].unique())) for c in raw.columns}
        frame = self._frame(X)
        self.classes_ = np.unique(y)
        self.n_features_out_ = frame.shape[1]
        self.model_ = LGBMClassifier(**{**LGBM_DEFAULTS, **(self.params or {})})
        self.model_.fit(frame, np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._frame(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def lgbm_lookup() -> LGBMLookup:
    return LGBMLookup()

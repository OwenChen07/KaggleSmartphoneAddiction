"""Target and frequency encoding, treating every column as a lookup key.

The premise, established in reports/external-research.md and verified on this
data: several columns are not *quantities* at all. The generator uses their
exact value as a key into a lookup table. `notifications_per_day` has a
per-value target-rate standard deviation 22x larger than sampling noise can
produce, and neighbouring integer values differ in addiction rate by 0.22 on
average. As raw numbers those columns are worth an OOF AUC of 0.5122; read as
lookup keys they are worth 0.8170.

A tree cannot recover that. It splits on `x <= t`, which asks an ordering
question about a variable whose order carries no meaning. Target encoding
replaces the value with the smoothed mean of the target at that exact level,
which reads the lookup directly.

Two encodings per column:

- **target**: smoothed mean of `y` among rows sharing the level,
  `(sum_y + prior * smooth) / (count + smooth)`. `smooth=10` is deliberate —
  larger values were measured as worse.
- **frequency**: how many rows share the level. Cheap, and it separates a
  common level whose mean is well estimated from a rare one whose mean is not.

Leakage is the entire difficulty, so the nesting is the important part of this
file. See `TargetFrequencyEncoder`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from .data import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS
from .validation import SEED

#: Columns encoded as lookup keys. All of them: the research finding is that
#: continuous columns benefit as much as the nominal ones, because at 691,369
#: rows even `daily_screen_time_hours` has ~500 rows behind each of its 1,389
#: distinct values, which is plenty to estimate a smoothed mean.
ENCODE_COLS = list(FEATURE_COLS)

MISSING_LEVEL = "__missing__"


def level_frame(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Cast columns to explicit string levels, with missing as its own level.

    **Do not simplify this to `X[c].astype(str)`.** That is correct on pandas
    2.x, where it writes the literal string "nan", and silently wrong on pandas
    3.0+, where the new string dtype preserves NA instead. Under 3.0 a
    subsequent `groupby` drops every missing row from the level statistics and
    `.map()` returns NA for them, so the encoding quietly vanishes on 4-20% of
    every column. Nothing raises and nothing warns.

    This project runs pandas 3.0.5, so the naive form would be actively broken
    here. Routing through `object` and filling *before* the string cast is
    correct on both versions. `assert_full_coverage` checks it rather than
    trusting it.
    """
    columns = ENCODE_COLS if columns is None else columns
    return pd.DataFrame(
        {c: X[c].astype(object).fillna(MISSING_LEVEL).astype(str).to_numpy() for c in columns},
        index=X.index,
    )


def assert_full_coverage(levels: pd.DataFrame) -> None:
    """Every row must belong to some level in every column.

    This is the guard against the pandas trap above: if missing values lost
    their level, a `groupby(...).size().sum()` would come up short of the row
    count and the encoding would be silently incomplete.
    """
    n = len(levels)
    for c in levels.columns:
        covered = int(levels.groupby(c, dropna=False)[c].size().sum())
        if covered != n:
            raise AssertionError(
                f"level coverage for {c!r} is {covered:,} of {n:,} rows — missing values "
                f"lost their level (see level_frame's note on pandas 3.0)"
            )


def _stats(level: pd.Series, y: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"lv": level.to_numpy(), "y": y})
    g = frame.groupby("lv", observed=True)["y"]
    return pd.DataFrame({"sum": g.sum(), "count": g.size()})


class TargetFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Append smoothed target-mean and frequency columns for each input column.

    **The nesting.** `transform` uses statistics fit on everything the encoder
    was shown, which is correct for validation and test rows — they were not
    part of that fit. It is *not* correct for the training rows themselves: a
    row would be encoded partly with its own target, which inflates CV and
    collapses on the leaderboard.

    So `fit_transform` is overridden rather than inherited. It computes an
    inner K-fold split of the training rows and encodes each inner fold from
    the other inner folds only, so **no row's encoding ever sees its own
    target**. Because scikit-learn's Pipeline calls `fit_transform` on
    intermediate steps during `fit` and `transform` during prediction, wiring
    it this way makes the correct thing happen automatically at both times.

    That gives two levels of protection working together: `run_cv` refits the
    whole pipeline per outer fold, so the encoder never sees the outer
    validation rows at all, and the inner split protects the training rows from
    themselves.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        smooth: float = 10.0,
        inner_splits: int = 5,
        seed: int = SEED,
        add_frequency: bool = True,
    ):
        self.columns = columns
        self.smooth = smooth
        self.inner_splits = inner_splits
        self.seed = seed
        self.add_frequency = add_frequency

    def _cols(self) -> list[str]:
        return ENCODE_COLS if self.columns is None else list(self.columns)

    def _encode_from(self, stats: dict[str, pd.DataFrame], levels: pd.DataFrame,
                     prior: float, n_ref: int) -> pd.DataFrame:
        out = {}
        for c in self._cols():
            st = stats[c]
            sm = (st["sum"] + prior * self.smooth) / (st["count"] + self.smooth)
            lv = levels[c]
            out[f"te_{c}"] = lv.map(sm).astype(np.float32).fillna(np.float32(prior))
            if self.add_frequency:
                out[f"fq_{c}"] = (
                    lv.map(st["count"]).astype(np.float32).fillna(np.float32(0.0)) / n_ref
                )
        return pd.DataFrame(out, index=levels.index)

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("TargetFrequencyEncoder needs y")
        y = np.asarray(y)
        levels = level_frame(X, self._cols())
        assert_full_coverage(levels)
        self.prior_ = float(y.mean())
        self.n_fit_ = len(X)
        self.stats_ = {c: _stats(levels[c], y) for c in self._cols()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        levels = level_frame(X, self._cols())
        enc = self._encode_from(self.stats_, levels, self.prior_, self.n_fit_)
        return pd.concat([pd.DataFrame(X).reset_index(drop=True),
                          enc.reset_index(drop=True)], axis=1)

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        self.fit(X, y)
        y = np.asarray(y)
        X = pd.DataFrame(X).reset_index(drop=True)
        levels = level_frame(X, self._cols()).reset_index(drop=True)

        inner = StratifiedKFold(
            n_splits=self.inner_splits, shuffle=True, random_state=self.seed
        )
        pieces = []
        for tr, va in inner.split(X, y):
            stats = {c: _stats(levels[c].iloc[tr], y[tr]) for c in self._cols()}
            prior = float(y[tr].mean())
            piece = self._encode_from(stats, levels.iloc[va], prior, len(tr))
            pieces.append(piece)
        enc = pd.concat(pieces).sort_index()
        return pd.concat([X, enc], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        extra = []
        for c in self._cols():
            extra.append(f"te_{c}")
            if self.add_frequency:
                extra.append(f"fq_{c}")
        return np.asarray([*base, *extra], dtype=object)


def encoded_numeric_cols(
    base_numeric: list[str], columns: list[str] | None = None, add_frequency: bool = True
) -> list[str]:
    """Numeric column list for a `tree_preprocessor` downstream of the encoder."""
    cols = ENCODE_COLS if columns is None else list(columns)
    extra = []
    for c in cols:
        extra.append(f"te_{c}")
        if add_frequency:
            extra.append(f"fq_{c}")
    return [*base_numeric, *extra]

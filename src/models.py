"""Model zoo for the bake-off.

Each entry builds a complete unfitted Pipeline (preprocessing + estimator) so
that `run_cv` can clone and fit it per fold. Adding a new candidate means
adding one function here and one name to `MODELS` — the harness, the log and
the submission writer need no changes. That is the point of the structure.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL_COLS, NUMERIC_COLS
from .encoding import ENCODE_COLS, TargetFrequencyEncoder, encoded_numeric_cols
from .features import (
    ENGINEERED_COLS,
    RECOVERABLE_COLS,
    GENERATOR_COLS,
    TRIG_COLS,
    EngineeredFeatures,
    GeneratorFeatures,
    LookupTrig,
    RecoverableImputer,
    engineered_numeric_cols,
    imputed_numeric_cols,
)
from .preprocessing import linear_preprocessor, missingness_only_preprocessor, tree_preprocessor
from .validation import SEED


class SeedAveraged(BaseEstimator, ClassifierMixin):
    """Average several copies of one estimator that differ only by random seed.

    HistGradientBoosting is *not* fully deterministic given its data: the bin
    edges come from a 200,000-row subsample and, when early stopping is on, the
    internal validation split is drawn at random. Both are driven by
    `random_state`, so re-seeding produces a genuinely different model — just a
    mildly different one. Measured pairwise Spearman between seeds here is
    0.993, so the errors are highly but not perfectly correlated, and averaging
    cancels the part that is not shared.

    This is variance reduction, not bias reduction. It cannot find structure a
    single model missed; it only stops one model's arbitrary choices from
    reaching the prediction. That is why the gain is bounded and why it shows
    up reliably rather than sometimes.

    Deliberately a wrapper rather than a change to `run_cv`: the fold
    assignment is untouched, so the OOF vector stays row-aligned with every
    other run in the log and remains comparable by paired bootstrap.

    `averaging="rank"` converts each member to ranks before averaging, matching
    `src/blend.py`; `"proba"` averages the probabilities directly. For members
    that share an architecture and therefore an output scale the two are nearly
    equivalent — the rank argument bites when combining different model
    families. Both are offered so the choice is measured, not assumed.
    """

    def __init__(self, base=None, n_seeds: int = 5, seed0: int = SEED, averaging: str = "rank"):
        self.base = base
        self.n_seeds = n_seeds
        self.seed0 = seed0
        self.averaging = averaging

    def fit(self, X, y):
        if self.averaging not in {"rank", "proba"}:
            raise ValueError("averaging must be 'rank' or 'proba'")
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for i in range(self.n_seeds):
            est = clone(self.base)
            est.set_params(clf__random_state=self.seed0 + i)
            est.fit(X, y)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        cols = [e.predict_proba(X)[:, 1] for e in self.estimators_]
        if self.averaging == "rank":
            n = len(X)
            avg = np.mean([rankdata(c, method="average") / n for c in cols], axis=0)
        else:
            avg = np.mean(cols, axis=0)
        return np.column_stack([1.0 - avg, avg])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def named_steps(self):
        """Expose the wrapped pipeline's steps so harness introspection —
        `validation._count_model_features` — sees the real preprocessed width
        rather than falling back to the raw input column count."""
        return self.estimators_[0].named_steps


def _tuned_pipeline() -> Pipeline:
    """The Phase 6 winning configuration, read from the search artifact."""
    params = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments" / "best_params.json").read_text()
    )["params"]
    pipe = histgbm_features()
    pipe.set_params(**params)
    return pipe


def histgbm_tuned_seedavg() -> SeedAveraged:
    """Phase 7: the tuned model, averaged over 5 seeds."""
    return SeedAveraged(base=_tuned_pipeline(), n_seeds=5, averaging="rank")


def dummy() -> Pipeline:
    """Sanity check. Must score exactly 0.500 — if it does not, the harness is
    wrong and every other number in the log is suspect."""
    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            ("clf", DummyClassifier(strategy="prior")),
        ]
    )


def baseline_histgbm_numeric() -> Pipeline:
    """Phase 0 floor: gradient boosting on the numeric columns only, defaults
    untouched, categoricals discarded."""
    return Pipeline(
        [
            ("prep", tree_preprocessor(categorical_cols=[])),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )


def logistic() -> Pipeline:
    return Pipeline(
        [
            ("prep", linear_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def random_forest() -> Pipeline:
    """Depth and leaf size are capped deliberately: at 691k rows an unbounded
    forest costs far more time than the accuracy it buys. The constraint is
    recorded here rather than left implicit in a wall-clock number."""
    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=50,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def histgbm() -> Pipeline:
    """Same estimator as the Phase 0 baseline, now with the categoricals."""
    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )


def histgbm_isna() -> Pipeline:
    """HistGBM plus explicit is-missing indicators. HistGBM already routes NaN
    natively, so this tests whether making missingness an *explicit* feature —
    and therefore usable in combination across columns — adds anything on top."""
    return Pipeline(
        [
            ("prep", tree_preprocessor(missing_indicators=True)),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )


def missingness_only() -> Pipeline:
    """No feature values at all — only the pattern of which are missing.
    An AUC meaningfully above 0.5 means the data is not missing at random."""
    return Pipeline(
        [
            ("prep", missingness_only_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )


def histgbm_features(columns: list[str] | None = None) -> Pipeline:
    """Phase 4: HistGBM on the raw columns plus engineered ratios.

    `EngineeredFeatures` sits *inside* the Pipeline, ahead of the
    ColumnTransformer, so it is cloned and fit per fold like every other step.
    It is stateless, so that placement changes nothing numerically — but it
    keeps the rule ("all preprocessing inside the pipeline") true by
    construction rather than by argument, which is what matters when a later
    feature does need to learn something.
    """
    return Pipeline(
        [
            ("feats", EngineeredFeatures(columns=columns)),
            ("prep", tree_preprocessor(numeric_cols=engineered_numeric_cols(columns))),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )


#: Engineered features whose permutation importance on held-out rows was
#: clearly above its own repeat-to-repeat spread. Measured on fold 1, full
#: data, 5 repeats — see reports/feature-engineering.md for the full table.
SURVIVING_FEATURES = [
    "social_share",
    "work_share",
    "gaming_share",
    "residual_screen",
    "weekend_ratio",
]


def histgbm_features_pruned() -> Pipeline:
    """Phase 4, pruned to the five engineered features that survived
    permutation importance on validation rows."""
    return histgbm_features(columns=SURVIVING_FEATURES)


def histgbm_features_no_control() -> Pipeline:
    """All engineered features except `sleep_deficit`, the monotonic control.

    Ablation for the claim that `sleep_deficit` is information-free: it carries
    a non-zero permutation importance, so this measures whether that
    corresponds to any actual predictive contribution.
    """
    return histgbm_features(columns=[c for c in ENGINEERED_COLS if c != "sleep_deficit"])


def histgbm_tuned_imputed(mode: str = "replace") -> Pipeline:
    """Phase 8: reconstruct the recoverable columns, then the tuned model.

    Step order matters. The imputer runs **first**, so `EngineeredFeatures`
    downstream builds its ratios from reconstructed values rather than from
    holes. Under `mode="replace"` that is most of the point: the ratios were
    previously missing for 17-36% of rows because a ratio inherits the
    missingness of both its inputs, and filling the inputs makes them
    computable.
    """
    params = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments" / "best_params.json").read_text()
    )["params"]
    pipe = Pipeline(
        [
            ("impute", RecoverableImputer(mode=mode)),
            ("feats", EngineeredFeatures()),
            ("prep", tree_preprocessor(numeric_cols=imputed_numeric_cols(mode))),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )
    pipe.set_params(**params)
    return pipe


def histgbm_tuned_imputed_augment() -> Pipeline:
    return histgbm_tuned_imputed(mode="augment")


def histgbm_imputed_seedavg() -> SeedAveraged:
    """Phase 8 final: augment-imputed features, 5-seed averaged.

    Note a known inefficiency: `SeedAveraged` clones the whole pipeline, so the
    five imputation regressors are refit once per seed with an identical
    `random_state` and therefore identical results. Five times the imputation
    cost for no benefit. Left as-is because hoisting the imputer out of the
    wrapper would move a *stateful* transformer outside the per-fold clone,
    which is precisely the leak this project spends its effort avoiding. Wasted
    CPU is the cheaper mistake.
    """
    return SeedAveraged(base=histgbm_tuned_imputed("augment"), n_seeds=5, averaging="rank")


def histgbm_encoded(
    impute: bool = False,
    lattice: bool = True,
    encode: bool = True,
    trig: bool = False,
    transductive: bool = False,
    inner_splits: int = 5,
    params_path: str = "best_params.json",
) -> Pipeline:
    """Phase 9: the lookup-key representation.

    Step order is deliberate. The engineered ratios and generator features are
    appended first; the target encoder then runs on the **raw** twelve columns
    only, so the encoded block describes the original lookup keys rather than
    derived quantities. `TargetFrequencyEncoder` overrides `fit_transform`, so
    Pipeline gives the training rows inner out-of-fold encodings during `fit`
    and the full-fit statistics during `predict` — automatically, and in the
    right direction each time.
    """
    params = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments" / params_path).read_text()
    )["params"]
    steps = []
    numeric = list(NUMERIC_COLS)
    if impute:
        steps.append(("impute", RecoverableImputer(mode="augment")))
        numeric = [*numeric, *[f"{c}__imp" for c in RECOVERABLE_COLS]]
    steps.append(("feats", EngineeredFeatures()))
    numeric = [*numeric, *ENGINEERED_COLS]
    if lattice:
        steps.append(("gen", GeneratorFeatures()))
        numeric = [*numeric, *GENERATOR_COLS]
    if trig:
        steps.append(("trig", LookupTrig()))
        numeric = [*numeric, *TRIG_COLS]
    if encode:
        freq_frame = None
        if transductive:
            # Feature rows only, from train and test. No labels are involved,
            # so this leaks nothing; see TargetFrequencyEncoder._fit_freq.
            from .data import FEATURE_COLS, load_test, load_train

            Xtr, _ = load_train()
            Xte, _ = load_test()
            freq_frame = pd.concat([Xtr[FEATURE_COLS], Xte[FEATURE_COLS]], ignore_index=True)
        steps.append((
            "enc",
            TargetFrequencyEncoder(inner_splits=inner_splits, freq_frame=freq_frame),
        ))
        numeric = encoded_numeric_cols(numeric)
    steps.append(("prep", tree_preprocessor(numeric_cols=numeric)))
    steps.append(("clf", HistGradientBoostingClassifier(random_state=SEED)))
    pipe = Pipeline(steps)
    pipe.set_params(**params)
    return pipe


def histgbm_encoded_full() -> Pipeline:
    return histgbm_encoded(impute=True, lattice=True, encode=True)


def histgbm_encoded_v2() -> Pipeline:
    """Phase 12: the four levers from reports/remaining-levers.md at once —
    transductive frequency counts, ten encoding folds, trig on the lookup
    columns, and the re-tuned parameters from Phase 10."""
    return histgbm_encoded(
        impute=True, lattice=True, encode=True, trig=True,
        transductive=True, inner_splits=10,
        params_path="best_params_encoded.json",
    )


def _catboost_lookup():
    """Imported lazily so the zoo still loads if catboost is not installed."""
    from .catboost_model import catboost_lookup

    return catboost_lookup()


def histgbm_native_cat() -> Pipeline:
    """Side experiment A: declare the categoricals to HistGBM natively.

    The ordinal encoding used everywhere else imposes an order the categories
    do not have — `gender` becomes Female=0 < Male=1 < Other=2, and a tree can
    then only cut that axis into contiguous ranges, so {Female, Other} versus
    {Male} costs two splits instead of one. `categorical_features` lets the
    split finder partition the levels as an unordered set instead.

    The encoder still runs: HistGBM wants small non-negative integer codes, not
    strings. What changes is that the estimator is told which columns those
    codes describe, so it stops treating them as ordered magnitudes.
    """
    n_numeric = len(NUMERIC_COLS)
    categorical_mask = [False] * n_numeric + [True] * len(CATEGORICAL_COLS)
    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    categorical_features=categorical_mask, random_state=SEED
                ),
            ),
        ]
    )


MODELS = {
    "dummy": (dummy, "Dummy (prior) — harness sanity check"),
    "baseline": (baseline_histgbm_numeric, "Phase 0: HistGBM, numeric cols only"),
    "logistic": (logistic, "Logistic regression, impute+scale+onehot"),
    "rf": (random_forest, "Random forest (200 trees, leaf>=50)"),
    "histgbm": (histgbm, "HistGBM, numeric + categorical"),
    "histgbm_isna": (histgbm_isna, "HistGBM + is-missing indicators"),
    "missingness_only": (missingness_only, "Is-missing pattern only (MNAR test)"),
    "histgbm_fe": (histgbm_features, "Phase 4: HistGBM + engineered ratios (all 15)"),
    "histgbm_fe_pruned": (histgbm_features_pruned, "Phase 4: HistGBM + 5 surviving ratios"),
    "histgbm_fe_nocontrol": (
        histgbm_features_no_control,
        "Phase 4: engineered ratios minus the sleep_deficit control",
    ),
    "histgbm_native_cat": (histgbm_native_cat, "HistGBM, native categorical splits"),
    "histgbm_seedavg": (histgbm_tuned_seedavg, "Phase 7: tuned HistGBM, 5-seed average"),
    "histgbm_imputed": (histgbm_tuned_imputed, "Phase 8: tuned + imputed cols (replace)"),
    "catboost_lookup": (
        _catboost_lookup,
        "Phase 11: CatBoost, ordered target statistics on raw levels",
    ),
    "histgbm_encoded": (histgbm_encoded, "Phase 9: lookup-key target encoding"),
    "histgbm_encoded_v2": (
        histgbm_encoded_v2,
        "Phase 12: transductive counts + 10 enc folds + trig",
    ),
    "histgbm_encoded_full": (
        histgbm_encoded_full,
        "Phase 9: target encoding + lattice + imputation",
    ),
    "histgbm_imputed_seedavg": (
        histgbm_imputed_seedavg,
        "Phase 8: imputed (augment) + 5-seed average",
    ),
    "histgbm_imputed_aug": (
        histgbm_tuned_imputed_augment,
        "Phase 8: tuned + imputed cols (augment)",
    ),
}

"""Model zoo for the bake-off.

Each entry builds a complete unfitted Pipeline (preprocessing + estimator) so
that `run_cv` can clone and fit it per fold. Adding a new candidate means
adding one function here and one name to `MODELS` — the harness, the log and
the submission writer need no changes. That is the point of the structure.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL_COLS, NUMERIC_COLS
from .features import EngineeredFeatures, engineered_numeric_cols
from .preprocessing import linear_preprocessor, missingness_only_preprocessor, tree_preprocessor
from .validation import SEED


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
    "histgbm_native_cat": (histgbm_native_cat, "HistGBM, native categorical splits"),
}

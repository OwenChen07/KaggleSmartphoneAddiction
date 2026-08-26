"""Permutation importance, measured on held-out rows.

    python -m src.importance --model histgbm_fe
    python -m src.importance --model histgbm_fe --on train    # the wrong way

Permutation importance asks: shuffle one column so it keeps its marginal
distribution but loses its relationship to the target, refit nothing, and see
how far the score falls. The drop is that column's contribution *to this fitted
model*.

**Which rows you measure it on decides what the number means.** Measured on the
rows the model was fit on, the score reflects what the model *used*, including
whatever it memorised: a high-cardinality or noise column the model overfit
will show a large drop, because breaking the column breaks the memorisation.
Measured on held-out rows, it reflects what the column contributes to
*generalisation* — a column the model overfit contributes nothing there, and
scores about zero. Only the second answers "should I keep this feature".

`--on train` exists to demonstrate the gap, not because it is ever the right
default. See reports/feature-engineering.md.

The engineered features are produced by a step inside the pipeline, so
permuting the pipeline's *inputs* would measure the raw columns and silently
recompute every engineered column from the shuffled input. To rank the
engineered columns themselves, the stateless `feats` step is peeled off the
fitted pipeline and its output frame is what gets permuted.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .data import load_train  # noqa: E402
from .models import MODELS  # noqa: E402
from .validation import N_SPLITS, SEED  # noqa: E402


def add_noise_column(X: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Append `noise_uniform`, a column of independent U(0,1) draws.

    It is pure noise by construction — sampled without reference to `y` — so
    its true contribution to generalisation is exactly zero, and we know that
    in advance rather than inferring it. Its purpose is to calibrate an
    importance measure: any method that assigns it a large importance is
    reporting something other than predictive value.

    Continuous and effectively unique per row, so a deep enough tree can carve
    single rows out on it. That is what makes it a sharp test of the
    train-rows-versus-held-out-rows distinction.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    out["noise_uniform"] = rng.random(len(X)).astype(np.float32)
    return out


def _split_feature_step(fitted: Pipeline) -> tuple[object | None, Pipeline]:
    """Separate a fitted pipeline into its `feats` step and the rest.

    Returns `(None, fitted)` unchanged when the pipeline has no `feats` step.
    The tail Pipeline reuses the already-fitted step objects, so it must not be
    refit — it is only ever called for prediction.
    """
    if not fitted.steps or fitted.steps[0][0] != "feats":
        return None, fitted
    return fitted.steps[0][1], Pipeline(fitted.steps[1:])


def fold_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    fold: int = 1,
    on: str = "val",
    n_repeats: int = 5,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit on one fold's training rows, then permute columns and re-score.

    Uses the same `StratifiedKFold(seed)` as `run_cv`, so `fold` here is the
    same fold that run produced, and the model measured is the model that
    produced that fold's OOF predictions.
    """
    if on not in {"val", "train"}:
        raise ValueError("`on` must be 'val' or 'train'")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(X, y))[fold - 1]

    est = clone(estimator)
    est.fit(X.iloc[train_idx], y[train_idx])

    score_idx = train_idx if on == "train" else val_idx
    feats, tail = _split_feature_step(est)
    X_score = X.iloc[score_idx]
    if feats is not None:
        X_score = feats.transform(X_score)

    result = permutation_importance(
        tail,
        X_score,
        y[score_idx],
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X_score.columns),
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="histgbm_fe", choices=list(MODELS))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--on",
        default="val",
        choices=["val", "train"],
        help="rows to permute on; 'val' is the only one that answers 'keep this feature?'",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--noise-col",
        action="store_true",
        help="append a pure-noise U(0,1) column, whose true importance is zero by construction",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=None,
        help="override the estimator's max_leaf_nodes, to let it overfit on purpose",
    )
    args = parser.parse_args()

    X, y = load_train(sample=args.sample)
    build, description = MODELS[args.model]

    estimator = build()
    if args.noise_col:
        # The noise column is not part of any declared feature list, so it is
        # fed through the pipeline's numeric block explicitly.
        from .data import CATEGORICAL_COLS
        from .features import engineered_numeric_cols
        from .preprocessing import tree_preprocessor

        X = add_noise_column(X)
        has_feats = estimator.steps[0][0] == "feats"
        numeric = engineered_numeric_cols(None) if has_feats else list(X.columns[:9])
        estimator.set_params(
            prep=tree_preprocessor(
                numeric_cols=[*numeric, "noise_uniform"], categorical_cols=CATEGORICAL_COLS
            )
        )
    if args.max_leaf_nodes is not None:
        estimator.set_params(clf__max_leaf_nodes=args.max_leaf_nodes)

    scope = "TRAINING rows (diagnostic only)" if args.on == "train" else "held-out validation rows"
    print(f"{description}")
    print(f"fold {args.fold}, permuting on {scope}, {args.repeats} repeats, n={len(X)}\n", flush=True)

    frame = fold_permutation_importance(
        estimator,
        X,
        y,
        fold=args.fold,
        on=args.on,
        n_repeats=args.repeats,
        n_jobs=args.n_jobs,
    )
    print(f"{'feature':<26}{'drop in AUC':>14}{'std':>11}")
    print("-" * 51)
    for row in frame.itertuples(index=False):
        print(f"{row.feature:<26}{row.importance:>14.6f}{row.std:>11.6f}")


if __name__ == "__main__":
    main()

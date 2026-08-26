"""Adversarial validation — is the test set drawn from the same distribution
as the training set?

    python -m src.adversarial
    python -m src.adversarial --with-features

The technique: throw away the real target, label every training row 0 and
every test row 1, and try to predict *which file a row came from*. That is an
ordinary binary classification problem, so it goes through ordinary stratified
CV and produces an ordinary AUC.

Reading the result:

- **AUC ~ 0.5** — the classifier cannot tell the two files apart. Train and
  test are exchangeable, so a model validated on held-out training rows is
  being validated on rows that look like the rows it will be scored on. The
  OOF estimate is trustworthy in the sense this test can speak to.
- **AUC >> 0.5** — something distinguishes them: a shifted feature, a time
  split, a different sampling frame. Then held-out training rows are *not* a
  fair stand-in for test rows, OOF will mislead by an amount nobody can
  estimate from the training set alone, and the per-feature importances say
  where the shift lives.

What it does **not** test: whether the *target* relationship is stable. Two
files can have identical feature distributions while `P(y | X)` differs
between them. Adversarial validation is silent on that, because it never looks
at `y`.

The classifier here has to be at least as expressive as the model whose OOF is
being defended — a weak discriminator finding nothing proves nothing — so it
is the same HistGradientBoostingClassifier used everywhere else.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .data import load_test, load_train  # noqa: E402
from .features import EngineeredFeatures, engineered_numeric_cols  # noqa: E402
from .importance import fold_permutation_importance  # noqa: E402
from .preprocessing import (  # noqa: E402
    missingness_only_preprocessor,
    tree_preprocessor,
)
from .validation import SEED, run_cv  # noqa: E402


def build_adversarial_frame(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stack train and test into one frame labelled by origin.

    The real target is never passed in, so it cannot leak into the
    discriminator. `id` is already excluded upstream by `FEATURE_COLS` — it is
    a row counter that separates the two files perfectly and would produce a
    meaningless AUC of 1.0.
    """
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    origin = np.concatenate(
        [np.zeros(len(X_train), dtype=np.int8), np.ones(len(X_test), dtype=np.int8)]
    )
    return X, origin


def adversarial_pipeline(with_features: bool = False, mode: str = "full") -> Pipeline:
    """`mode` decides what the discriminator is allowed to look at.

    - `full`       — values and missingness, as the models actually see them.
    - `missingness`— only the is-missing pattern; values discarded entirely.
    - `values`     — median-impute every missing cell. **This does not erase
                     missingness**: every imputed cell lands on exactly the
                     median, so the discriminator can still count the spike
                     there. Kept because that failure is worth seeing.
    - `complete`   — keep only rows with no missing value anywhere, which
                     removes the missingness channel by construction at the
                     cost of ~74% of the rows.

    Running these splits the total separation into the part carried by the
    numbers and the part carried by which numbers are absent.
    """
    if mode == "missingness":
        return Pipeline(
            [
                ("prep", missingness_only_preprocessor()),
                ("clf", HistGradientBoostingClassifier(random_state=SEED)),
            ]
        )
    if mode == "complete":
        return Pipeline(
            [
                ("prep", tree_preprocessor()),
                ("clf", HistGradientBoostingClassifier(random_state=SEED)),
            ]
        )
    if mode == "values":
        prep = tree_preprocessor()
        return Pipeline(
            [
                ("prep", prep),
                ("impute", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(random_state=SEED)),
            ]
        )
    if with_features:
        return Pipeline(
            [
                ("feats", EngineeredFeatures()),
                ("prep", tree_preprocessor(numeric_cols=engineered_numeric_cols(None))),
                ("clf", HistGradientBoostingClassifier(random_state=SEED)),
            ]
        )
    return Pipeline(
        [
            ("prep", tree_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--with-features",
        action="store_true",
        help="also give the discriminator the Phase 4 engineered features",
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "missingness", "values", "complete"],
        help="what the discriminator may look at; see adversarial_pipeline",
    )
    parser.add_argument("--no-importance", action="store_true")
    args = parser.parse_args()

    X_train, _ = load_train()
    X_test, _ = load_test()
    X, origin = build_adversarial_frame(X_train, X_test)

    if args.mode == "complete":
        keep = X.notna().all(axis=1).to_numpy()
        X, origin = X.loc[keep].reset_index(drop=True), origin[keep]
        print(f"complete-case subset: {keep.sum():,} of {len(keep):,} rows ({keep.mean():.1%})")

    print(f"train {len(X_train):,} rows (label 0)  test {len(X_test):,} rows (label 1)")
    print(f"combined {len(X):,} rows, test share {origin.mean():.4f}\n", flush=True)

    result = run_cv(
        adversarial_pipeline(args.with_features, args.mode),
        X,
        origin,
        description="adversarial: train vs test",
        model="adversarial",
        n_splits=args.folds,
    )

    auc = result.oof_auc
    print()
    print(f"  adversarial AUC   {auc:.6f}")
    print(f"  fold spread       {result.fold_std:.6f}")
    print(f"  distance from 0.5 {auc - 0.5:+.6f}")
    print()
    if abs(auc - 0.5) < 0.01:
        print("  -> indistinguishable. Train and test look like one population;")
        print("     held-out training rows are a fair stand-in for test rows.")
    else:
        print("  -> separable. Train and test differ; OOF is not a safe proxy.")
        print("     The importances below say which columns carry the shift.")

    if args.no_importance:
        return
    print("\nper-feature permutation importance for the discriminator")
    print("(fold 1, validation rows — anything near zero carries no shift)\n")
    frame = fold_permutation_importance(
        adversarial_pipeline(args.with_features, args.mode),
        X,
        origin,
        fold=1,
        n_repeats=args.repeats,
        n_splits=args.folds,
        n_jobs=args.n_jobs,
    )
    print(f"{'feature':<26}{'drop in AUC':>14}{'std':>11}")
    print("-" * 51)
    for row in frame.itertuples(index=False):
        print(f"{row.feature:<26}{row.importance:>14.6f}{row.std:>11.6f}")


if __name__ == "__main__":
    main()

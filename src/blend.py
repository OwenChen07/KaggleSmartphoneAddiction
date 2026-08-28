"""Phase 6 — rank-average blending over logged runs.

    python -m src.blend 012 008 004                 # evaluate on OOF only
    python -m src.blend 012 008 --submit            # also write a submission

**Rank-average, not probability-average, because the metric is AUC.** ROC AUC
depends only on the *ordering* of the predictions, so any strictly increasing
transform of a model's output leaves its AUC untouched. That has a direct
consequence for blending: averaging raw probabilities lets the *shape* of each
model's output distribution decide how much say it gets. A model whose
probabilities bunch near 0 and 1 moves a probability average far more than one
whose predictions sit in a narrow band, even if both rank the rows equally
well. Converting each model to ranks first strips that arbitrary scaling out,
so every model contributes exactly the information AUC actually reads.

Ranks are normalised to (0, 1] by dividing by n, which changes no ordering but
keeps the blended vector in a familiar range and makes weights comparable
across vectors of different length.

Because every run in this project shares the same fold assignment, the OOF
vectors are row-aligned, so a blend can be evaluated **without refitting
anything** — the vectors on disk are enough. Only writing a submission needs
the saved test predictions.
"""

from __future__ import annotations

import argparse
import itertools
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .data import load_train  # noqa: E402
from .validation import (  # noqa: E402
    LOG_PATH,
    SEED,
    SUBMISSION_DIR,
    bootstrap_auc_diff,
    load_oof,
)


def rank_normalise(v: np.ndarray) -> np.ndarray:
    """Average-rank transform scaled to (0, 1]. Ties share their mean rank."""
    return rankdata(v, method="average") / len(v)


def rank_average(vectors: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(vectors)
    if len(weights) != len(vectors):
        raise ValueError("need one weight per vector")
    total = float(sum(weights))
    stacked = np.zeros(len(vectors[0]), dtype=np.float64)
    for v, w in zip(vectors, weights):
        stacked += (w / total) * rank_normalise(v)
    return stacked


def to_logit(p: np.ndarray, clip: float = 30.0) -> np.ndarray:
    """Map probabilities to the log-odds scale, clipped.

    This target saturates hard — the top screen-time decile is essentially all
    positive — and in that region probabilities have no resolution left while
    logits still do. Two models that both say 0.999 may disagree by a lot in
    log-odds, and a combiner working on probabilities cannot see it.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def logit_stack_score(
    y: np.ndarray, oofs: list[np.ndarray], n_splits: int = 5, seed: int = SEED
) -> dict[str, float]:
    """Honest score for a logistic-regression stacker over OOF logits.

    **A stacker fit on the OOF matrix and scored on the same rows reads high**,
    because the coefficients were chosen using those labels. So the combiner is
    fit on half the OOF rows and scored on the held-out half, repeated over
    `n_splits` shuffles, and the mean reported. The equal-weight rank average
    is scored on the *same* held-out halves so the comparison is paired.

    The reason to prefer this over `rank_average` is one degree of freedom:
    a rank average weights every member positively by construction, while a
    stacker can give a member a **negative** coefficient. A weak,
    decorrelated model is often useful as a correction rather than as
    something to average in, and only the stacker can express that.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit

    Z = np.column_stack([to_logit(o) for o in oofs])
    R = np.column_stack([rank_normalise(o) for o in oofs])
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=0.5, random_state=seed
    )
    stack, plain, coefs = [], [], []
    for fit_idx, score_idx in splitter.split(Z, y):
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Z[fit_idx], y[fit_idx])
        stack.append(roc_auc_score(y[score_idx], lr.decision_function(Z[score_idx])))
        plain.append(roc_auc_score(y[score_idx], R[score_idx].mean(axis=1)))
        coefs.append(lr.coef_[0])
    stack, plain = np.array(stack), np.array(plain)
    return {
        "logit_stack": float(stack.mean()),
        "rank_average": float(plain.mean()),
        "diff": float((stack - plain).mean()),
        "diff_sd": float((stack - plain).std()),
        "coefs": np.mean(coefs, axis=0).tolist(),
        "n_splits": n_splits,
    }


def _log() -> pd.DataFrame:
    return pd.read_csv(LOG_PATH, dtype={"run_id": str})


def describe(run_id: str) -> str:
    row = _log().loc[lambda d: d["run_id"] == str(run_id)]
    if row.empty:
        raise KeyError(f"run_id {run_id!r} not in {LOG_PATH}")
    return f"{row.iloc[0]['model']}"


def load_test_pred(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = SUBMISSION_DIR / f"{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no saved test predictions for run {run_id!r} at {path}. "
            f"Re-run it without --no-submission."
        )
    frame = pd.read_csv(path)
    return frame["id"].to_numpy(), frame["addicted_label"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run_ids to blend")
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--all-subsets", action="store_true", help="score every subset of >=2 runs")
    parser.add_argument("--submit", action="store_true", help="write submissions/blend_<ids>.csv")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument(
        "--logit-stack",
        action="store_true",
        help="compare a logistic stacker on OOF logits against the equal-weight rank average",
    )
    args = parser.parse_args()

    _, y = load_train()
    oofs = {r: load_oof(r) for r in args.runs}

    print("inputs")
    for r, v in oofs.items():
        print(f"  {r} {describe(r):<22} OOF AUC {roc_auc_score(y, v):.6f}")

    best_single = max(args.runs, key=lambda r: roc_auc_score(y, oofs[r]))
    print(f"\nstrongest single model: {best_single} ({roc_auc_score(y, oofs[best_single]):.6f})")

    if args.all_subsets:
        print("\nevery subset, equal weights (rank-average)\n")
        rows = []
        for k in range(2, len(args.runs) + 1):
            for combo in itertools.combinations(args.runs, k):
                b = rank_average([oofs[r] for r in combo])
                rows.append(("+".join(combo), roc_auc_score(y, b)))
        rows.sort(key=lambda t: -t[1])
        base = roc_auc_score(y, oofs[best_single])
        print(f"{'blend':<28}{'OOF AUC':>10}{'vs best single':>16}")
        print("-" * 54)
        for name, auc in rows:
            print(f"{name:<28}{auc:>10.6f}{auc - base:>+16.6f}")
        return

    if args.logit_stack:
        res = logit_stack_score(y, [oofs[r] for r in args.runs])
        print(f"\nhonest comparison over {res['n_splits']} half-splits "
              f"(combiner fit on one half, scored on the other)\n")
        print(f"  equal-weight rank average  {res['rank_average']:.6f}")
        print(f"  logistic stack on logits   {res['logit_stack']:.6f}")
        print(f"  difference                 {res['diff']:+.6f}  (sd {res['diff_sd']:.6f})")
        print("\n  stacker coefficients (negative = used as a correction):")
        for r, c in zip(args.runs, res["coefs"]):
            print(f"    {r} ({describe(r)}): {c:+.4f}")
        return

    blended = rank_average([oofs[r] for r in args.runs], args.weights)
    auc = roc_auc_score(y, blended)
    print(f"\nblend of {'+'.join(args.runs)}"
          f"{'' if args.weights is None else ' weights ' + str(args.weights)}")
    print(f"  blended OOF AUC {auc:.6f}")

    res = bootstrap_auc_diff(y, oofs[best_single], blended, n_boot=args.n_boot)
    print(f"  vs {best_single} alone   {res['diff']:+.6f}  "
          f"95% CI [{res['ci_low']:+.6f}, {res['ci_high']:+.6f}]")
    print("  -> " + (
        "the blend is a real improvement." if res["significant"] and res["diff"] > 0
        else "WORSE than the single model." if res["significant"]
        else "indistinguishable from the single model. Not an improvement."
    ))

    if args.submit:
        ids = None
        preds = []
        for r in args.runs:
            rid, p = load_test_pred(r)
            if ids is None:
                ids = rid
            elif not np.array_equal(ids, rid):
                raise ValueError(f"run {r} submission ids do not match run {args.runs[0]}")
            preds.append(p)
        test_blend = rank_average(preds, args.weights)
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        out = SUBMISSION_DIR / f"blend_{'_'.join(args.runs)}.csv"
        pd.DataFrame({"id": ids, "addicted_label": test_blend}).to_csv(out, index=False)
        print(f"\n  wrote {out}")
        print("  NOTE: not logged. experiments/log.csv rows are single runs of the")
        print("        harness; a blend has no fit of its own to record.")


if __name__ == "__main__":
    main()

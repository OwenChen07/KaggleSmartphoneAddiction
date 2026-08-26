"""Phase 6 — randomised hyperparameter search over HistGBM.

    python -m src.tuning --n-iter 40 --search-rows 250000

The search is run with `cv=3` on a **subsample** of the training rows, purely
to keep the runtime sane: a 40-candidate search at 5-fold on all 691k rows is
40x the cost of a full run. Nothing from the search is ever reported as a
result. The winning parameters are handed back to the standard full-data
5-fold `run_cv`, and *that* number is the one that goes in the log — so the
official figure is measured on the same folds, at the same size, as every
other row in `experiments/log.csv` and stays paired with them.

The subsample is a real cost, not a free saving: at 250k rows the model is
weaker overall, so the search optimises a slightly different problem than the
one it is selecting for. The usual symptom is that it picks more capacity than
the full data needs, or less. That is the trade being made for the runtime,
and it is why the winner is re-measured rather than trusted.

Search space, and why each range:

- `learning_rate` — log-uniform 0.01 to 0.3. The single most important knob;
  interacts directly with `max_iter`.
- `max_iter` — 100 to 600. Low learning rates need more trees to get there.
- `max_leaf_nodes` — 15 to 127. Tree capacity. The default 31 is the main
  thing worth challenging on a 691k-row dataset.
- `max_depth` — None, 6, 8, 12. `None` lets `max_leaf_nodes` govern shape
  alone; a cap changes which shapes are reachable.
- `min_samples_leaf` — 20 to 500. At 691k rows the default 20 is very
  permissive and is a plausible source of variance.
- `l2_regularization` — 0 to 10, log-ish. Shrinks leaf values.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold  # noqa: E402

from .data import load_test, load_train  # noqa: E402
from .models import histgbm_features  # noqa: E402
from .validation import SEED, log_run, run_cv, save_oof, save_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEST_PARAMS_PATH = ROOT / "experiments" / "best_params.json"

SEARCH_SPACE = {
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__max_iter": randint(100, 600),
    "clf__max_leaf_nodes": randint(15, 128),
    "clf__max_depth": [None, 6, 8, 12],
    "clf__min_samples_leaf": randint(20, 500),
    "clf__l2_regularization": loguniform(1e-3, 10),
}


def run_search(X: pd.DataFrame, y: np.ndarray, *, n_iter: int, cv: int, seed: int):
    """RandomizedSearchCV over the Phase 4 pipeline.

    Scoring is `roc_auc`, matching the competition metric. `refit=False` — the
    search only needs to identify parameters; refitting happens in the full
    5-fold run afterwards.
    """
    search = RandomizedSearchCV(
        histgbm_features(),
        SEARCH_SPACE,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        refit=False,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X, y)
    return search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--search-rows", type=int, default=250_000)
    parser.add_argument("--search-folds", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--no-final", action="store_true", help="search only, skip the full run")
    parser.add_argument(
        "--params",
        default=None,
        help=(
            "JSON dict of clf__ params. Skips the search entirely and takes this "
            "configuration straight to the full-data 5-fold run. Used to re-measure "
            "runner-up candidates from a search that has already happened."
        ),
    )
    parser.add_argument("--label", default=None, help="description for a --params run")
    args = parser.parse_args()

    if args.params is not None:
        params = json.loads(args.params)
        X, y = load_train()
        X_test, test_ids = load_test()
        est = histgbm_features()
        est.set_params(**params)
        result = run_cv(
            est, X, y, X_test,
            description=args.label or "Phase 6: HistGBM + engineered, given params",
            model="histgbm_tuned",
        )
        run_id = log_run(result)
        save_oof(result, run_id)
        path = save_submission(result, test_ids, run_id)
        print(f"  logged as run {run_id} -> {path}")
        return

    X_search, y_search = load_train(sample=args.search_rows)
    print(f"search on {len(X_search):,} rows, cv={args.search_folds}, {args.n_iter} candidates")
    print("(subsampled for runtime; the winner is re-measured on full data)\n", flush=True)

    search = run_search(
        X_search, y_search, n_iter=args.n_iter, cv=args.search_folds, seed=SEED
    )

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cols = [c for c in results.columns if c.startswith("param_")]
    print(f"\ntop {args.top} candidates (subsampled CV — not comparable to the log)\n")
    show = results.head(args.top)[["mean_test_score", "std_test_score", *cols]]
    show = show.rename(columns=lambda c: c.replace("param_clf__", ""))
    print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    best = {k: v for k, v in search.best_params_.items()}
    print(f"\nbest subsampled CV score: {search.best_score_:.6f}")
    print("best params:")
    for k, v in sorted(best.items()):
        print(f"  {k} = {v!r}")

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(
        json.dumps(
            {
                "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
                "search_rows": int(len(X_search)),
                "search_folds": int(args.search_folds),
                "n_iter": int(args.n_iter),
                "search_score": float(search.best_score_),
            },
            indent=2,
        )
    )
    print(f"\nwrote {BEST_PARAMS_PATH}")

    if args.no_final:
        return

    print("\n=== re-measuring the winner on full data, standard 5-fold ===", flush=True)
    X, y = load_train()
    X_test, test_ids = load_test()
    tuned = histgbm_features()
    tuned.set_params(**{k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()})
    result = run_cv(
        tuned, X, y, X_test, description="Phase 6: tuned HistGBM + engineered", model="histgbm_tuned"
    )
    run_id = log_run(result)
    save_oof(result, run_id)
    path = save_submission(result, test_ids, run_id)
    print(f"  logged as run {run_id} -> {path}")


if __name__ == "__main__":
    main()

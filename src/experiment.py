"""Experiment runner.

    python -m src.experiment --model baseline
    python -m src.experiment --model all
    python -m src.experiment --model histgbm --sample 50000 --no-log

Runs a model from the zoo through the shared CV harness, writes a submission
and appends a row to experiments/log.csv. Sampled runs are never logged — the
log is the record backing the README, and a row in it must mean a full-data
run.
"""

from __future__ import annotations

import argparse
import warnings

# numpy 2.x built against Apple's Accelerate BLAS raises spurious floating-point
# warnings from `matmul` — they fire on clean random input with no NaN, inf or
# rank deficiency anywhere, and the results are unaffected. Verified by
# reproducing on `rng.standard_normal((100000, 29)) @ w`. Silenced so that a
# genuine numerical problem is not lost in the noise. Do not widen this filter.
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

from .data import load_test, load_train  # noqa: E402
from .models import MODELS
from .validation import log_run, run_cv, save_oof, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", choices=[*MODELS, "all"])
    parser.add_argument("--sample", type=int, default=None, help="row subsample; forces --no-log")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-submission", action="store_true")
    args = parser.parse_args()

    sampled = args.sample is not None
    if sampled and not args.no_log:
        print("note: --sample given, forcing --no-log (sampled runs never enter the log)")
    should_log = not (args.no_log or sampled)
    want_test = not (args.no_submission or sampled)

    print("loading data...", flush=True)
    X, y = load_train(sample=args.sample)
    X_test, test_ids = (load_test() if want_test else (None, None))
    print(f"train {X.shape}  positive rate {y.mean():.4f}", flush=True)

    names = list(MODELS) if args.model == "all" else [args.model]
    results = {}

    for name in names:
        build, description = MODELS[name]
        print(f"\n=== {name}: {description} ===", flush=True)
        result = run_cv(
            build(),
            X,
            y,
            X_test,
            description=description,
            model=name,
            n_splits=args.folds,
        )
        results[name] = result

        if should_log:
            run_id = log_run(result)
            save_oof(result, run_id)
            if want_test:
                path = save_submission(result, test_ids, run_id)
                print(f"  logged as run {run_id} -> {path}", flush=True)
            else:
                print(f"  logged as run {run_id}", flush=True)

    if len(results) > 1:
        print("\n" + "=" * 78)
        print(f"{'model':<20}{'feats':>7}{'OOF AUC':>10}{'fold mean':>12}{'fold std':>11}{'secs':>9}")
        print("-" * 78)
        for name, r in sorted(results.items(), key=lambda kv: -kv[1].oof_auc):
            print(
                f"{name:<20}{r.n_model_features:>7}{r.oof_auc:>10.5f}{r.fold_mean:>12.5f}"
                f"{r.fold_std:>11.5f}{r.fit_seconds:>9.0f}"
            )


if __name__ == "__main__":
    main()

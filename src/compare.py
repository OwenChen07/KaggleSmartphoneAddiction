"""Paired bootstrap comparison between two logged runs.

    python -m src.compare 002 005

Answers the only question that matters when one model scores higher than
another: is the gap bigger than the noise? Because every run shares the same
seed and fold assignment, the two OOF vectors are row-aligned, so the same
bootstrap resample can be applied to both and the shared row-level noise
cancels.

A difference whose 95% interval straddles zero is not an improvement, however
much better the headline number looks.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import pandas as pd  # noqa: E402

from .data import load_train  # noqa: E402
from .validation import LOG_PATH, bootstrap_auc_diff, load_oof  # noqa: E402


def _describe(run_id: str) -> str:
    log = pd.read_csv(LOG_PATH, dtype={"run_id": str})
    row = log.loc[log["run_id"] == str(run_id)]
    if row.empty:
        raise KeyError(f"run_id {run_id!r} not in {LOG_PATH}")
    return f"{run_id} ({row.iloc[0]['model']}: {row.iloc[0]['description']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="baseline run_id")
    parser.add_argument("run_b", help="challenger run_id")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    _, y = load_train()
    oof_a, oof_b = load_oof(args.run_a), load_oof(args.run_b)

    res = bootstrap_auc_diff(y, oof_a, oof_b, n_boot=args.n_boot)

    print(f"A = {_describe(args.run_a)}")
    print(f"B = {_describe(args.run_b)}")
    print()
    print(f"  AUC(A)          {res['auc_a']:.5f}")
    print(f"  AUC(B)          {res['auc_b']:.5f}")
    print(f"  difference      {res['diff']:+.5f}")
    print(f"  95% CI          [{res['ci_low']:+.5f}, {res['ci_high']:+.5f}]  ({res['n_boot']} resamples)")
    print()
    if res["significant"]:
        direction = "better" if res["diff"] > 0 else "WORSE"
        print(f"  -> B is {direction} than A; the interval excludes zero.")
    else:
        print("  -> indistinguishable: the interval includes zero. Not an improvement.")


if __name__ == "__main__":
    main()

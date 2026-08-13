"""Cross-validation harness — the spine of this project.

Every model, every feature set, every experiment goes through `run_cv` with the
same `SEED`, so every run sees byte-identical folds. That is what makes model A
vs model B a *paired* comparison rather than a comparison confounded by the
split. `bootstrap_auc_diff` then puts a confidence interval on the difference,
so "raised AUC from A to B" is a claim with an error bar attached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "experiments" / "log.csv"
SUBMISSION_DIR = ROOT / "submissions"
OOF_DIR = ROOT / "experiments" / "oof"

SEED = 42
N_SPLITS = 5

LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "description",
    "model",
    "n_features",
    "n_model_features",
    "n_train",
    "oof_auc",
    "fold_mean",
    "fold_std",
    "fit_seconds",
    "public_lb",
    "gap",
]


@dataclass
class CVResult:
    description: str
    model: str
    y: np.ndarray
    oof: np.ndarray
    test_pred: np.ndarray | None
    fold_aucs: list[float] = field(default_factory=list)
    n_features: int = 0
    n_model_features: int = 0
    n_train: int = 0
    fit_seconds: float = 0.0

    @property
    def oof_auc(self) -> float:
        """AUC over the full out-of-fold vector. This is the headline number —
        it is computed once over all rows, not averaged across folds."""
        return float(roc_auc_score(self.y, self.oof))

    @property
    def fold_mean(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def fold_std(self) -> float:
        """Spread across folds — the noise floor. A gain between two models
        that is small relative to this is not yet evidence of anything."""
        return float(np.std(self.fold_aucs))

    def summary(self) -> str:
        return (
            f"{self.description:<38} OOF AUC {self.oof_auc:.5f}  "
            f"folds {self.fold_mean:.5f} +/- {self.fold_std:.5f}  "
            f"({self.fit_seconds:.0f}s)"
        )


def _count_model_features(fitted_estimator, fallback: int) -> int:
    """Width of the matrix the estimator actually sees, after preprocessing.

    Differs from the input column count whenever encoding changes the width —
    one-hot expansion, added indicators — so the log records what the model was
    really fit on rather than what was handed to the pipeline.
    """
    try:
        return int(len(fitted_estimator.named_steps["prep"].get_feature_names_out()))
    except Exception:
        return fallback


def run_cv(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame | None = None,
    *,
    description: str,
    model: str | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    verbose: bool = True,
) -> CVResult:
    """Stratified K-fold CV producing out-of-fold predictions and, if `X_test`
    is given, fold-averaged test predictions.

    The estimator is `clone`d per fold and fit only on that fold's training
    rows, so any preprocessing inside the pipeline never sees validation data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64) if X_test is not None else None
    fold_aucs: list[float] = []
    n_model_features = 0
    start = time.perf_counter()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y[train_idx])

        if fold == 1:
            n_model_features = _count_model_features(est, X.shape[1])

        oof[val_idx] = est.predict_proba(X.iloc[val_idx])[:, 1]
        fold_auc = float(roc_auc_score(y[val_idx], oof[val_idx]))
        fold_aucs.append(fold_auc)

        if test_pred is not None:
            test_pred += est.predict_proba(X_test)[:, 1] / n_splits

        if verbose:
            print(f"  fold {fold}/{n_splits}  AUC {fold_auc:.5f}", flush=True)

    result = CVResult(
        description=description,
        model=model or type(estimator).__name__,
        y=y,
        oof=oof,
        test_pred=test_pred,
        fold_aucs=fold_aucs,
        n_features=X.shape[1],
        n_model_features=n_model_features,
        n_train=len(X),
        fit_seconds=time.perf_counter() - start,
    )
    if verbose:
        print(result.summary(), flush=True)
    return result


# --------------------------------------------------------------------------
# Experiment log
# --------------------------------------------------------------------------


def _next_run_id() -> str:
    if not LOG_PATH.exists():
        return "001"
    log = pd.read_csv(LOG_PATH, dtype={"run_id": str})
    if log.empty:
        return "001"
    return f"{int(log['run_id'].astype(int).max()) + 1:03d}"


def log_run(result: CVResult, run_id: str | None = None) -> str:
    """Append one row to experiments/log.csv and return the run_id.

    `public_lb` and `gap` are left blank; fill them with `record_lb` once the
    submission has actually scored. Nothing is projected or estimated here.
    """
    run_id = run_id or _next_run_id()
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": result.description,
        "model": result.model,
        "n_features": result.n_features,
        "n_model_features": result.n_model_features,
        "n_train": result.n_train,
        "oof_auc": round(result.oof_auc, 6),
        "fold_mean": round(result.fold_mean, 6),
        "fold_std": round(result.fold_std, 6),
        "fit_seconds": round(result.fit_seconds, 1),
        "public_lb": "",
        "gap": "",
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row], columns=LOG_COLUMNS)
    frame.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
    return run_id


def record_lb(run_id: str, public_lb: float) -> None:
    """Fill in the public LB score for a run and compute the OOF-to-LB gap.

    The gap column across runs is the artifact this project exists to produce.
    """
    log = pd.read_csv(LOG_PATH, dtype={"run_id": str})
    mask = log["run_id"] == str(run_id)
    if not mask.any():
        raise KeyError(f"run_id {run_id!r} not in {LOG_PATH}")
    log.loc[mask, "public_lb"] = public_lb
    log.loc[mask, "gap"] = round(float(log.loc[mask, "oof_auc"].iloc[0]) - public_lb, 6)
    log.to_csv(LOG_PATH, index=False)


def save_oof(result: CVResult, run_id: str) -> Path:
    """Persist the OOF vector so any two runs can be compared later without
    refitting. Because every run uses the same seed and fold assignment, OOF
    vectors from different runs are row-aligned and directly comparable."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    path = OOF_DIR / f"{run_id}_{result.model}.npy"
    np.save(path, result.oof)
    return path


def load_oof(run_id: str) -> np.ndarray:
    matches = sorted(OOF_DIR.glob(f"{run_id}_*.npy"))
    if not matches:
        raise FileNotFoundError(f"no OOF vector saved for run {run_id!r}")
    return np.load(matches[0])


def save_submission(result: CVResult, test_ids: np.ndarray, run_id: str) -> Path:
    if result.test_pred is None:
        raise ValueError("CVResult has no test predictions — pass X_test to run_cv")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / f"{run_id}.csv"
    pd.DataFrame({"id": test_ids, "addicted_label": result.test_pred}).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Is the difference real?
# --------------------------------------------------------------------------


def bootstrap_auc_diff(
    y: np.ndarray,
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    n_boot: int = 500,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired bootstrap CI on AUC(b) - AUC(a) over the OOF predictions.

    Both models are resampled on the *same* bootstrap indices, so the shared
    row-level noise cancels and the interval reflects only the difference
    between the models. If the interval straddles zero, the improvement is not
    distinguishable from noise and should not be claimed as one.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx]
        if y_boot.min() == y_boot.max():  # degenerate resample, AUC undefined
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(y_boot, oof_b[idx]) - roc_auc_score(y_boot, oof_a[idx])

    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "auc_a": float(roc_auc_score(y, oof_a)),
        "auc_b": float(roc_auc_score(y, oof_b)),
        "diff": float(roc_auc_score(y, oof_b) - roc_auc_score(y, oof_a)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "n_boot": int(len(diffs)),
    }

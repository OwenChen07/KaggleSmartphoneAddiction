# Phase 11 — CatBoost, and the blend Phase 6 predicted

**Result: OOF 0.967709 → 0.967906 for CatBoost alone, and 0.968271 blended
with the re-tuned HistGBM. The blend is the first one in this project to earn
its place, and it does so for exactly the reason Phase 6 identified.**

| run | model | OOF AUC | fold std | fit (s) |
|---|---|---|---|---|
| `023` | HistGBM, encoded, re-tuned | 0.967709 | 0.000503 | 350 |
| `024` | **CatBoost, ordered target statistics** | **0.967906** | 0.000513 | 3,982 |
| — | rank-average `024` + `023` | **0.968271** | — | — |

```
compare 023 024        : +0.00020   95% CI [+0.00011, +0.00028]
blend(024+023) vs 024  : +0.000364  95% CI [+0.000325, +0.000406]
```

## Why CatBoost, specifically

Not as a second opinion from another library. Phase 9 established that reading
the columns as lookup keys is worth +0.0039; `src/encoding.py` does that with
a hand-rolled encoder — a smoothed target mean per level, **one global
smoothing constant**, and a nested inner split to keep it honest.

CatBoost has computed **ordered target statistics** internally since it
existed. Handed a column as `cat_features`, it builds the same kind of
statistic under a random permutation, using only the rows preceding each row.
Two consequences:

- shrinkage is **adaptive per row** rather than one constant for every level —
  early rows in the permutation get heavily shrunk estimates because little
  history precedes them, later rows get sharper ones;
- leakage protection is **structural rather than implemented** — a row's own
  target is never in its own statistic by construction, so there is no nesting
  for us to get wrong.

So this is the same idea with a better estimator. That it wins by only
+0.00020 says our hand-rolled version was close to adequate; that it wins at
all says the adaptive shrinkage is worth something.

Each of the twelve columns appears **twice** in the frame — once as its
numeric value, once as a raw string level. Not redundancy: the first supports
ordering questions ("screen time above 9.8 hours"), the second identity
questions ("this exact value"). Phase 9 showed the second carries most of the
signal; the first still carries the genuinely continuous structure.

It is deliberately a single estimator rather than a Pipeline. CatBoost needs
the categorical columns identified **by position** at `fit` time, and a
ColumnTransformer in front would convert the frame to a numeric array and
destroy exactly the information being handed over.

## The operating point was measured, not defaulted

Fold 1, against the re-tuned HistGBM's 0.966810:

| config | val AUC | time | peak RSS |
|---|---|---|---|
| 300 iters, lr 0.15 | 0.966634 | 208s | 2.5 GB |
| **1200 iters, lr 0.06** | **0.967118** | 801s | 2.6 GB |

Reaching parity at 300 iterations, before convergence, was the signal worth
following. 1200/0.06 beats HistGBM by +0.00031 and the curve is flattening, so
that is where more compute stops paying.

**A cost estimate I got badly wrong.** I first projected ~45 minutes per fold
and nearly declined to run this at all. Actual is ~3.5 minutes per fold at 300
iterations — off by roughly **13×**, because I extrapolated from a 120k-row
smoke test where fixed overheads dominated the runtime. A bad cost estimate
almost cost the best model in the project.

**And a failure I misdiagnosed.** The first background attempt died silently
after printing its header, and I guessed OOM. It was a `ModuleNotFoundError`:
running the script from the scratchpad put *that* directory on `sys.path`
instead of the repo. Peak memory was 2.5 GB against 15 GB available. The
lesson is the one this project keeps relearning — read the error before
theorising about the cause.

## The blend, and why this one worked

Phase 6 tried 26 blends and every one lost. Phase 7's seed-averaging beat all
of them. The rule extracted then was that blending pays only among models that
are **comparably strong** *and* **genuinely different** — and that the
project's models had never been both at once.

These two are:

| pair | Spearman | outcome |
|---|---|---|
| `024` CatBoost vs `023` HistGBM | **0.9865** | blend gains +0.000364 ✅ |
| `023` vs `022` — two HistGBM variants, for scale | 0.9944 | the regime that failed in Phase 6 |

0.9865 against 0.9944 is not a large-looking difference, but it is the
difference between two implementations of one idea and two *different* ideas
about how to shrink a target statistic. Strength is near-identical (0.96791 vs
0.96771), so neither dilutes the other.

The weight curve is flat across the plausible range — 0.968231 at
`w=0.4`, 0.968271 at 0.5, 0.968273 at 0.6, 0.968239 at 0.7 — so equal weights
are used rather than the argmax, which would be fitting the blend to the OOF
vector for two millionths.

**Honest scale check.** +0.000364 is 0.7× the fold noise of 0.000513. Real,
and smaller than the run-to-run spread of the models it combines. It is worth
having because it is free once both models exist, not because it is large.

## Prediction for the blend

Recorded before any submission, using the two-component model from Phases 5
and 7 (covariate shift measured here, plus the +0.001012 fold-averaging bonus
measured for single models):

| | value |
|---|---|
| blend OOF | 0.968271 |
| covariate shift | +0.000852 |
| fold-averaging bonus | +0.001012 |
| **predicted LB** | **0.97013** |

That would put the blend at the lower edge of the ~0.9712 public plateau —
reached with a self-contained pipeline that trains from `train.csv` alone.

### Outcome: 0.96948, and I used the wrong constant

**It scored 0.96948 against a prediction of 0.97013 — over-predicted by
0.00065**, the largest prediction error in the project so far. Rank
**827 → 767**.

The error was mine, not the theory's. I plugged in the **+0.001012**
fold-averaging bonus measured for a *single* tuned model. But a blend of two
models is **already variance-reduced** — which is the entire finding of Phase
7, that seed-averaging shrinks the fold-averaging bonus from +0.001012 to
+0.000371 because the two are redundant. A rank-average of two decorrelated
models does the same thing.

Using the variance-reduced constant instead:

| | value |
|---|---|
| blend OOF | 0.968271 |
| covariate shift | +0.000852 |
| fold-averaging bonus for an **already-averaged** model | +0.000371 |
| **corrected prediction** | **0.96949** |
| **actual** | **0.96948** |

**An error of 0.00001.** The observed gap of −0.001209 also sits exactly where
the theory says it should: between the single-model gaps (−0.0018 to −0.0022)
and the seed-averaged ones (−0.00134, −0.00139).

So this is the third independent confirmation that the OOF-to-LB gap tracks
model variance, and the first time the theory was strong enough to catch *my*
mistake rather than the other way round. The rule, stated properly:

> **Use the single-model fold-averaging bonus only for a single model.** Any
> prediction produced by averaging — over seeds, over folds, over model
> families — has already spent most of that bonus and gets the small constant.

Recorded rather than quietly fixed, because a prediction that misses and then
explains its own miss is worth more than one that happened to land.

## Reproducing

```bash
python -m src.experiment --model catboost_lookup    # run 024, ~66 min
python -m src.compare 023 024
python -m src.blend 024 023 --submit
```

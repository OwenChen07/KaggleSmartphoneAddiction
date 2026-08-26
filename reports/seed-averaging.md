# Phase 7 — Seed averaging, and a confirmed prediction

**Result: OOF 0.963192 → 0.964060 (+0.000868, CI `[+0.00082, +0.00092]`), public
LB 0.96511 → 0.96540, rank 1678 → 1554. It also produced the sharpest
successful forward prediction in the project: the leaderboard score was called
in advance to within 0.00005.**

Run `017`.

## What it does

`SeedAveraged` (in `src/models.py`) trains 5 copies of the tuned pipeline that
differ *only* by `random_state`, and averages their ranked predictions.

HistGradientBoosting is not fully deterministic given its data. Two things are
drawn at random and both are driven by `random_state`:

- the **bin edges**, computed from a 200,000-row subsample of the training data
- the **early-stopping validation split** (`validation_fraction=0.1`)

So re-seeding gives a genuinely different model — just a mildly different one.
Measured pairwise Spearman between seeds: **0.993**. Highly correlated, but not
identical, and averaging cancels the part that is not shared.

This is **variance reduction, not bias reduction**. It cannot find structure a
single model missed; it only stops one model's arbitrary choices from reaching
the prediction. That is why the gain is bounded, and why it appears reliably
rather than sometimes.

### Why it is a wrapper and not a change to `run_cv`

The fold assignment is untouched. The OOF vector therefore stays row-aligned
with all 16 earlier runs and remains comparable by paired bootstrap. This was
the explicit reason for preferring seed averaging over raising `K`: more folds
would have bought a similar amount while invalidating every paired comparison
in `experiments/log.csv`.

## How many seeds

Cumulative rank-average on fold 1, full training data:

| seeds | AUC | gain vs 1 |
|---|---|---|
| 1 | 0.962679 | — |
| 2 | 0.963188 | +0.000509 |
| 3 | 0.963306 | +0.000627 |
| 4 | 0.963377 | +0.000698 |
| **5** | **0.963375** | **+0.000696** |
| 8 | 0.963413 | +0.000735 |
| 12 | 0.963449 | +0.000770 |
| 16 | 0.963467 | +0.000788 |

Saturates fast. Five seeds captures **88%** of what sixteen deliver; going
5 → 16 buys +0.00009 for 3.2× the compute. Variance reduction goes as 1/k, and
because the members are 0.993 correlated there is little independent error to
remove. **Five seeds.**

`max_features` was tested as a way to force more diversity, and rejected:

| setting | 5-seed rank-avg (fold 1) |
|---|---|
| 1.0 (default) | 0.963371 |
| 0.8 | 0.963391 |
| 0.6 | 0.963058 |

At 0.8 the difference is noise; at 0.6 the individual models degrade faster
than the added diversity repays.

## Result

| run | model | OOF AUC | fold std | public LB | gap |
|---|---|---|---|---|---|
| `012` | tuned, single | 0.963192 | 0.000474 | 0.96511 | −0.001918 |
| `017` | **tuned, 5-seed average** | **0.964060** | 0.000514 | **0.96540** | **−0.001340** |

```
compare 012 017 :  +0.000868   95% CI [+0.00082, +0.00092]  -> better
```

The gain is **1.7× the fold noise** — the third result in this project that is
both statistically real and practically worth having, after the engineered
features (+0.0014) and tuning (+0.0071).

It also beats the Phase 6 blend, which is the same idea done worse:

```
blend(012+014) 0.963472  vs  seedavg 0.964060 : +0.000587  CI [+0.000538,+0.000633]
```

And blending the runner-up model *into* the seed average now **hurts**
(−0.000229, CI `[-0.000261, -0.000194]`), exactly as the Phase 6 dilution
finding predicts: run `014` is weaker, so once the strong model is already
variance-reduced there is nothing left for a weaker partner to contribute.

## The prediction

Phase 6 left the OOF-to-LB gap partly unexplained. The README's revised claim
was that the gap tracks **model variance**, because test predictions are an
average of 5 fold models while each OOF row is scored by one — and averaging
helps a high-variance model more.

Seed averaging is a direct test of that claim, because it *removes variance
before the fold-averaging happens*. If the theory holds, the fold-averaging
bonus should shrink and the gap should get smaller.

Measured on an 80/20 same-population split, comparing on identical rows:

| model | single | 5-fold average | ensemble gain |
|---|---|---|---|
| tuned single (`012`) | 0.962207 | 0.963220 | +0.001012 |
| seed-averaged (`017`) | 0.962900 | 0.963271 | **+0.000371** |

**Seed averaging cut the fold-averaging bonus by 63%.** The theory holds.

That yields two competing predictions, both written down before submitting:

| prediction | reasoning | predicted LB |
|---|---|---|
| **variance theory** | covariate shift +0.000923 + ensemble +0.000371 = gap −0.00129 | **0.96535** |
| naive | run `012`'s gap of −0.001918 simply holds | 0.96598 |

**It scored 0.96540**, a gap of **−0.001340**.

| prediction | error |
|---|---|
| variance theory (0.96535) | **+0.00005** |
| naive constant gap (0.96598) | +0.00058 |

The variance-based prediction was **more than 10× more accurate**. The gap
table's non-monotonicity is now explained rather than merely recorded: `012`
spiked to −0.0019 because it is a high-variance model, and `017` fell back to
−0.0013 because averaging removed most of that variance before the leaderboard
ever saw it.

## Full gap table

| run | model | OOF AUC | public LB | gap |
|---|---|---|---|---|
| `002` | HistGBM, numeric only | 0.954677 | 0.95576 | −0.001083 |
| `005` | HistGBM, numeric + categorical | 0.954704 | 0.95578 | −0.001076 |
| `008` | HistGBM + engineered features | 0.956091 | 0.95708 | −0.000989 |
| `012` | tuned, single | 0.963192 | 0.96511 | −0.001918 |
| `017` | tuned, 5-seed average | 0.964060 | 0.96540 | −0.001340 |

## A known defect in the log

Run `017`'s `n_model_features` is recorded as **12**, which is wrong — the true
value is 27. `validation._count_model_features` reaches for
`named_steps["prep"]`, which the `SeedAveraged` wrapper did not expose, so it
fell back to the raw input column count. A `named_steps` property was added to
the wrapper so future runs record correctly. **The existing row was left
uncorrected**, because `experiments/log.csv` is append-only and editing a
logged value is worse than documenting a wrong one.

## Reproducing

```bash
python -m src.experiment --model histgbm_seedavg     # run 017
python -m src.compare 012 017
```

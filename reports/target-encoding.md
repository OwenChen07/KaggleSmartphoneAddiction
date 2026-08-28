# Phase 9 — Reading the columns as lookup keys

**Result: OOF 0.963192 → 0.967257. The largest gain in the project after
tuning, and unlike tuning it came from changing the *representation* rather
than the model.**

| run | model | OOF AUC | fold std | vs previous best |
|---|---|---|---|---|
| `012` | tuned, no encoding | 0.963192 | 0.000474 | — |
| `020` | previous best (imputed + seed avg) | 0.964709 | 0.000517 | — |
| `021` | **+ target encoding + lattice** | **0.967130** | 0.000521 | **+0.00242** `[+0.00228, +0.00256]` |
| `022` | + imputation as well | **0.967257** | 0.000458 | +0.00013 `[+0.00005, +0.00021]` |

```
compare 012 021 :  +0.00394   95% CI [+0.00378, +0.00409]
compare 020 021 :  +0.00242   95% CI [+0.00228, +0.00256]
compare 021 022 :  +0.00013   95% CI [+0.00005, +0.00021]
```

---

## The thing we had wrong for eight phases

`reports/eda.md` records `notifications_per_day` at univariate AUC **0.492**
and `app_opens_per_day` at **0.541**. Every phase since treated them as
near-worthless numeric quantities, and the Phase 4 permutation-importance
table put them mid-table on that basis.

**They are not quantities.** The generator uses the exact value as a key into
a lookup table. Measured here:

| column | levels | rows/level | per-value target-rate sd | sd from sampling alone | ratio | mean \|rate diff\| between adjacent values |
|---|---|---|---|---|---|---|
| `notifications_per_day` | 231 | 2,700 | 0.1919 | 0.0087 | **22.0×** | **0.2248** |
| `app_opens_per_day` | 166 | 3,679 | 0.1952 | 0.0075 | **26.1×** | **0.2170** |
| `age` | 18 | 36,802 | 0.0397 | 0.0024 | 16.8× | 0.0590 |
| `daily_screen_time_hours` | 1,108 | 535 | 0.3259 | 0.0196 | 16.6× | 0.0713 |

87 notifications and 88 notifications have nothing to do with each other — the
addiction rate between neighbouring values differs by 0.22 on average, with
~2,700 rows behind each. That is 22× more than sampling noise can produce.

The cost of the wrong representation, measured on those two columns alone:

| the same two columns | OOF AUC |
|---|---|
| as raw numbers | **0.5122** |
| as target-encoded lookup keys | **0.8170** |

A tree cannot bridge that. It splits on `x <= t`, which asks an ordering
question about a variable whose order carries no meaning. **Chance to 0.82 was
sitting in columns the EDA had written off**, and no amount of tuning,
ensembling or feature engineering on top of the numeric view could reach it.

Note what this says about the earlier phases. Every negative result in
`reports/tuning-and-ensembling.md` — the saturation at 0.963, the ten
configurations landing within 0.0003 of each other — was *correct*, and
correct about the wrong question. They measured that no more signal was
reachable **from that representation**. Use a ceiling estimate to stop tuning;
never to stop looking.

## What was built

`src/encoding.py` — `TargetFrequencyEncoder`. Two columns per input column:

- **target**: the smoothed mean of `y` among rows sharing the exact level,
  `(sum_y + prior·s) / (count + s)` with `s = 10`.
- **frequency**: the share of rows at that level, which separates a common
  level whose mean is well estimated from a rare one whose mean is not.

Applied to **all twelve** columns, continuous ones included. That works here
because of the counts: at 691,369 rows even `daily_screen_time_hours` has ~500
rows behind each of its 1,389 distinct values, so a smoothed mean is a
well-estimated quantity at every one.

`src/features.py` — `GeneratorFeatures`, two further channels:

- **The decimal lattice.** `frac(x)` and `floor(10x) % 10` for the six
  fractional columns. Rows whose `daily_screen_time_hours` ends in `.0` are
  addicted at 0.6513 and rows ending in `.2` at 0.7365 — an **8.5 point
  swing** against a base rate of 0.7094, across 50–68k rows per digit.
  `weekend_screen_time` swings 10.5 points. This is a *separate channel* from
  target encoding, which estimates every exact value independently and so
  cannot express "everything ending in .2 shares something".
- **The accounting identity.** `daily ≥ social + gaming + work_study` holds
  with 36 violations in 421,427 rows, all at −1e-6 — float rounding, not
  exceptions. `unaccounted_screen` is the generator's remainder bucket and has
  a univariate AUC of **0.7657** on its own.

Phase 4's `residual_screen` subtracted only two of the three terms and never
expressed the identity. **It was deliberately left unchanged.** Runs 008–020
were fit with it, and redefining it would make them unreproducible while the
log still claimed otherwise — `--oof-as` would catch it, which is precisely
why it must not be done quietly. The corrected version arrives under a new
name.

### Isolating the two channels (fold 1)

| config | val AUC | Δ |
|---|---|---|
| tuned baseline | 0.962679 | — |
| + decimal lattice only | 0.964536 | +0.00186 |
| + target encoding only | 0.965873 | +0.00319 |
| + both | 0.966281 | **+0.00360** |

They are close to additive, which supports the claim that they are genuinely
different channels rather than two views of one.

## Leakage: the entire difficulty

Target encoding uses the target, so a careless version inflates CV and
collapses on the leaderboard. Two layers protect this one.

**Outer:** `run_cv` refits the whole pipeline per fold, so the encoder never
sees the outer validation rows at all.

**Inner:** `fit_transform` is overridden rather than inherited. It splits the
training rows again and encodes each inner fold from the *other* inner folds
only, so no row's encoding ever sees its own target. `transform` — used for
validation and test rows, which were not part of the fit — uses the full
statistics. Because scikit-learn's Pipeline calls `fit_transform` on
intermediate steps during `fit` and `transform` during prediction, the correct
behaviour happens automatically in both directions.

### Verified rather than asserted

| test | result |
|---|---|
| **permuted target** — encode against a shuffled `y` | te_ columns score **0.4958–0.5017**, mean 0.4996 ✅ |
| real target, in-sample | `te_daily_screen_time_hours` 0.8738 — high, and *should* be |
| flip one row's label, measure the change in its own encoding | 2.6e-2 at the flipped row against 1.97e-1 max elsewhere — no outsized self-influence ✅ |

The permuted-target test is the one that separates signal from leakage: a
leaking encoder would still score well above 0.5 against a shuffled label,
because it would be reading targets rather than levels.

## The pandas 3.0 trap, which would have hit us

This repo runs **pandas 3.0.5**. The obvious `df[c].astype(str)` is correct on
pandas 2.x — it writes the literal string `"nan"`, so missing values get their
own level — and **silently wrong on 3.0+**, where the new string dtype
preserves NA instead. `groupby` then drops every missing row from the level
statistics, `.map()` returns NA for them, and the encoding quietly vanishes on
4–20% of every column. Nothing raises and nothing warns.

`level_frame` uses `.astype(object).fillna("__missing__").astype(str)`, correct
on both, and `assert_full_coverage` checks `groupby(...).size().sum() ==
len(df)` rather than trusting it.

## Imputation is now nearly redundant

Imputation was worth **+0.00072** before encoding (run 019) and **+0.00013**
after it (run 022) — a 5.5× shrinkage, and now below the fold noise of
0.00046. The encoding reads each exact value's target statistics directly,
which is most of what the reconstructed values were providing. Kept, because
the interval still excludes zero and it costs little, but it is no longer
doing real work.

## Predictions, recorded before submitting

The honest leakage test is on held-out data: a leaky encoding shows up as
**CV/LB divergence**, not as a uniformly bad score. Using the two-component
model from Phases 5 and 7 (covariate shift measured here, plus the +0.001012
fold-averaging bonus measured for a single tuned model):

| run | OOF | covariate shift | predicted LB |
|---|---|---|---|
| `021` | 0.967130 | +0.000877 | **0.96902** |
| `022` | 0.967257 | +0.000906 | **0.96917** |

If either lands near its prediction, the encoding is clean. If the LB comes in
far below, the CV is inflated and the nesting has a hole.

### Outcome: 0.96905

Run `022` was submitted. **It scored 0.96905 against a prediction of
0.96917 — an error of −0.00012**, and a gap of −0.001793, squarely inside the
−0.00099 to −0.00216 range the other seven submissions have produced.

**There is no CV/LB divergence, so the encoding is clean on held-out data.**
That is the test that mattered: a leaky encoder inflates CV specifically, so
it shows up as the leaderboard falling far short of a gap-based prediction,
not as a uniformly poor score. The prediction was made from an offset
calibrated on runs that contain no target encoding at all, and it held.

**Rank 1,256 → 827 of 2,982.**

| run | OOF | public LB | gap | rank |
|---|---|---|---|---|
| `012` | 0.963192 | 0.96511 | −0.001918 | 1,678 |
| `020` | 0.964709 | 0.96610 | −0.001391 | 1,256 |
| `022` | **0.967257** | **0.96905** | −0.001793 | **827** |

## Reproducing

```bash
python -m src.experiment --model histgbm_encoded        # run 021
python -m src.experiment --model histgbm_encoded_full   # run 022
python -m src.compare 020 021
python -m src.compare 021 022
```

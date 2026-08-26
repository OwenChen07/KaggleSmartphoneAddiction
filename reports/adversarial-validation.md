# Phase 5 — Adversarial validation

**Result: train and test are separable, at AUC 0.565161 rather than 0.5 — and
the entire separation is in the *missingness pattern*, not in any feature
value. That turns out to explain the OOF-to-leaderboard gap this project has
been tracking since Phase 3.**

## The technique

Throw away the real target. Label every training row `0` and every test row
`1`, and try to predict which file a row came from. That is an ordinary binary
classification problem, so it runs through the ordinary CV harness and yields
an ordinary AUC.

- **AUC ≈ 0.5** — the classifier cannot tell the files apart. Train and test
  are exchangeable, so held-out training rows are a fair stand-in for test
  rows and the OOF estimate transfers.
- **AUC ≫ 0.5** — something distinguishes them. Held-out training rows are
  then *not* representative of what the model will be scored on, and the
  per-feature importances say where the difference lives.

The discriminator must be at least as expressive as the model whose OOF is
being defended — a weak discriminator finding nothing proves nothing — so it
is the same `HistGradientBoostingClassifier` used everywhere else.

`id` is excluded. It is a row counter that separates the two files perfectly
and would produce a meaningless AUC of 1.0.

**What this does not test:** whether `P(y | X)` is stable. Two files can have
identical feature distributions while the target relationship differs.
Adversarial validation is silent on that, because it never looks at `y`.

## The measurement

691,369 train rows (label 0) against 296,302 test rows (label 1); 987,671
combined, test share 0.3000.

| discriminator sees | adversarial AUC | fold std |
|---|---|---|
| everything (values + missingness) | **0.565161** | 0.001206 |
| the is-missing pattern only, values discarded | **0.565382** | 0.000794 |
| values, median-imputed | 0.561474 | 0.001356 |
| **complete cases only** (no missing value in any column) | **0.499160** | 0.002091 |

The missingness pattern **alone** reproduces the full separation (0.565382 vs
0.565161 — the missingness-only model is, if anything, marginally better).
Restricted to the 25.9% of rows with nothing missing anywhere, the
discriminator collapses to **0.499160**, indistinguishable from chance.

The median-imputed row is a **failed** attempt at a values-only control, kept
because the failure is instructive: median imputation does not erase
missingness, it *encodes* it. Every imputed cell lands on exactly the median,
so the discriminator can simply count the size of the spike there. It still
scored 0.561. Complete-case analysis is what actually removes the channel.

Direct inspection agrees. Two-sample KS tests on the nine numeric columns
(non-missing values only) are uniformly non-significant — statistics around
0.002, p-values from 0.14 to 0.997 — while every column's *missing rate*
differs:

| column | missing train % | missing test % | diff |
|---|---|---|---|
| `social_media_hours` | 19.381 | 15.996 | **−3.385** |
| `app_opens_per_day` | 11.674 | 8.675 | **−2.999** |
| `daily_screen_time_hours` | 13.864 | 11.066 | **−2.799** |
| `academic_work_impact` | 6.397 | 8.681 | **+2.284** |
| `work_study_hours` | 7.452 | 9.375 | +1.923 |
| `notifications_per_day` | 9.775 | 11.549 | +1.774 |
| `gaming_hours` | 18.343 | 20.054 | +1.710 |
| `age` | 4.184 | 5.784 | +1.600 |
| `stress_level` | 7.977 | 6.624 | −1.353 |
| `sleep_hours` | 6.434 | 7.578 | +1.145 |
| `weekend_screen_time` | 16.209 | 17.110 | +0.901 |
| `gender` | 4.199 | 4.796 | +0.597 |

The synthetic generator drew both files from the same underlying distribution
and then masked cells at different rates.

## Why this does not invalidate OOF, and what it does instead

The instinctive reaction to AUC 0.565 is "the split is dirty, stop trusting
OOF." That is the wrong conclusion here, for a reason Phase 3 already
established: **missingness carries no target signal in this data.** A model
trained on nothing but the 12 is-missing flags scores 0.50038, and target rates
between missing and present rows differ by at most 0.0042 against a base rate
of 0.7094.

So the shift is real but lives in a channel that is uninformative about `y`.
`P(y | X)` is unchanged — the values are identically distributed — and a model
fit on training rows remains correct for test rows.

What *does* change is the **mix of easy and hard rows**. Test has more rows
with the strongest predictors present, and rows with those columns present are
much easier to classify:

| missingness of the 3 strongest features | train % | test % | diff | OOF AUC in stratum | n |
|---|---|---|---|---|---|
| none missing | 66.14 | 68.69 | **+2.55** | **0.96314** | 457,274 |
| `weekend_screen_time` missing | 6.51 | 8.27 | +1.76 | 0.95268 | 44,986 |
| `daily_screen_time` missing | 5.19 | 4.44 | −0.75 | 0.95370 | 35,883 |
| `social_media_hours` missing | 8.77 | 7.39 | −1.38 | 0.94576 | 60,632 |
| `daily` + `weekend` missing | 2.78 | 2.61 | −0.18 | 0.91454 | 19,231 |
| `daily` + `social` missing | 3.69 | 2.38 | −1.31 | 0.92119 | 25,517 |
| `weekend` + `social` missing | 4.72 | 4.59 | −0.13 | 0.93722 | 32,623 |
| **all three missing** | 2.20 | 1.64 | **−0.56** | **0.80469** | 15,223 |

Complete rows score 0.963; rows missing all three score 0.805. **The test set
is simply easier than the training set**, because it has 2.55 points more of
the former and 0.56 points less of the latter.

### Quantifying it

Standard covariate-shift reweighting: the adversarial classifier's own OOF
probability gives the density ratio `p_test(x) / p_train(x) = p / (1 − p)` for
each training row. Resampling the OOF predictions under those weights
estimates what the model would score on a population distributed like the test
set. Weights are well behaved — trimmed at the 99.5th percentile, ranging
0.31× to 1.88×, with an effective sample size of 654,814 of 691,369 — so this
is not a degenerate reweighting driven by a handful of rows.

| run | measured OOF | reweighted to test | unweighted control | predicted shift |
|---|---|---|---|---|
| `005` | 0.954704 | 0.955959 (sd 0.000222) | 0.954715 (sd 0.000215) | **+0.001244** |
| `008` | 0.956091 | 0.957271 (sd 0.000190) | 0.956072 (sd 0.000221) | **+0.001199** |

A separate estimate using only the 3-feature missingness pattern above gives
**+0.001303**, so the number is not an artifact of the weighting method.

**The observed OOF-to-LB gap for run `005` is +0.001076** (LB 0.95578 against
OOF 0.954704). The covariate shift predicts +0.001244. Sign correct, magnitude
correct to within 0.00017.

### This revises the explanation in the README

The README previously attributed the negative gap to fold-averaging: test
predictions are the mean of 5 fold models while each OOF row is scored by one
model, and averaging reduces variance.

That effect is real, and was measured directly here. Splitting the training
set into 80% (`A`) and 20% (`B`) from the same population — so there is no
distribution shift — and comparing on the identical rows of `B`:

| | AUC on `B` |
|---|---|
| single fold-1 model | 0.953864 |
| 5-model average | 0.954453 |
| **ensemble-averaging effect (paired, same rows)** | **+0.000590** |

So fold-averaging is worth about +0.0006 and covariate shift about +0.0012,
against an observed gap of +0.00108. **They do not add up** — the sum would be
+0.0018, which over-predicts. At most one of them can be close to the whole
story, and this analysis does not resolve which.

Honest limits on the decomposition:

- The public leaderboard is scored on an unknown subset of the test rows, with
  its own missingness mix, not on all 296,302.
- AUC is a global rank statistic and is not linear in the row mix, so two
  effects estimated separately have no reason to sum.
- The same-population nested experiment compares an OOF AUC over 553,095 rows
  with a test AUC over 138,274, which is unpaired and carries its own sampling
  noise.

What is solid: the gap is **not** a symptom of leakage or overfitting, the
failure mode gap-tracking exists to catch. It is a property of how the two
files were masked, it is predictable in advance from the training data alone,
and its predicted size matches what the leaderboard actually returned.

### A forward prediction

Covariate-shift reweighting predicted run `008` (OOF 0.956091) should score
about **0.95729** on the leaderboard from the shift alone, or **0.95717** if
the historical gap of −0.001076 simply held. Both were written down before
submitting.

**It scored 0.95708** — a gap of **−0.000989**.

| prediction | value | error |
|---|---|---|
| covariate-shift reweighting | 0.95729 | +0.00021 |
| historical gap holds | 0.95717 | +0.00009 |
| **actual** | **0.95708** | — |

Both land within 0.0002, and both slightly over-predict. The "assume the gap
is stable" prediction was marginally the better one, which is a mild point
against the reweighting estimate being the complete mechanism — though with a
single observation and a sampling noise of roughly 0.0007 on one leaderboard
AUC, the two predictions are not distinguishable from each other.

The gap across three submissions is now −0.001083, −0.001076 and −0.000989:
still tightly clustered, still negative, and still with no sign of the
OOF-overstates-performance pattern that would indicate leakage.

## Reproducing

```bash
python -m src.adversarial                              # full, with importances
python -m src.adversarial --mode missingness --no-importance
python -m src.adversarial --mode complete --no-importance
python -m src.adversarial --mode values --no-importance   # the failed control
```

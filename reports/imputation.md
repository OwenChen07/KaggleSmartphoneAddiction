# Phase 8 — Reconstructing the missing values

**Result: imputation helps, but only when the reconstructions are added
*alongside* the originals. Replacing the missing values with them makes the
model measurably worse.**

| run | model | OOF AUC | fold std | vs run `012` |
|---|---|---|---|---|
| `012` | tuned, no imputation | 0.963192 | 0.000474 | — |
| `018` | tuned + imputed, **replace** | 0.961674 | 0.000579 | **−0.00152** `[-0.00164, -0.00139]` |
| `019` | tuned + imputed, **augment** | **0.963913** | 0.000571 | **+0.00072** `[+0.00064, +0.00081]` |

Both intervals exclude zero. Same change, opposite signs, depending only on
whether the original NaN is kept.

---

## Why this was worth trying

Phase 5 measured OOF AUC by missingness stratum and found an enormous spread:
rows with all three top features present score **0.963**, rows missing all
three score **0.805**. If those absent values could be reconstructed from the
columns that *are* present, the hard rows get easier.

The competing explanation was that the 0.805 stratum is simply at its
information ceiling — those rows have less information, full stop, and no
method invents more. **Imputation cannot create information.** So the first
job was to find out which it was.

## Stage 0 — the gate

For each numeric column, train a regressor to predict it from the other 11
features, using only the 38.9% of training rows with nothing missing. Measure
R² on held-out complete rows. The decision rule was fixed in advance: stop if
the top-3 features all came back under 0.20.

| column | R² | univariate AUC on the target |
|---|---|---|
| **`daily_screen_time_hours`** | **0.8716** | 0.890 |
| `weekend_screen_time` | 0.6648 | 0.881 |
| `social_media_hours` | 0.6068 | 0.858 |
| `work_study_hours` | 0.5448 | 0.655 |
| `gaming_hours` | 0.5013 | 0.622 |
| `sleep_hours` | 0.0245 | 0.527 |
| `notifications_per_day` | 0.0228 | 0.492 |
| `app_opens_per_day` | 0.0198 | 0.541 |
| `age` | 0.0038 | 0.502 |

The columns split into two clean groups, and the correlation structure says
why: the five screen-time columns are mutually predictive (`daily` vs
`weekend` **r = +0.80**, `daily` vs `social` +0.60, `daily` vs `work` +0.52),
while age, sleep, notifications and app opens are mutually independent — the
largest |r| among all remaining pairs is **0.086**.

Two things worth noticing:

- The strongest single predictor of the target is **87% recoverable**, and it
  is missing in 13.9% of rows.
- **The recoverable columns are exactly the predictive ones.** The four
  unrecoverable columns are the four with near-chance univariate AUC. Nothing
  guaranteed that; it just happened to be true here.

**Only the five columns above R² 0.50 are imputed.** Filling `age` or
`notifications_per_day` would replace an honest NaN — which HistGBM routes
deliberately — with a fabricated constant it cannot tell apart from a real
measurement. That is strictly worse than leaving the hole.

### A caveat that survived into the result

This R² is measured with all 11 other columns present. Rows that actually need
imputation frequently miss several at once, so real performance is worse than
these numbers. Stage 0 is an optimistic upper bound by construction.

## Stage 1 — the transformer

`RecoverableImputer` fits one `HistGradientBoostingRegressor` per target
column, trained on the rows where that column is present, predicting it from
the other 11 raw features. The predictors themselves may be missing; HistGBM
routes NaN natively, so no cascading imputation is needed.

**Unlike `EngineeredFeatures`, this transformer is stateful**, which makes fold
discipline load-bearing rather than decorative. Fitted on the full training set
before splitting, it would leak validation rows into the imputations used to
score those same rows, and every OOF number downstream would be fiction. It is
therefore a step *inside* the Pipeline, cloned and refit per fold.

It also runs **first**, ahead of `EngineeredFeatures`, so the ratios are built
from reconstructed values. Under `replace` that is much of the point: a ratio
inherits the missingness of both inputs, so the engineered features were
previously missing for 17–36% of rows.

Training on present-rows and applying to missing-rows assumes the two groups
are comparable. Phase 3 established exactly that — a model trained on nothing
but the is-missing flags scores 0.50038 — so the regressor is not extrapolating.

## Why replacing makes it worse

This is the interesting result. Comparing the imputed values against the real
distribution on a validation fold:

| column | real sd | imputed sd | ratio | values outside real 5th–95th pct |
|---|---|---|---|---|
| `daily_screen_time_hours` | 2.721 | 2.314 | 0.85 | 3.2% |
| `weekend_screen_time` | 2.857 | 2.156 | 0.75 | **0.0%** |
| `social_media_hours` | 1.316 | 0.859 | 0.65 | 1.4% |
| `work_study_hours` | 1.260 | 0.806 | 0.64 | 1.7% |
| `gaming_hours` | 0.935 | 0.497 | 0.53 | 0.2% |

*(by construction, 10% of real values fall outside that band)*

The means are almost exactly right — 7.657 against 7.637 for daily screen
time. The **variance is crushed**, and the tails are close to empty.
`weekend_screen_time` imputations produce **not one** value beyond the real
5th/95th percentiles out of tens of thousands of rows.

That is ordinary **regression to the mean**: a conditional-mean estimate is
necessarily less variable than the thing it estimates, and the shrinkage gets
worse as R² falls — which is exactly the ordering in the table (0.85 at
R² 0.87, down to 0.53 at R² 0.50).

Replacement therefore feeds the model mid-range fabrications that are
indistinguishable, to it, from real measurements. Screen time is the strongest
predictor and its **extremes** carry the signal, so flattening the tails
destroys exactly the part that mattered. Worse, the model loses the NaN marker
and can no longer route unknown rows deliberately — it is told a confident
value where it previously knew it had none.

Augmenting avoids both problems. The original NaN survives, so native routing
still works and the model still knows the value is unknown; the reconstruction
arrives as an extra column it can weigh as it likes. **It gets a hint instead
of a lie.**

## Reproducing

```bash
python -m src.experiment --model histgbm_imputed        # run 018, replace
python -m src.experiment --model histgbm_imputed_aug    # run 019, augment
python -m src.compare 012 018
python -m src.compare 012 019
```

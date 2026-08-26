# Phase 4 — Feature engineering

**Result: the first real improvement in this project.** OOF AUC 0.954704 →
**0.956091**, a difference of **+0.001387** with a 95% paired-bootstrap CI of
**[+0.00129, +0.00148]**. That interval excludes zero and, unlike the
categorical-features result from Phase 3, the effect is also larger than the
0.0008 fold-to-fold noise floor — so it is both statistically real and big
enough to care about.

Runs: `008` (all 15 engineered features) and `009` (pruned to 5).

---

## The design constraint that shaped the feature list

The three screen-time columns already carry almost all the univariate signal
(`daily_screen_time_hours` 0.890, `weekend_screen_time` 0.881,
`social_media_hours` 0.858). A gradient-boosted tree can split on each of them
directly. So the question is not "what is predictive?" — it is "what can a
tree not already reach by splitting on the raw columns?"

That question rules out more than it first appears.

**A tree splits on `x <= t`. Any strictly monotonic function of a single column
gives the identical partition of the rows**, just with relabelled thresholds.
So `8 - sleep_hours` ("sleep deficit") and `24 - waking_load` ("free hours")
cannot add anything to a tree, however meaningful they are to a human reading
a column list. Reciprocals are monotonic too on positive data, so
`daily_screen / app_opens` and `app_opens / daily_screen` are the *same
feature* to a tree; only one of each such pair is worth including.

**What a tree genuinely cannot reach cheaply is a ratio of two columns.** To
approximate the boundary `a / b <= t` with axis-aligned splits, it has to
stair-step: split on `b`, then on `a` within each `b` range, then again, each
step spending depth and dividing the rows available to fit the next. Handing
the model `a / b` directly collapses that staircase into a single split. Every
feature below is a ratio, difference, or product of **two or more** columns —
except one deliberate control.

### The 15 candidates

| feature | definition | rationale |
|---|---|---|
| `social_share` | `social_media_hours / daily_screen_time_hours` | composition, not level |
| `gaming_share` | `gaming_hours / daily_screen_time_hours` | composition |
| `work_share` | `work_study_hours / daily_screen_time_hours` | composition |
| `weekend_ratio` | `weekend_screen_time / daily_screen_time_hours` | weekend vs weekday, normalised |
| `weekend_gap` | `weekend_screen_time - daily_screen_time_hours` | same contrast, unnormalised |
| `session_length` | `daily_screen_time_hours / app_opens_per_day` | hours per app open |
| `notif_per_open` | `notifications_per_day / app_opens_per_day` | prompting per visit |
| `notif_per_screen_hour` | `notifications_per_day / daily_screen_time_hours` | prompting per hour |
| `tracked_screen` | `social_media_hours + gaming_hours` | itemised total |
| `residual_screen` | `daily_screen - social_media - gaming` | screen time the items don't explain |
| `waking_load` | `daily_screen + sleep + work_study` | time budget against 24h |
| `screen_sleep_ratio` | `daily_screen_time_hours / sleep_hours` | screen against rest |
| `age_x_screen` | `age * daily_screen_time_hours` | age interaction, product |
| `screen_per_age` | `daily_screen_time_hours / age` | age interaction, ratio |
| `sleep_deficit` | `8 - sleep_hours` | **control — predicted worthless** |

`sleep_deficit` is included precisely because the argument above says it cannot
help. It is a prediction made before running anything, and checking it tests
the *measurement*, not the feature. What actually happened to it is the most
interesting result in this phase — see below.

### Division by zero and missingness

`_safe_divide` converts any non-finite quotient to NaN rather than letting an
inf reach the bin mapper. On this data the guard never fires: the minima are
0.5 hours of daily screen time, 15 app opens and age 18, so no denominator
reaches zero. It is there because "never reaches zero" is a fact about the
*training* data, and an inf arriving silently would be a corruption rather
than a failure.

Missing inputs propagate. Because raw missing rates run 4–19% per column, a
two-column ratio is missing for **17–36%** of rows:

| feature | missing % | feature | missing % |
|---|---|---|---|
| `residual_screen` | 35.5 | `notif_per_screen_hour` | 21.9 |
| `tracked_screen` | 30.0 | `work_share` | 21.0 |
| `social_share` | 27.7 | `screen_sleep_ratio` | 19.0 |
| `gaming_share` | 27.6 | `age_x_screen` | 18.2 |
| `weekend_ratio` / `weekend_gap` | 25.9 | `screen_per_age` | 18.2 |
| `waking_load` | 25.2 | `notif_per_open` | 17.4 |
| `session_length` | 23.8 | `sleep_deficit` | 5.8 |

They are left NaN. Imputing would invent a value for a third of a column;
HistGradientBoosting instead learns a default routing direction for NaN at
every split, which is strictly more information than a filled-in median.

### Where the transformer sits

`EngineeredFeatures` is a proper sklearn transformer placed as the **first step
inside the Pipeline**, ahead of the ColumnTransformer, so it is cloned and fit
per fold like everything else.

It is also completely **stateless** — row-wise arithmetic, nothing learned from
the data it is fit on, `fit` has genuinely nothing to do. That is the strongest
possible position on leakage: there is no fitted quantity that *could* carry
information across a fold boundary. It still goes inside the pipeline, because
the rule ("all preprocessing inside the fold") should hold by construction
rather than by an argument that has to be re-checked the day a feature does
need to learn something.

---

## Permutation importance, on held-out rows

Fold 1, full 691,369 rows, 5 repeats, `scoring="roc_auc"`, permuted on the
**validation** rows. Engineered features in **bold**.

| feature | drop in AUC | std | | feature | drop in AUC | std |
|---|---|---|---|---|---|---|
| `daily_screen_time_hours` | 0.149422 | 0.000878 | | **`waking_load`** | 0.000401 | 0.000099 |
| `weekend_screen_time` | 0.099896 | 0.001073 | | **`sleep_deficit`** | 0.000335 | 0.000028 |
| `social_media_hours` | 0.054065 | 0.000361 | | **`screen_per_age`** | 0.000208 | 0.000025 |
| `app_opens_per_day` | 0.010986 | 0.000169 | | **`tracked_screen`** | 0.000150 | 0.000019 |
| `notifications_per_day` | 0.009776 | 0.000171 | | **`age_x_screen`** | 0.000122 | 0.000025 |
| **`social_share`** | 0.007697 | 0.000152 | | **`notif_per_screen_hour`** | 0.000040 | 0.000018 |
| **`work_share`** | 0.005976 | 0.000107 | | **`screen_sleep_ratio`** | 0.000029 | 0.000017 |
| **`gaming_share`** | 0.004605 | 0.000124 | | **`weekend_gap`** | 0.000022 | 0.000007 |
| **`residual_screen`** | 0.004070 | 0.000121 | | **`session_length`** | 0.000002 | 0.000003 |
| `work_study_hours` | 0.002916 | 0.000065 | | `academic_work_impact` | 0.000000 | 0.000000 |
| **`weekend_ratio`** | 0.002171 | 0.000083 | | `gender` | 0.000000 | 0.000000 |
| `gaming_hours` | 0.000592 | 0.000050 | | `stress_level` | −0.000002 | 0.000002 |
| `age` | 0.000581 | 0.000012 | | **`notif_per_open`** | −0.000003 | 0.000003 |
| `sleep_hours` | 0.000480 | 0.000033 | | | | |

**Survived** — clearly above their own repeat-to-repeat spread, and above two
of the nine raw columns: `social_share`, `work_share`, `gaming_share`,
`residual_screen`, `weekend_ratio`. The three activity *shares* are the win.
Every one of them is a composition ratio, which is exactly what the design
argument predicted would be hardest for a tree to reach on its own.

**Did not survive** — `session_length`, `notif_per_open`,
`notif_per_screen_hour`, `screen_sleep_ratio`, `weekend_gap` all sit at or
below 4e-5, essentially indistinguishable from zero.

`weekend_gap` failing while `weekend_ratio` succeeds (0.002171, ~100× larger)
is the sharpest single illustration of the design argument. Both encode
"weekend versus weekday". The *difference* is reachable by a tree from the two
raw columns with a couple of splits; the *ratio* is the one that needs the
staircase, and it is the one that pays.

`session_length` at 2e-6 is the cleanest failure: hours-per-app-open simply
does not carry information about addiction here beyond what the two source
columns already give.

---

## The control feature did not behave as predicted, and the reason matters

`sleep_deficit = 8 - sleep_hours` is provably information-free to a tree. It
scored **0.000335** — small, but 12× its own std of 0.000028, and *larger* than
`tracked_screen`, `age_x_screen` and four other genuine two-column features.

The prediction was not wrong about the feature. It was wrong about what
permutation importance measures. Counting actual split usage in the fitted
fold-1 ensemble:

| feature | # splits used |
|---|---|
| `sleep_hours` | 67 |
| `sleep_deficit` | 35 |

The two columns are perfectly redundant, so at each split the model picks one
essentially arbitrarily, and the 102 sleep-related splits get **divided
between them**. Permuting `sleep_deficit` then destroys the 35 splits that
happen to reference it, and `sleep_hours` does not step in to compensate —
the fitted trees point at specific columns, not at "the sleep information".

So permutation importance answers **"how much does *this fitted model* rely on
*this specific column*?"** — not "how much is this information worth?" With
correlated or redundant features, a group's importance gets split across its
members, and each member individually understates the group. This is the
standard correlated-features caveat, and here it is visible on a feature we
constructed to be exactly redundant.

Practical consequence: **permutation importance is a ranking aid, not a
pruning rule.** Which the next experiment confirms.

---

## Pruning to the survivors made it worse

Run `009` keeps only the 5 survivors (17 model features vs 27).

| run | features | OOF AUC | fold std |
|---|---|---|---|
| `008` | all 15 engineered | **0.956091** | 0.00068 |
| `009` | 5 survivors only | 0.956023 | 0.000781 |
| `005` | no engineered features | 0.954704 | 0.000782 |

```
compare 008 009 :  difference -0.00007   95% CI [-0.00011, -0.00002]  -> B is WORSE
compare 005 009 :  difference +0.00132   95% CI [+0.00123, +0.00141]  -> B is better
```

Dropping the ten near-zero features cost **0.00007 AUC**, and the interval
excludes zero, so the loss is statistically real. Individually each of those
ten was indistinguishable from noise; **collectively they were worth a small
but measurable amount.** Permutation importance measures each feature's
marginal contribution *with all the others still present*, so a group of weak,
partly-redundant features can each measure ~0 while the group is not ~0.

And then the second half of the lesson: 0.00007 is **11× below the 0.0008 fold
noise**. It is statistically real and practically irrelevant. This project has
now produced that pattern twice in opposite directions — the categorical
features (+0.00003, significant, useless) and this prune (−0.00007,
significant, useless).

**Decision: keep all 15.** Not because the pruning result is important, but
because there is no reason to pay even a tiny measurable cost for dropping
columns that cost 4 seconds of fit time.

---

## Permuting on training rows instead: no difference here, and why

The same measurement on the **training** rows of fold 1 (3 repeats) produces
essentially the same table — same ordering in the top 11, values matching to
about 3 decimal places (`daily_screen_time_hours` 0.149372 vs 0.149422,
`social_share` 0.007963 vs 0.007697).

That is an honest null result, and it has a cause: at 691k rows with
`max_leaf_nodes=31` and 100 iterations, this model barely overfits at all, so
"what it used" and "what generalises" are the same thing.

To show that the distinction is real when a model *does* overfit, a column of
pure U(0,1) noise — true importance exactly zero by construction — was added
to a deliberately overfitted model (20,000 rows, `max_leaf_nodes=512`):

| permuted on | `noise_uniform` importance | rank |
|---|---|---|
| **training rows** | **+0.001356** | 10th of 13 — *above* `gender`, `academic_work_impact` and `stress_level`* |
| **held-out rows** | **−0.000431** | 13th of 13, last |

On training rows a provably worthless column outranks three genuinely (if
weakly) informative ones, because breaking it breaks the memorisation the
model built on it. On held-out rows it correctly lands at zero. That is the
failure mode the "always measure on held-out rows" rule exists to prevent —
it just does not happen to bite this particular well-regularised model.

---

## Reproducing

```bash
python -m src.experiment --model histgbm_fe             # run 008
python -m src.experiment --model histgbm_fe_pruned      # run 009
python -m src.compare 005 008
python -m src.importance --model histgbm_fe --repeats 5 --n-jobs 2
python -m src.importance --model histgbm_fe --on train --repeats 3 --n-jobs 2
python -m src.importance --model histgbm --sample 20000 --max-leaf-nodes 512 \
       --noise-col --on train --repeats 5
```

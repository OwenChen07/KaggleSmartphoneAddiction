# Research: what the rest of the leaderboard found

Date: 2026-08-26. Sources: the public notebook library for
`playground-series-s6e8`, read via the Kaggle API. **Every claim below that
this project acts on was re-verified on our own `data/train.csv` before being
written down** — the verification numbers are ours, not quoted.

---

## 1. The strategic finding: the plateau is public-LB overfitting

Our position (rank 1,256, LB 0.96610) sits ~0.005 below a cluster of ~500
teams at ≈0.9712. The obvious reading is that they know something. The
partially-correct reading is that **the top of this leaderboard is being
selected on noise.**

The highest-signal source is Rayk Kretzschmar's
[*Why every S6E8 notebook above 0.97110 probably overfits*](https://www.kaggle.com/code/raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits)
(64 votes), which is unusual in that it diagnoses **its own 0.97115
submission** as overfit and declines to select it. Three arguments:

1. **The leaderboard rewards the wrong sign.** An honest OOF signal (a
   distilled student model) improves the teacher on all 15 validation slices;
   the public LB prefers *subtracting* it. Audited against submission IDs, not
   recalled.
2. **A miniature leaderboard in OOF.** Repeatedly exposing 59,260 OOF rows
   (the public set is ≈20% of 296,302 test rows) as a pseudo-public board and
   selecting the best weight produces a gain on the selected split and a loss
   on the untouched rows.
3. **Season 6 already ran the experiment.** Across seven finished episodes,
   **S6E2, S6E6 and S6E7 each had zero public-top-10 teams remain in the
   private top 10.** In S6E7 the public winner finished **private rank 440**.

**What this means for us.** Chasing the last 0.002 of that plateau is
optimising a sample of ~59k rows. The disciplined OOF-plus-paired-bootstrap
method this repo is built on is the right instrument for the private split —
we should keep using it and stop treating 0.9712 as the target. It does *not*
mean the gap is illusory: most of the 0.005 below the plateau is real, and §2
is where it lives.

---

## 2. The real gap: our representation, not our model

This is the finding that matters, and it explains a puzzle in our own EDA.

`reports/eda.md` records `notifications_per_day` at univariate AUC **0.492**
and `app_opens_per_day` at **0.541** — both indistinguishable from chance —
and every phase since has treated them as near-worthless numeric quantities.

**They are not quantities. The generator uses the exact value as a key into a
lookup table.** Verified on our data:

| column | levels | rows/level | per-value target-rate sd | sd from sampling alone | ratio | mean \|rate diff\| between adjacent values |
|---|---|---|---|---|---|---|
| `notifications_per_day` | 231 | 2,700 | 0.1919 | 0.0087 | **22.0×** | **0.2248** |
| `app_opens_per_day` | 166 | 3,679 | 0.1952 | 0.0075 | **26.1×** | **0.2170** |
| `age` | 18 | 36,802 | 0.0397 | 0.0024 | 16.8× | 0.0590 |
| `daily_screen_time_hours` | 1,108 | 535 | 0.3259 | 0.0196 | 16.6× | 0.0713 |

Adjacent integer values differ in addiction rate by **0.22 on average**, with
~2,700 rows behind each. That is 22× larger than sampling noise can produce:
87 notifications and 88 notifications have nothing to do with each other.

The consequence, measured on our data with a nested fold-safe target encoder:

| the same two columns | OOF AUC |
|---|---|
| as raw numbers | **0.5122** |
| as target-encoded lookup keys | **0.8170** |

**Chance to 0.82, from columns we have been feeding the model as magnitudes
for eight phases.** No tree can recover this: a split on `x <= t` asks an
ordering question of a variable that has no meaningful order. Target encoding
reads the lookup directly.

tomasa2's [*What Moved the Score, and What Didn't*](https://www.kaggle.com/code/tomasa2/s6e8-what-moved-the-score-and-what-didn-t)
reports target-encoding **every** column — continuous ones included — as
"larger than everything else in this notebook combined", which matches: it
works because `daily_screen_time_hours` has 1,389 distinct values over 691k
rows, ≈500 rows per level, so a smoothed target mean is well estimated at
every one.

---

## 3. The decimal lattice

A second channel, independent of the lookup structure. Verified on our data —
the first decimal digit of a column predicts the target:

| first decimal of `daily_screen_time_hours` | addiction rate | rows |
|---|---|---|
| .0 | **0.6513** | 61,155 |
| .2 | **0.7365** | 65,419 |
| .4 | 0.6721 | 51,975 |
| .9 | 0.7326 | 51,060 |

**An 8.5-point swing** against a base rate of 0.7094. Per-column spreads:

| column | digit-rate spread |
|---|---|
| `weekend_screen_time` | **0.1047** |
| `daily_screen_time_hours` | 0.0852 |
| `sleep_hours` | 0.0677 |
| `social_media_hours` | 0.0443 |

No behavioural story explains this — it is a fingerprint of how the generator
produced the numbers. Crucially it is a **different channel from target
encoding**, which estimates every exact value independently and so has no way
to express "everything ending in .2 shares something".

---

## 4. The accounting identity

Verified: `daily_screen_time_hours ≥ social_media + gaming + work_study`, with
**36 violations in 421,427 rows (0.01%)**, all at −1e-6, i.e. float rounding.
Effectively exact.

Our Phase 4 `residual_screen` feature computed `daily − social − gaming` and
missed `work_study`, so it never expressed the identity. The complete residual
is the generator's "other" bucket and is a genuine quantity.

The same identity is violated in **60.7% of the real source dataset** — the
generator manufactured it. Also worth noting: `work_study_hours` and
`gaming_hours` have AUC ≈0.50 in the real data and are predictive here, which
independently explains why our external-data experiment was worthless.

---

## 5. What our project already got right

Four findings we derived independently were confirmed by the public work,
which is a useful check on the harness:

| our finding | independent confirmation |
|---|---|
| imputation must **augment, not replace** (`+0.00072` vs `−0.00152`) | §5: "same imputer, opposite sign… for a NaN-native model an imputed column should be an extra feature, never a substitute" |
| the CV→LB offset **tracks model variance**, not the pipeline | §9: a smaller library has a *larger* offset "because it has done less variance-averaging of its own, so more is left for the 5-fold test averaging to recover" |
| blending pays only among **comparably strong** models | §8.2: contribution tracks solo OOF, not decorrelation; a visible cliff below which members contribute nothing however different |
| the original source dataset is worthless as training data | §4: −0.00008; §8.4: kNN into it scores 0.9036 standalone and −0.000029 incremental |

---

## 6. A trap that hits us specifically

**We run pandas 3.0.5.** On pandas ≥3.0 the new string dtype **preserves NA
instead of writing `"nan"`**, so `df[c].astype(str)` no longer gives missing
values their own level. `groupby` silently drops every missing row from the
level statistics and `.map()` returns NA for them. On 4–20% of every column
the encoding quietly disappears. Nothing errors, nothing warns.

The safe form, correct on both versions:

```python
df[c].astype(object).fillna("__missing__").astype(str)
```

and assert `groupby(...).size().sum() == len(df)` rather than trusting it. The
same change also breaks `dtype == object` as a categorical test.

---

## 7. What to do, in order

1. **Target-encode every column as a high-cardinality categorical**, including
   the continuous ones — nested fold-safe (outer-fold statistics, inner OOF
   encodings), smoothing ≈10, missing as an explicit level. This is the +0.005.
   Nine variations of it (pairwise, multi-resolution, other smoothings) were
   all measured as negative; plain single-feature TE at smoothing 10 is the
   sweet spot.
2. **Hand raw string levels to CatBoost as `cat_features`** and let its
   ordered target statistics do the job. Reported as beating a hand-rolled
   encoder, for one argument.
3. **Decimal lattice features** — `frac(x)` and `floor(10x) % 10` for the six
   fractional columns.
4. **Fix `residual_screen`** to the full identity, `daily − social − gaming −
   work_study`.
5. **Transductive frequency encoding** — count each value's occurrences across
   train **and** test (features only; no labels, so it is legitimate). Worth
   ≈+0.0003 to a single model, ~nothing inside a dense ensemble.
6. **Logit stacking over rank-averaging** if a library gets built — a stacker
   can assign negative weights; hill climbing cannot.

Explicitly **not** worth doing, all measured by others and consistent with our
own nulls: more hyperparameter search, more architectures below ≈0.966 solo,
autoencoder features, kNN features into the source dataset, importance-screen
top features.

---

## 8. Honest framing

Adopting §7.1 alone should move us most of the way to ≈0.970. The remaining
distance to 0.9712 is substantially assembled from other competitors'
published prediction files (tomasa2 puts that at ≈0.0007) and from public-LB
selection, which §1 argues against chasing.

The realistic goal is **≈0.970 on a self-contained pipeline**, which by the
current board is around rank 700–900 — and, if §1 is right, a better private
position than several teams currently above us.

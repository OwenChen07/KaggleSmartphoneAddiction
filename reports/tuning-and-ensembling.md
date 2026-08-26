# Phase 6 — Tuning and ensembling

**Tuning was expected to buy little. It bought +0.0071 — about 15× the fold
noise, and by far the largest gain in the project. Ensembling then bought
+0.00028, which is real but smaller than the noise it sits in.**

| run | model | OOF AUC | fold std | fit (s) | public LB |
|---|---|---|---|---|---|
| `008` | Phase 4 defaults | 0.956091 | 0.000680 | 26 | 0.95708 |
| **`012`** | **tuned** | **0.963192** | 0.000474 | 65 | **0.96511** |
| `013` | search runner-up #2 | 0.962583 | 0.000513 | 128 | — |
| `014` | search runner-up #3 | 0.962955 | 0.000511 | 69 | — |
| — | rank-average `012`+`014` | **0.963472** | — | — | not submitted |

```
compare 008 012 :  difference +0.00710   95% CI [+0.00694, +0.00726]  -> better
```

## The search

`RandomizedSearchCV`, 40 candidates, `scoring="roc_auc"`, `refit=False`.

**The search ran with `cv=3` on a 250,000-row subsample, not on the full
691,369 rows.** A 40-candidate search at 5-fold on all rows costs roughly 40×
a full run; at 3-fold on 250k it took 9 minutes. That subsample is a real cost,
not a free saving — at 250k rows the model is weaker overall, so the search
optimises a slightly different problem than the one it selects for. This is
why the winner is **re-measured** on the standard full-data 5-fold `run_cv`
rather than trusted: the number in the log comes from the same folds, at the
same size, as every other row, and stays paired with them. The subsampled
score (0.959422) is **not** comparable to anything in `experiments/log.csv`.

| parameter | range | why |
|---|---|---|
| `learning_rate` | log-uniform 0.01–0.3 | the dominant knob; interacts with `max_iter` |
| `max_iter` | 100–600 | low learning rates need more trees |
| `max_leaf_nodes` | 15–127 | tree capacity; default 31 is the thing to challenge |
| `max_depth` | None, 6, 8, 12 | `None` lets leaf count govern shape alone |
| `min_samples_leaf` | 20–500 | default 20 is very permissive at 691k rows |
| `l2_regularization` | log-uniform 1e-3–10 | shrinks leaf values |

Winner: `learning_rate=0.1445`, `max_iter=463`, `max_leaf_nodes=69`,
`max_depth=8`, `min_samples_leaf=263`, `l2_regularization=0.0667`.

## Why the expectation was wrong

The prior — "tuning buys little" — is usually right, and it is right when the
defaults are already in the neighbourhood of correct. They were not. **The
sklearn defaults are tuned for datasets far smaller than 691,369 rows.**
`max_leaf_nodes=31` and `max_iter=100` were badly underfitting; there was
capacity being left on the table, not variance to be squeezed out.

One-at-a-time ablation, each change applied alone to the Phase 4 defaults:

| single change | OOF AUC | Δ |
|---|---|---|
| `max_leaf_nodes` 31 → 69 | 0.960117 | **+0.004026** |
| `max_iter` 100 → 463 | 0.957911 | +0.001821 |
| `learning_rate` 0.1 → 0.1445 | 0.956904 | +0.000814 |
| `min_samples_leaf` 20 → 263 | 0.956154 | +0.000063 |
| `l2_regularization` 0 → 0.0667 | 0.956067 | −0.000023 |
| `max_depth` None → 8 | 0.953452 | **−0.002638** |
| **all six together (run `012`)** | **0.963192** | **+0.007101** |

Two things worth reading carefully:

**The parts do not sum to the whole.** The individual deltas add to +0.0041;
together they deliver +0.0071. Hyperparameters interact, so one-at-a-time
tuning would have stopped well short of what a joint search found. That is the
argument for randomised search over hand-tuning one knob at a time.

**`max_depth=8` makes things worse on its own (−0.0026) and is part of the
winning combination.** Alone, capping depth at 8 while `max_leaf_nodes` is
still 31 only removes reachable tree shapes. Combined with
`max_leaf_nodes=69`, it stops the larger trees becoming deep and narrow. A
parameter's sign is not a property of the parameter — it depends on the rest
of the configuration.

## Ensembling: rank-average, and why

ROC AUC depends only on the **ordering** of predictions, so any strictly
increasing transform of a model's output leaves its AUC unchanged. That has a
direct consequence for blending: averaging raw probabilities lets the *shape*
of each model's output distribution decide how much say it gets. A model whose
probabilities pile up near 0 and 1 shifts a probability average far more than
one whose predictions sit in a narrow band, even if both rank the rows equally
well — so probability-averaging silently weights by confidence calibration,
which is precisely the thing the metric ignores. Converting each model to
ranks first strips that arbitrary scaling out, so each model contributes
exactly the information AUC reads.

Because every run shares the same fold assignment, OOF vectors are row-aligned
and a blend can be scored **without refitting anything**.

### First attempt: every blend was worse

Blending the tuned model with the earlier, weaker ones — all 26 subsets of
`{012, 008, 004, 005, 009}`:

| blend | OOF AUC | vs `012` alone |
|---|---|---|
| `012` alone | 0.963192 | — |
| `012`+`008` | 0.960982 | −0.002210 |
| `012`+`005` | 0.960683 | −0.002509 |
| `012`+`004` | 0.956516 | −0.006676 |
| `012`+`008`+`005`+`009` | 0.958896 | −0.004296 |

Not one of the 26 beat the single tuned model. Weighting did not rescue it —
sweeping the weight on `012` from 0.5 to 0.98 the score climbs monotonically
toward the tuned model alone, i.e. the optimum is "don't blend":

| weight on `012` | 0.5 | 0.7 | 0.85 | 0.95 | 0.98 |
|---|---|---|---|---|---|
| `012`+`008` | 0.960982 | 0.962193 | 0.962814 | 0.963093 | 0.963155 |

**Blending averages skill as well as errors.** The partners were both highly
rank-correlated with the tuned model (Spearman 0.985 for `008`, 0.985 for
`005`) *and* strictly weaker by 0.007–0.023 AUC. Correlated means little
independent error to cancel; weaker means every unit of weight given away
costs more than it returns. The random forest is the least correlated of them
(0.958, genuinely the most "diverse") and hurts the most, because it is also
the weakest by a distance.

**Diversity only pays when the models are comparably strong.** Diversity plus
a large skill gap is just dilution.

### Second attempt: comparably strong models

That diagnosis is testable. The search's 2nd and 3rd-ranked candidates scored
within 0.00007 of the winner on the subsampled CV but with materially
different shapes — `max_depth` 12 and `None` against 8, `learning_rate` 0.038
and 0.101 against 0.144. Re-measured on full data they became runs `013`
(0.962583) and `014` (0.962955): comparable in strength, different in
construction.

| blend | OOF AUC | vs `012` alone |
|---|---|---|
| `012`+`014` | **0.963472** | **+0.000281** |
| `012`+`013`+`014` | 0.963348 | +0.000157 |
| `012`+`013` | 0.963212 | +0.000020 |
| `013`+`014` | 0.963033 | −0.000159 |

```
compare 012 vs blend(012+014) :  +0.000281   95% CI [+0.000245, +0.000321]
compare 012 vs blend(012+013+014):  +0.000157   95% CI [+0.000113, +0.000201]
```

Now blending helps, and the interval excludes zero. The hypothesis held.

The weight sweep is nearly flat between 0.4 and 0.7 (optimum ~0.6 at
0.963480, against 0.963472 at equal weights), so equal weighting is used —
picking the argmax weight off the OOF vector is fitting the blend to the
validation data for a gain of 8 millionths.

### And yet: +0.000281 is 0.59× the fold noise

Run `012`'s fold std is 0.000474. The blend's gain is smaller than the
spread between folds of the model it improves. It is statistically real —
500 paired bootstrap resamples, interval well clear of zero — and it doubles
the fit cost for something that would be invisible in any single train/test
split. **This is the third time in this project that "statistically
significant" and "worth having" have come apart**, and the pattern is always
the same: 691k rows buys enough power to resolve differences far below the
level at which a difference matters.

## What was not submitted, and why

The blend was **not submitted**. The submission rule for this project is that
a submission must be a full-data run logged in `experiments/log.csv`, and a
blend has no fit of its own to log — `experiments/log.csv` rows are single
runs of the harness. `src/blend.py --submit` writes the file, and it was left
unsubmitted deliberately rather than by oversight.

Run `016` (tuned + external data, OOF 0.963349) does technically clear the
bar. It was also not submitted: it beats `012` by +0.00016, which is below
`012`'s own fold noise, so the leaderboard point would have cost a submission
to measure something the OOF already says is negligible.

Two of the three available submissions were used, on runs `008` and `012` —
the two changes large enough that the leaderboard could say something the
cross-validation could not.

## Reproducing

```bash
python -m src.tuning --n-iter 40 --search-rows 250000 --search-folds 3   # run 012
python -m src.blend 012 008 004 005 009 --all-subsets
python -m src.blend 012 014
python -m src.blend 012 013 014
```

# Phase 12 — the last four levers

**Result: three landed, one was null, and the whole exercise moved the blend
by +0.00008 — below the noise floor. This report is mostly a record of
diminishing returns, which is the useful part.**

| run / blend | OOF AUC | vs previous |
|---|---|---|
| `023` HistGBM encoded, re-tuned | 0.967709 | — |
| `025` **+ 10 enc folds + transductive counts + trig** | **0.967839** | +0.00013 `[+0.00008, +0.00018]` |
| blend `024`+`023` (previous best) | 0.968271 | — |
| **blend `024`+`025`** | **0.968351** | **+0.00008** `[+0.000057, +0.000103]` |

Every interval excludes zero. Every gain is a fraction of the 0.00045 fold
noise.

## The three that worked

Isolated on fold 1, each added on top of the last:

| lever | fold-1 AUC | Δ |
|---|---|---|
| baseline (`023` config) | 0.966812 | — |
| + 10 encoding folds instead of 5 | 0.966947 | +0.000135 |
| + transductive frequency counts | 0.966996 | +0.000049 |
| + trig on the two lookup columns | 0.967075 | +0.000079 |

**Transductive counts** take the level counts over train *and* test rows —
987,671 instead of ~553,000. The distinction that makes this legitimate is
worth stating precisely: test *features* are handed to us at prediction time,
and only test *labels* are hidden. Counting how often a value occurs uses no
labels. The **target encoding stays fold-based and nested**; only the counts
changed.

It was reported as +0.00032 for a single model and +0.000015 inside a dense
stack. We measured +0.000049 — much closer to the stacked figure than the solo
one, even though we ship only two models. Being "near-solo" was not close
enough.

**Trig features** add no information whatsoever. The model already has it,
exactly, from the encoded columns. What changes is the *price*: a split on
`sin(2πx/20)` selects a **union of disjoint intervals** of x, where a split on
`x` can only take one contiguous range. Against a lookup table that jumps 0.22
in target rate between neighbouring values, that is much cheaper than target
encoding's one-split-per-value.

That is the third time this project has met the same distinction — after the
decimal lattice and the imputation-augment result. **"The model has this
information" does not imply "the model can afford to use it."**

## The one that was null, and why

Logit stacking, scored honestly (combiner fit on half the OOF rows, scored on
the held-out half, five shuffles, rank average scored on the same halves so
the comparison is paired):

| | |
|---|---|
| equal-weight rank average | 0.968199 |
| logistic stack on logits | 0.968203 |
| **difference** | **+0.000005** (sd 0.000004) |

The coefficients say why: **+0.5687** for CatBoost and **+0.4512** for
HistGBM. Both positive. A stacker's one advantage over a rank average is the
ability to assign a **negative** weight — to use a weak, decorrelated member
as a *correction* rather than as something to average in. With two comparably
strong members that both help, there is nothing to correct.

The published +0.0004 for this lever came from a 12-model library containing
members that needed exactly that treatment. **The lever was real; our library
is the wrong shape for it.** The code is kept, because the moment a third and
weaker member joins, this is precisely the situation where a stacker starts to
matter.

## The three-way blend is worse than the two-way

| blend | OOF |
|---|---|
| `024` + `025` | **0.968351** |
| `024` + `025` + `023` | 0.968292 |

Adding `023` *lowers* it. `025` is `023` plus three small levers, so they are
near-duplicates, and averaging both against CatBoost dilutes the diversity
rather than adding to it — the Phase 6 finding again, from a new direction.
Spearman between CatBoost and the HistGBM member is 0.9861, essentially
unchanged from 0.9865 before these levers, which confirms they did not buy any
new independence.

## Honest accounting

The predicted leaderboard gain over our last submission is **+0.00009**:

| | |
|---|---|
| blend OOF | 0.968351 |
| covariate shift | +0.000850 |
| fold-averaging bonus (already-averaged) | +0.000371 |
| **predicted LB** | **0.96957** |
| currently submitted | 0.96948 |

That is inside the ~0.0007 sampling noise of a single public-LB AUC. **We
would not be able to tell whether it worked.**

Which matches what `reports/remaining-levers.md` predicted before any of it
was built: perhaps +0.0005 combined, from levers whose published figures were
solo measurements. We got roughly half that in OOF and a sixth of it in
expected LB, because our two-model blend already captures most of the variance
reduction these levers supply.

**The right conclusion is that this line of work is finished.** Not because
the levers were wrong — three of four were real and all four were correctly
predicted in sign — but because each one is now a smaller fraction of the
noise than the last. The next honest move is to stop, not to look for a fifth.

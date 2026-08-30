# Loop iterations — a self-paced search for what is left

Started 2026-08-28 from OOF 0.968351 / LB 0.96955 (rank 759 of 2,982), after
`reports/remaining-levers.md` concluded the work was close to done. Ten
iterations budgeted, one hypothesis each, with a standing rule: **three
consecutive iterations without a gain whose CI excludes zero and the search
stops.**

---

## Iteration 1 — CatBoost hyperparameters. NULL (strike 1)

**Hypothesis.** CatBoost is the stronger blend member (run `024`, OOF
0.967906) but its parameters were never searched — they came from a two-point
fold-1 sweep in Phase 11. The Phase 10 re-tune moved HistGBM +0.00045 on a
model that had *already* been tuned once, so an untuned model should have at
least that much on the table.

**Falsifier.** A sweep over depth, learning rate, L2 and iteration count fails
to beat run `024`'s fold-1 val AUC of 0.967118 by +0.0002.

### The sweep

Seven configurations, ranked on a 250,000-row subsample of fold 1's training
rows and scored on the **full** fold-1 validation set, so only the ranking is
being read.

| config | subsample val AUC | vs current |
|---|---|---|
| **slow+long** d8, lr 0.03, l2 3, 2400 iters | **0.966041** | **+0.000212** |
| current d8, lr 0.06, l2 3, 1200 iters | 0.965829 | — |
| deep+slow d10, lr 0.03, l2 10, 2400 | 0.965818 | −0.000011 |
| light L2 0.5 | 0.965726 | −0.000103 |
| shallower d6 | 0.965720 | −0.000109 |
| heavy L2 20 | 0.965679 | −0.000150 |
| deeper d10 | 0.965433 | −0.000396 |

Only the learning rate mattered. Every depth change and every L2 change hurt,
including d10 *paired with* the good learning rate — which isolates depth as
the problem rather than the combination. The winning direction is the same one
the Phase 10 HistGBM re-tune found, which is mild evidence it is a property of
this representation rather than of either library.

### And then it did not transfer

Before spending ~2.2 hours on a full 5-fold at 2400 iterations, the winner was
re-measured on the **full** fold-1 training rows:

| | fold-1 val AUC |
|---|---|
| run `024` config (reference) | 0.967118 |
| tuned d8 / lr 0.03 / 2400 | **0.967157** |
| **delta** | **+0.000039** |

**The subsample promised +0.000212 and full data delivered +0.000039 — a 5×
shrinkage.** The *ranking* transferred; the *magnitude* did not.

The mechanism is the one flagged as a caveat back in
`reports/tuning-and-ensembling.md`: a subsample search "optimises a slightly
different problem than the one it selects for". At 250,000 rows the model is
data-limited, so a slower learning rate with more iterations buys real
accuracy. At 553,095 rows there is enough data that the faster rate already
converges, and most of the benefit evaporates.

**+0.000039 is 0.09× the fold noise of 0.00045.** Not worth 2.2 hours of
compute, and not worth a log row. Recorded as a null.

### What this costs the method

This is the second time a subsample search has misled about magnitude — Phase
6's search also over-stated what the full data would show. The honest
conclusion is that **a subsample search is a device for ranking candidates,
never for sizing a gain**, and any figure it produces needs re-measuring at
full scale before it justifies compute. That check cost 25 minutes here and
saved 2.2 hours.

### A process failure worth recording

The first launch of the sweep died with `ModuleNotFoundError: No module named
'src'` — a script run from the scratchpad puts *that* directory on `sys.path`
rather than the repo. This is the identical failure already written up in
`reports/catboost.md`, repeated within the same session by the person who
wrote it up.

Documenting a mistake does not prevent it. What prevents it is making the fix
part of the launch itself, so `PYTHONPATH=/home/user/KaggleSmartphoneAddiction`
is now a standing rule in the loop's own prompt rather than something to
remember.

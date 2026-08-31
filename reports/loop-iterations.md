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

---

## Iteration 2 — LightGBM as a third family: **a real gain, +0.000156**

Iteration 1 tried to make an existing member better and failed. This iteration
tried the other direction: add a member that is *worse* but *different*.

### Why a weaker model can help

The blend gains from disagreement. If two models rank rows almost identically,
averaging them cancels almost nothing; the average is just a slightly quieter
copy of either. What a third member contributes is the part of its ranking the
others do not already have — so the question is never "is it as good?" but
"does what it gets right, and what it gets wrong, differ from the others?"

The three members now attack the lookup-key problem three genuinely different
ways:

| member | how it handles a categorical level |
|---|---|
| `histgbm_encoded_v2` (025) | nested target encoder, **one global smoothing constant** for every level |
| `catboost_lookup` (024) | ordered target statistics, **shrinkage adapted per row** under a random permutation |
| `lgbm_lookup` (026) | **no target statistic at all** — sorts levels by accumulated gradient inside the node, then cuts that ordering |

The third is the reason to expect decorrelation, and also the reason to expect
it to be weaker: sorting by gradient is a heuristic where the other two
estimate the quantity directly.

### It is weaker, and it is more different

Full 5-fold, run `026`: **OOF 0.965839**, folds ±0.00053, 2123s.

    026 vs 024   -0.00207   95% CI [-0.00219, -0.00194]

Decisively the weakest of the three. But on the full OOF vectors:

| pair | Spearman |
|---|---|
| CatBoost vs HistGBM enc v2 | 0.9861 |
| CatBoost vs **LightGBM** | 0.9819 |
| HistGBM enc v2 vs **LightGBM** | 0.9791 |

LightGBM is *further* from each established member than they are from each
other. The diversity premise holds on full data, not just on the screening
fold.

### The result

    blend 024+025+026    OOF 0.968507
    baseline 024+025     OOF 0.968351
    vs 024+025         +0.000156   95% CI [+0.000120, +0.000199]
    -> a real improvement on the baseline blend.

**New best OOF: 0.968507.** A member 0.0021 weaker than the best single model
made the blend better, which is the clearest demonstration in this project so
far that blend membership is about decorrelation, not about solo strength.

### The screening bar was wrong, and so was the published cliff

Two things this iteration overturns.

**The +0.0002 single-fold screening bar would have discarded this.** Neither
config cleared it (fold-1 blend deltas +0.000108 and +0.000130). The full
5-fold delta is +0.000156 — *between* the two screening estimates, so the
screening numbers were not even biased, merely too noisy at one fold to clear
an arbitrary threshold. A single fold has ~1/√5 the precision of the OOF and
the bar was set without reference to that. **The bar is now: screen for sign
and consistency, not for magnitude against a fixed threshold.**

**The ~0.966 "competitiveness cliff" from the published ledger is
contradicted.** A member at 0.9658 — 0.0002 *below* the supposed cliff —
helps, consistently, in every subset it appears in. A cliff in solo score is
the wrong frame entirely; there is no threshold a member must clear, only a
trade between how much weaker it is and how much less correlated.

### Two negative results worth keeping

**Adding both LightGBM configs hurts** (fold 1: -0.000152 against +0.000108
and +0.000130 for either alone). Two near-duplicates split the blend's weight
between themselves and dilute the two established members without adding a
third independent view. Diversity is a property of the *set*, not of each
member.

**A 4-way blend is a marginal extra that I do not trust.** Every one of the
top seven subsets contains 026, which is real evidence. But the top of the
table is:

| subset | OOF AUC |
|---|---|
| 024+025+026+023 | 0.968534 |
| 024+025+026+022 | 0.968518 |
| **024+025+026** | **0.968507** |
| 024+025+026+022+023 | 0.968484 |

Adding 023 measures +0.000027, 95% CI [+0.000010, +0.000043] — nominally
excluding zero. **That interval is optimistic and should not be read as a
gain.** It was chosen as the best of 26 subsets scored on the same OOF rows
the bootstrap then resamples; the bootstrap prices row noise but not the
selection over subsets. The honest statement is that the top four subsets are
separated by 0.00005 — a tenth of the fold noise — and are not distinguishable.

**The three-way is adopted** because it was the hypothesis stated before the
subsets were scored, and because 023 is a second HistGBM rather than a fourth
mechanism. Taking the argmax of a 26-row table is how a project starts
fitting its own validation set.

### Harness change

`src/blend.py` gained `--vs`, which bootstraps a candidate blend against a
*baseline blend* instead of against its own strongest member. Without it there
is no way to ask "does a third member help?" — the built-in comparison only
answered "does the blend beat its best member?", which stays true whether the
new member helps or hurts. The baseline is built by the same rank-average as
the candidate so the difference isolates the extra member. Verified by
reproducing the known Phase 12 result from a fresh direction: 024+025 over
024+022 is +0.000178, CI [+0.000145, +0.000209].

### The leaderboard, and a prediction that missed

Submitted `blend_024_025_026.csv`. The prediction was stated *before*
submitting, so this is a test rather than a story told afterwards.

| | OOF | public LB |
|---|---|---|
| 024+023 (Phase 11) | 0.968271 | 0.96948 |
| 024+025 (Phase 12) | 0.968351 | 0.96955 |
| **024+025+026 (this)** | **0.968507** | **0.96978** |

Rank **901 -> 838** of 3,389 teams.

    predicted  0.96969   (slope 0.875 fitted on the two earlier blends)
    actual     0.96978
    error     +0.00009   -- the prediction was LOW

The direction of the miss is the interesting part. The two earlier blends gave
ΔLB/ΔOOF = 0.875; this one gives **1.47**. The OOF gain of +0.000156 bought
+0.00023 on the leaderboard — half again as much as it "should" have.

**So the slope is not a slope.** Three points, two incompatible ratios. The
honest conclusion is narrower than the one I was heading for: OOF improvements
have transferred to the leaderboard in the same direction *every time* across
eleven submissions, but the magnitude is not predictable to better than about
a factor of two, and any future LB prediction in this project should be given
as a direction plus a range, not a number.

There is a plausible mechanism, flagged as a hypothesis rather than a finding:
the OOF-to-LB gap here was already decomposed into covariate shift plus a
*variance-dependent* fold-averaging bonus, and the test predictions average
five fold-models. A gain that comes from **decorrelation** reduces prediction
variance directly, so it should collect more of that bonus than a gain of the
same OOF size that comes from a single member simply being stronger. That
would predict exactly what happened here — a diversity gain over-transferring
where the earlier strength gains under-transferred. **One point is not
evidence for it.** It becomes testable if a future iteration produces another
diversity-driven gain and another strength-driven one.

---

## Iteration 3 — a non-tree function class: **null, and mildly harmful**

### Hypothesis and falsifier, stated before the run

Every member of the blend is an axis-aligned tree ensemble: each carves the
space into boxes and fits a constant in each. They differ in *how* they choose
the boxes, which is what iteration 2's gain came from. An MLP differs in what
it can express at all — a smooth global function, where a diagonal boundary
costs one unit instead of a staircase of splits.

**Hypothesis.** Since iteration 2 established that decorrelation rather than
solo strength buys blend gains, the most mechanistically distinct member
available should pay even though it scores worse solo.

**Falsifier.** The 4-way blend against the 3-way baseline has a CI including
zero or negative.

**Secondary prediction.** Spearman below 0.97 against every tree member; all
tree-tree pairs sit at 0.979 or above.

### Result: falsified

Run `027`, full 5-fold: **OOF 0.964171**, folds ±0.00039, 664s.

    blend 024+025+026+027   0.968321
    baseline 024+025+026    0.968507
    vs 024+025+026        -0.000186   95% CI [-0.000218, -0.000155]
    -> WORSE than the baseline blend.

**0.41x the fold noise, and the interval excludes zero on the wrong side.**
Not merely a null — adding this member actively costs.

The fold-1 probe predicted this almost exactly (-0.000181 for the registered
width, -0.000144 for a wider one, against -0.000186 at full scale). Worth
noting against iteration 2, where the fold-1 screen *underestimated* a real
gain: a single fold is not reliably pessimistic or optimistic, it is simply
noisy. The lesson stays "measure it properly", not "fold 1 lies in a known
direction".

### The secondary prediction was also wrong, and that is the interesting part

| pair | Spearman |
|---|---|
| CatBoost vs HistGBM | 0.9861 |
| CatBoost vs LightGBM | 0.9819 |
| HistGBM vs LightGBM | 0.9791 |
| CatBoost vs **MLP** | 0.9769 |
| HistGBM vs **MLP** | 0.9820 |
| LightGBM vs **MLP** | **0.9688** |

Predicted below 0.97 against every tree member; it holds only against
LightGBM. Against HistGBM it is 0.9820 — inside the tree-tree range.

**Why: a design flaw I own.** `mlp_encoded` reuses run 025's *exact* Phase 12
representation. Function class pushes the two apart, a shared feature space
pulls them back together, and the second effect roughly cancelled the first.
So this result is weaker evidence against neural members in general than it
looks; it tests "a non-tree model on an identical representation", not "a
non-tree model".

### What actually explains the loss

Not "too weak" on its own. The decisive comparison is *which* members a
candidate is decorrelated from:

| member | solo OOF | mean rho vs the two strong members | blend delta |
|---|---|---|---|
| LightGBM `026` | 0.965839 | 0.9805 | **+0.000156** |
| MLP `027` | 0.964171 | 0.9795 | −0.000186 |

The MLP is **0.001 less correlated** with the members that carry the blend —
essentially nothing — while paying **0.00167** in solo strength. The genuine
distinctiveness it does have is spent against LightGBM (0.9688), the *junior*
member, where it buys almost nothing.

**The rule this yields, sharper than iteration 2's:** a new member's
decorrelation only pays *against the members already carrying the blend*.
Decorrelation from a weak member is nearly worthless, because that member's
own contribution is small to begin with. Iteration 2's headline —
"decorrelation beats solo strength" — was too loose. The corrected version:
**a member must be decorrelated from the leaders, and the cost of that
decorrelation is paid in solo strength at roughly a 1:1 exchange rate.**
LightGBM bought 0.0056 of decorrelation from CatBoost for 0.0021 of strength
and paid off; the MLP bought 0.0010 for 0.0037 and did not.

### Kept anyway

The code stays in the tree (`nn_preprocessor`, `mlp_encoded`), and run `027`
stays in the log with its OOF vector on disk. A negative member is still a
measured, reproducible artefact, and the next iteration's question — whether a
*different representation* rather than a different function class produces
decorrelation from the leaders — needs this vector to compare against.

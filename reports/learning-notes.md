# Learning notes

The concepts this project is built on, explained for someone learning tabular
ML. Every number quoted here comes from a run recorded in
`experiments/log.csv` or from a script in `src/`; nothing is illustrative.

---

## 1. Why out-of-fold predictions matter

A model's score on data it was fit on is not an estimate of anything useful.
It tells you how well the model memorised, and a model with enough capacity
can memorise perfectly while learning nothing that transfers.

The fix is to score each row using a model that never saw it. Split the data
into 5 folds; for each fold, fit on the other 4 and predict this one. Every
row ends up with a prediction from a model trained without it. Stack those
predictions and you have an **out-of-fold (OOF) vector** — one prediction per
training row, all of them honest.

Two properties make this more than "a validation set, but 5 times":

- **Every row is used for evaluation.** A single 80/20 split evaluates on 20%
  of your data. OOF evaluates on 100%, so the estimate is far less noisy.
- **Every row is used for training.** Each fold's model sees 80%, and you get
  5 of them.

This project computes OOF AUC **once over the whole stacked vector**, not as
the average of 5 per-fold AUCs. Those are different numbers — the fold average
weights each fold equally regardless of how the positives fall in it — and the
single global figure is the more stable one.

---

## 2. Why identical folds make comparisons paired

`src/validation.py` hard-codes `StratifiedKFold(shuffle=True,
random_state=42)` for every run in the project. That is the single most
important design decision in the repo.

When two models see byte-identical folds, their OOF vectors are **row-aligned**:
position 12,345 in both vectors is the same training row, scored by a model
that did not train on it. So you can compare them **row by row**.

This matters because most of the variation in a CV score is not about the
model at all — it is about which rows landed where. Row 12,345 might be
intrinsically hard, and *both* models get it wrong. If you compare model A on
split 1 against model B on split 2, that shared difficulty does not cancel and
you are measuring split noise on top of the model difference.

`bootstrap_auc_diff` exploits the alignment directly: it resamples rows, and
scores **both** models on the *same* resampled rows. The shared per-row noise
cancels, and what is left is the difference between the models.

The practical payoff, from `experiments/log.csv`: this harness resolved the
categorical features' contribution as +0.00003 with a CI of `[+0.00002,
+0.00004]` — three hundred-thousandths of an AUC point — and then the
leaderboard, on data neither model had seen, returned +0.00002. A comparison
that is not paired cannot do that.

A side benefit: because the vectors are aligned, **a blend can be evaluated
without refitting anything**. `src/blend.py` scored 26 different blends from
files on disk.

---

## 3. Why AUC is invariant to monotonic transforms

ROC AUC has a clean interpretation: **the probability that a randomly chosen
positive row is ranked above a randomly chosen negative row.** It reads only
the *ordering* of the predictions, never their values.

So if you take every prediction and pass it through any strictly increasing
function — multiply by 3, take the log, apply a sigmoid — the ordering is
unchanged, and the AUC is **exactly** unchanged.

Three consequences run through this project:

**Calibration cannot help.** `CalibratedClassifierCV` and friends adjust
predicted probabilities to better match observed frequencies, using a monotonic
map. Under AUC that is a no-op. Time spent on calibration for an AUC metric is
time spent for zero points. (Under log-loss or Brier score it would matter a
great deal — the metric decides.)

**Blending must use ranks, not probabilities.** If you average two models'
raw probabilities, the one whose outputs spread toward 0 and 1 dominates the
average, while a model whose predictions sit in a narrow band barely moves it
— even if both rank rows equally well. You have silently weighted by output
scale, which is exactly the thing AUC ignores. Converting each model to ranks
first (`scipy.stats.rankdata`) removes the arbitrary scaling. See
`reports/tuning-and-ensembling.md`.

**But "monotonic" has a direction, and the direction matters.** See §7 — this
project got that wrong and had to be corrected by measurement.

---

## 4. What permutation importance on training data gets wrong

Permutation importance shuffles one column — destroying its relationship to
the target while keeping its marginal distribution — and measures how far the
score falls. The drop is that column's contribution.

**Which rows you measure on decides what the number means.**

On **training** rows, the drop reflects what the model *used*, including
whatever it memorised. A column the model overfit shows a large drop, because
breaking the column breaks the memorisation. On **held-out** rows, the drop
reflects what the column contributes to *generalisation*.

Demonstrated directly (`src/importance.py --noise-col`). A column of pure
U(0,1) noise — true importance exactly zero, by construction — was added to a
deliberately overfitted model (20,000 rows, `max_leaf_nodes=512`):

| permuted on | `noise_uniform` importance | rank |
|---|---|---|
| training rows | **+0.001356** | 10th of 13 — above `gender`, `academic_work_impact`, `stress_level` |
| held-out rows | **−0.000431** | 13th of 13, last |

On training rows a provably worthless column outranks three genuinely
informative ones.

**An honest caveat.** On the *actual* well-regularised model at 691k rows, the
train and validation tables are nearly identical (`daily_screen_time_hours`
0.149372 vs 0.149422). The failure mode needs a model that overfits to appear.
The rule is still "always measure on held-out rows", because you do not know in
advance whether your model is in the regime where it bites.

### The bigger caveat: correlated features

Permutation importance answers *"how much does **this fitted model** degrade
when I scramble **this specific column**?"* — not *"how much is this
information worth?"* When two columns are correlated, shuffling one creates
feature combinations that **never occur in real data**, and the model's
behaviour on impossible inputs is not evidence about anything.

So the importance table is a **ranking aid, not a pruning rule**. In this
project, pruning to the features the table endorsed made the model *worse*:
0.956091 → 0.956023, CI `[-0.00011, -0.00002]`. Ten features that each measured
≈0 individually were collectively worth a real, if tiny, amount — because each
was measured with all the others still present.

---

## 5. What adversarial validation detects

Throw away the real target. Label training rows 0, test rows 1, and try to
predict which file a row came from. AUC ≈ 0.5 means the two are exchangeable.
AUC ≫ 0.5 means they differ, and the feature importances say where.

Here it returned **0.565161** — separable. Decomposing it:

| discriminator sees | AUC |
|---|---|
| everything | 0.565161 |
| the is-missing pattern only | 0.565382 |
| complete cases only | **0.499160** |

**All of the separation is missingness.** The values are drawn from identical
distributions — KS tests on all nine numeric columns are non-significant
(p from 0.14 to 0.997) — but every column's *missing rate* differs by 0.6 to
3.4 percentage points.

**A high adversarial AUC does not automatically invalidate OOF.** It says
train and test differ; whether that *matters* depends on whether the shift
touches something the model uses to predict `y`. Phase 3 had already
established that missingness carries no target signal here (a model on the
is-missing flags alone scores 0.50038). So `P(y | X)` is unchanged and the
models remain correct — what changes is the *mix* of easy and hard rows. Test
has 2.55 points more fully-observed rows, which score 0.963, and fewer rows
missing all three top features, which score 0.805. **The test set is simply
easier.**

That is not a curiosity — it is quantifiable and it predicted the leaderboard.
Reweighting the OOF rows by the adversarial model's own density ratio
(`p / (1 − p)`, the standard covariate-shift correction) predicted run `008`
would score **0.95729**. It scored **0.95708**.

**What adversarial validation does not test:** whether `P(y | X)` is stable. It
never looks at `y`. Two files can have identical feature distributions and
completely different target relationships, and this method would report 0.5.

Also note the failed control that got kept: a "values only" discriminator using
median imputation still scored 0.561, because **median imputation does not
erase missingness, it encodes it** — every imputed cell lands on exactly the
median and the discriminator counts the spike. Complete-case analysis is what
actually removes the channel.

---

## 6. "Statistically significant" ≠ "practically useful"

With 691,369 rows there is enough statistical power to resolve differences far
below the level at which a difference matters. This project produced the
pattern **four** times:

| finding | difference | 95% CI | fold noise | verdict |
|---|---|---|---|---|
| categorical features | +0.00003 | `[+0.00002, +0.00004]` | 0.00078 | real, irrelevant |
| pruning to "surviving" features | −0.00007 | `[-0.00011, -0.00002]` | 0.00078 | real, irrelevant |
| external source dataset (tuned) | +0.00016 | `[+0.00008, +0.00025]` | 0.00044 | real, irrelevant |
| rank-average blend `012`+`014` | +0.00028 | `[+0.00025, +0.00032]` | 0.00047 | real, marginal |

Compare with the two that actually mattered:

| finding | difference | 95% CI | fold noise | verdict |
|---|---|---|---|---|
| engineered ratio features | +0.00139 | `[+0.00129, +0.00148]` | 0.00068 | **real and useful** |
| hyperparameter tuning | +0.00710 | `[+0.00694, +0.00726]` | 0.00047 | **real and large** |

The confidence interval answers *"is this difference real?"* The **fold
standard deviation** answers *"is it bigger than the run-to-run wobble I would
see anyway?"* You need both. A CI excluding zero on a difference 11× smaller
than your fold noise means you have precisely measured something that will
never show up in practice.

The discipline this implies: report the interval *and* the noise floor, and
say plainly when a real result is not a useful one.

---

## 7. Two claims this project got wrong, and how measurement caught them

The most useful thing in this repo is not the score. It is two confident,
written-down predictions that turned out to be false.

### "A monotonic transform of one column cannot help a tree"

The argument: a tree splits on `x <= t`, so a transform preserving the
ordering produces the identical partition of the rows. `sleep_deficit =
8 - sleep_hours` was included as a control precisely because this argument
said it must be worthless.

Ablating it **cost** +0.00018, CI `[+0.00015, +0.00022]`.

Isolating the mechanism with three duplicate columns run through full CV:

| duplicate of `sleep_hours` | OOF AUC | vs no duplicate |
|---|---|---|
| exact copy | 0.954704 | **+0.000000**, CI `[0, 0]` |
| `3.7 × sleep_hours` (increasing) | 0.954704 | **+0.000000**, CI `[0, 0]` |
| `8 − sleep_hours` (decreasing) | 0.954895 | **+0.000191**, CI `[+0.000155, +0.000230]` |

An exact copy changes nothing, so it is not "an extra column". A strictly
*increasing* rescale changes nothing **bit-identically**, so the argument is
correct for increasing transforms. Only **order reversal** does anything.

Why: `x <= t` sends the boundary group **left**; the reversed `8 - x <= 8 - t`
sends it **right**. Where a threshold falls inside a group of tied values, the
two columns express *different* partitions and the original cannot produce the
reversed one at any threshold. Ties are everywhere because
HistGradientBoosting **quantises every column to at most 255 bins** before
searching: `sleep_hours` has 451 distinct values, and 144 of its 241 occupied
bins straddle a boundary in the reversed column's binning.

The correct statement is narrower than the one I started with: **monotonic
*increasing* transforms are useless to a tree; decreasing ones are not, once
binning or ties are in play.**

### "The OOF-to-leaderboard gap is a stable property of the pipeline"

The README claimed this, on the evidence of two submissions whose gaps agreed
to 7 millionths (−0.001083 and −0.001076). Then a third agreed too
(−0.000989). Then the tuned model broke it: **−0.001918**, nearly double.

The explanation was measurable. Test predictions are the average of 5 fold
models, while each OOF row is scored by 1 — and averaging helps a
**high-variance** model more than a low-variance one. Measured on an 80/20
same-population split, comparing on identical rows:

| model | single model | 5-model average | ensemble gain |
|---|---|---|---|
| default (run `005`) | 0.953864 | 0.954453 | **+0.000590** |
| tuned (run `012`) | 0.962207 | 0.963220 | **+0.001012** |

The tuned model has 69 leaves and 463 iterations against 31 and 100. More
capacity means more variance, means more to gain from averaging, means a
bigger gap. The gap is not a property of *the pipeline* — it is a property of
*the model's variance*, and it moves when you change the model.

**The lesson is not "we were sloppy".** Both claims were reasonable given the
evidence available when they were made. The lesson is that a claim which has
only ever been tested in a narrow regime is not yet a law, and the way you
find out is to keep testing it when the regime changes.

---

## 8. Reproducibility is a dependency problem

The first thing this phase did was fail. Regenerating run `005`'s OOF vector
produced 0.954815 against a logged 0.954704 — off by 1.1e-4, which is larger
than several differences this project set out to resolve.

The harness was fine. scikit-learn 1.5.2, 1.6.1, 1.7.0 and 1.8.0 all reproduce
both logged values to six decimals, `fold_std` included. Only **1.9.0**
diverges, because it changed how `_BinMapper` places histogram bin edges: on
this data `n_bins_non_missing_` goes from `[18,255,255,255,255,255,231,166,255]`
to `[18,254,253,246,255,242,231,166,255]`. Move the bin edges and every
candidate split point moves with them.

`requirements.txt` said `scikit-learn>=1.4`. That upper-bound-free pin was
enough to silently invalidate every paired comparison in the project. It now
reads `>=1.5,<1.9`, with the reasoning inline.

Two general points:

- **A comparison is only paired if both sides came from the same
  environment.** Fixing the seed is not enough; the library version is part of
  the experiment.
- **Regenerating an old result is a test, so make it one.** `--oof-as` refuses
  to write a vector whose reproduced AUC does not match the log. That guard is
  what turned a silent corruption into a five-minute diagnosis.

A related wrinkle: `RandomForestClassifier` accumulates `predict_proba` across
threads, so its output moves at ~3.3e-16 per row depending on scheduling. At
691k rows that flips enough near-ties to shift AUC by 3e-6 — 250× below the
fold noise, and harmless, but it means "bit-identical" is the wrong bar for
that estimator. Hence `--oof-tol`, which defaults to exact.

---

## 9. When a mechanism check is unreliable, throw it out

Trying to explain the categorical result, I walked the fitted trees' internal
node arrays to count splits per feature. The output was self-contradictory:
`age`, never declared categorical, came back with all its splits flagged
`is_categorical`, while `gender` — which `clf.is_categorical_` correctly
reports as categorical — came back with none.

Those counts were interesting-looking and wrong. They were removed from both
reports and replaced with permutation importance, which measures the model
from the outside and needs no assumptions about private data structures.

Worth stating as a habit: **an explanation that rests on a measurement you
cannot verify is worse than no explanation**, because it reads as
authoritative. The discarded attempt is recorded in
`reports/native-categoricals.md` so that nobody re-derives the same wrong
numbers and believes them.

# Design choices

Why this repository is built the way it is. Every section states a decision,
the reason for it, and — where one exists — the measurement that forced it.
Numbers come from `experiments/log.csv`, `experiments/blends.csv`, or a report
under `reports/`. Nothing here is projected.

**Final state:** OOF AUC **0.968507**, public LB **0.96978**, rank 838 of
3,389. 28 logged full-data runs, 3 logged blends.

---

## 0. What the project optimises for

The score is not the deliverable. The deliverable is a **method for deciding
whether a change helped**, on a dataset where the real differences between
good models are 1–3 ten-thousandths of AUC and the fold-to-fold noise is
5–8 ten-thousandths.

That constraint drives almost every choice below. When the effect you are
hunting is smaller than the noise of a single measurement, you cannot rely on
"run it and see if the number went up". You need paired comparisons, a fixed
environment, and an interval around every claim.

The competition itself was chosen to cover a gap: two prior projects are deep
learning on image-like inputs. This one is tabular — pandas, scikit-learn,
tree ensembles, and validation methodology.

---

## 1. One fold assignment, defined exactly once

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, wrapped in
`make_folds` in [src/validation.py](src/validation.py), used by every run in
the project.

**Why.** Two models scored on different splits are not comparable: most of the
variance in a CV score is about *which rows landed where*, not about the
model. With byte-identical folds, any two runs produce **row-aligned** OOF
vectors — position 12,345 is the same row in both, scored by a model that did
not train on it — so shared per-row difficulty cancels.

**Why it is a function and not an inline splitter.** `make_folds` was
extracted so the split has exactly one definition in the codebase. A probe
script that re-derives the folds by hand is one mistyped argument away from
scoring on a different partition and reporting a "difference" that is really
just a different split.

**Consequence that pays for itself.** Because the vectors are aligned, a blend
can be scored **without refitting anything**. `src/blend.py` evaluated 26
candidate blends from files on disk.

---

## 2. The OOF vector is the unit of record, not the fold-mean AUC

`CVResult.oof_auc` computes ROC AUC **once over the whole stacked vector**,
not as the mean of 5 per-fold AUCs.

**Why.** They are different numbers. The fold average weights each fold
equally regardless of how the positives fall in it; the global figure uses
every row once and is the more stable estimator.

`fold_std` is still recorded and reported alongside — not as the headline, but
as the **noise floor**. Any claimed gain gets quoted as a multiple of it. At
691k rows that floor is 0.0005–0.0008, which is how we know that tuning
(+0.0071, ~15×) is a real result and that the engineered-feature gain
(+0.0014, ~2×) is a modest one.

---

## 3. Paired bootstrap is the acceptance test for every claim

`bootstrap_auc_diff` resamples rows and scores **both** models on the *same*
resampled indices, then reports a 95% CI on the difference.

**Why paired.** Resampling each model independently would price in the shared
row-level noise twice and produce an interval far too wide to resolve anything
at this scale.

**The rule.** If the interval straddles zero, the change is not claimed as an
improvement. This is what makes the README's "what didn't work" section
possible: a null result here is a *measured* null, not a shrug.

**How sharp it is.** The harness resolved the three categorical features'
contribution as +0.00003 with CI `[+0.00002, +0.00004]` — an interval that
excludes zero on a difference three decimal places below anything that
matters. That is a deliberate demonstration that "statistically significant"
and "worth having" are separate questions.

**Where it does *not* apply.** A bootstrap prices row noise, not selection.
When the best of 26 blend subsets measured +0.000027 with CI `[+0.000010,
+0.000043]`, that interval was explicitly **rejected** as optimistic, because
the subset was chosen by argmax on the same rows the bootstrap resamples. The
pre-registered 3-way blend was adopted instead. See
[reports/loop-iterations.md](reports/loop-iterations.md), iteration 2.

---

## 4. Preprocessing lives inside the Pipeline, and is cloned per fold

`run_cv` calls `clone(estimator)` for each fold and fits only on that fold's
training rows. Every transformer in [src/preprocessing.py](src/preprocessing.py)
is returned **unfitted**.

**Why.** A transformer fit on the full training set before splitting has seen
the rows it will be scored on. Medians, scalers, category vocabularies and
quantile grids all carry information across the fold boundary. Fitting inside
the pipeline makes leakage structurally impossible rather than something to
remember not to do.

**The stronger version.** [src/features.py](src/features.py) is *stateless* —
row-wise arithmetic only, `fit` genuinely has nothing to do. That is the
strongest possible position on leakage: there is no fitted quantity that
*could* leak. It is still written as a proper transformer so it composes,
clones and fits per fold like everything else.

---

## 5. Three preprocessor families, because the model families need different things

| builder | numerics | categoricals | for |
|---|---|---|---|
| `tree_preprocessor` | passthrough, **NaN intact** | ordinal, missing as its own level | HistGBM, LightGBM |
| `linear_preprocessor` | median-impute + `StandardScaler` | one-hot, `drop="first"` | logistic regression |
| `nn_preprocessor` | median-impute + `QuantileTransformer` | one-hot, `drop="first"` | MLP |

**NaN is preserved for trees, deliberately.** HistGradientBoosting learns a
default routing direction for missing values at every split, which is strictly
more information than a filled-in median. Imputing would *discard* signal.
Confirmed the hard way: adding explicit is-missing indicators to HistGBM
produced **bit-identical** OOF predictions (`np.array_equal` → `True`). The
model already had that information by construction.

**`drop="first"` on the linear one-hot** avoids the dummy-variable trap —
without it each one-hot group sums to the all-ones vector and the design
matrix is rank-deficient.

**Missing indicators cover numerics only.** A categorical is imputed to an
explicit `__MISSING__` level that one-hot then gives its own column, so an
is-missing flag for the same column would be an exact duplicate — perfectly
collinear, and enough to make lbfgs emit overflow warnings during line search.
This was a real bug, found and fixed.

**`QuantileTransformer` rather than `StandardScaler` for the net.** Many
engineered columns are ratios with denominators approaching zero, so they carry
heavy tails. Standardising divides by a standard deviation those tails inflate,
squeezing the bulk of the mass into a narrow band; a net's first layer
saturates and the gradient dies. A linear model does not care — a scale change
is absorbed by its coefficient. Rank-normalising is also monotone per column,
which is exactly the class of transform AUC cannot see.

**Indicators are hand-rolled, not `sklearn.impute.MissingIndicator`.** That
transformer emits columns only for features that had missing values *in the
fold it was fit on*, which lets the feature count drift between folds.

---

## 6. The experiment log is an append-only ledger with entry rules

`experiments/log.csv` is the one tracked artifact in an otherwise gitignored
data path. One row per full-data run: `run_id, timestamp, description, model,
n_features, n_model_features, n_train, oof_auc, fold_mean, fold_std,
fit_seconds, public_lb, gap`.

- **Sampled runs are refused entry.** `--sample` is for fast iteration; a row
  in the log always means all 691,369 rows.
- **`public_lb` and `gap` are blank until a submission actually scores.**
  Filled by `record_lb`, never estimated. Rows without a submission stay blank.
- **`n_model_features` records the width the estimator actually saw**, after
  encoding — which differs from the input column count whenever one-hot or
  indicators expand it. Added after run `017` silently logged 12 against a
  true 27.

**Why the ledger matters more than the scores in it.** The OOF-to-LB gap
column across runs is the artifact the project exists to produce. It is what
makes a resume claim about validation methodology defensible rather than a
leaderboard brag — and it is what allowed the project's own headline claim
("the gap is a stable property of the pipeline") to be **falsified** by later
rows. The correction is kept in the README on purpose.

---

## 7. OOF vectors are gitignored, and regeneration is verified rather than trusted

`experiments/oof/` holds ~5MB per run and is fully regenerable, so it is not
tracked. A fresh clone therefore has the log but not the vectors `src/compare.py`
needs.

```bash
python -m src.experiment --model histgbm --oof-as 005 --no-submission
```

refits a logged run and writes its vector under the original `run_id` **without
appending a second log row** — and **refuses to write unless the reproduced OOF
AUC matches the logged value to six decimals.** That check doubles as a
determinism test on the whole environment, and it is what caught §8.

---

## 8. `scikit-learn>=1.5,<1.9` — an upper bound that is load-bearing

Regenerating run `005` returned 0.954815 against a logged 0.954704. The harness
was fine: 1.5.2, 1.6.1, 1.7.0 and 1.8.0 all reproduce the logged value to six
decimals. Only **1.9.0** diverges — it changed histogram bin placement, so
`n_bins_non_missing_` moved from `[18,255,255,255,255,255,231,166,255]` to
`[18,254,253,246,255,242,231,166,255]` on identical data and seed. Move the bin
edges and every candidate split moves.

The difference, 0.00011, is larger than several of the differences this project
is trying to resolve. Every logged row was produced under the older binning, and
the method is paired comparison against those vectors, so mixing binning regimes
inside one log would silently confound every comparison.

**The general principle: a comparison is only paired if both sides came from the
same environment. Fixing the seed is not enough.** Re-baselining onto 1.9 stays
a legitimate alternative — it just means re-running every logged row, not
appending to them.

---

## 9. The model zoo is a function plus a dict entry

[src/models.py](src/models.py) exposes `MODELS: dict[str, (factory, description)]`.
Adding a model is adding a function and one line. The runner, the log, the
submission writer and the OOF store all work off that key.

**Why factories, not instances.** The harness clones per fold; a shared fitted
instance would be a leak waiting to happen and a threading hazard besides.

**Why every variant gets its own key.** When iteration 5 retuned LightGBM, the
new parameters were registered as a **new** entry `lgbm_lookup_v2` rather than
edited into `lgbm_lookup`, so run `026` stays reproducible from the repo. A log
row whose model definition has since been mutated is not a record of anything.

**Optional dependencies are imported lazily** (`_lgbm_lookup`, `_catboost_lookup`)
so the zoo still loads without them.

---

## 10. Feature engineering: only what a tree cannot reach cheaply

A tree splits on `x <= t`. Any **strictly increasing** function of a single
column therefore yields the identical partition, just with relabelled
thresholds. So `24 - screen_time` and `log(x)` cannot add anything to a tree,
however meaningful they are to a human reading the column list. Reciprocals are
monotone on positive data too, so only one of each `a/b` ↔ `b/a` pair appears.

What a tree genuinely cannot reach cheaply is a **ratio of two columns**: to
approximate `a/b <= t` with axis-aligned splits it must stair-step the boundary,
spending depth and rows at every step. Handing it the quotient turns that
staircase into one split. All 15 engineered columns are ratios, differences, or
products.

Result: +0.0014 OOF (run `008` over `005`), ~2× the noise floor.

**A deliberate control was included, and it failed informatively.**
`sleep_deficit = 8 - sleep_hours` was added expressly to be worthless — a test
of the *measurement*, not of the feature. Ablating it cost **+0.00018**, CI
`[+0.00015, +0.00022]`. Running it down: an exact copy of `sleep_hours` changes
the OOF bit-identically, a 3.7× rescale likewise, and only the order-**reversing**
`8 - x` does anything. `x <= t` sends a boundary tie-group left; `8 - x <= 8 - t`
sends it right, so where a threshold falls inside a group of binned or tied
values the two columns express partitions the other cannot reach. HistGBM
quantises to 255 bins and `sleep_hours` has 451 distinct values, so 144 of its
241 occupied bins straddle a boundary in the reversed column.

**The claim is true for increasing transforms and false for decreasing ones.**
The original text has been corrected rather than deleted.

**Permutation importance is a ranking aid, not a pruning rule.** It endorsed 5
of the 15 engineered features and put the other 10 at or below 4e-5. Dropping
those 10 cost **−0.00007**, CI `[-0.00011, -0.00002]` — the interval excludes
zero, so the loss is real. Each measured ≈0 *with all the others present*;
collectively they were worth something.

---

## 11. Reading the columns as lookup keys — the single largest gain

The EDA said `notifications_per_day` had univariate AUC 0.492 and
`app_opens_per_day` 0.541 — both chance. Four phases treated them as
near-worthless quantities.

**They are not quantities.** The synthetic generator uses the exact value as a
key into a lookup table. Verified on our own data: `notifications_per_day` has a
per-value target-rate standard deviation **22× larger** than sampling noise can
produce, and neighbouring integer values differ in addiction rate by 0.22 on
average. As raw numbers those columns are worth OOF 0.5122; read as lookup keys,
**0.8170**.

A tree cannot recover that. `x <= t` asks an ordering question about a variable
whose order carries no meaning.

**Design decisions inside [src/encoding.py](src/encoding.py):**

- **Every column is encoded, not just the nominal ones.** At 691k rows even
  `daily_screen_time_hours` has ~500 rows behind each of its 1,389 distinct
  values — plenty to estimate a smoothed mean.
- **Both statistics are emitted per column.** Target mean
  `(sum_y + prior*smooth)/(count + smooth)` with `smooth=10` (larger values
  measured worse), plus level **frequency**, which separates a common level
  whose mean is well estimated from a rare one whose mean is not.
- **The raw column is kept alongside its encoding.** Not redundancy: the value
  supports ordering questions ("screen time above 9.8 hours"), the level
  supports identity questions ("this exact value").
- **`fit_transform` is overridden, not inherited.** `transform` uses statistics
  fit on everything the encoder saw, which is correct for validation and test
  rows. It is *wrong* for the training rows themselves — a row would be encoded
  partly with its own target. So `fit_transform` runs an **inner** K-fold and
  encodes each inner fold from the others only. Because sklearn's Pipeline calls
  `fit_transform` during `fit` and `transform` during prediction, wiring it this
  way makes the correct thing happen at both times automatically. Two nested
  layers of protection: `run_cv` keeps the outer validation rows out entirely,
  the inner split protects the training rows from themselves.
- **Frequency counts may be transductive; target statistics never are.** Test
  *features* are handed to us at prediction time; only test *labels* are hidden.
  Counting how often a value occurs uses no labels, so counting over all 987,671
  train+test rows leaks nothing and estimates "how common is this value" better
  than the ~553,000 rows in a training fold. The target encoding does use labels
  and stays fold-based and nested. Only the counts change.
- **`level_frame` routes through `object` before the string cast, on purpose.**
  `X[c].astype(str)` writes the literal `"nan"` on pandas 2.x and silently
  preserves NA on pandas 3.0+, where a subsequent `groupby` would drop every
  missing row from the level statistics and the encoding would vanish on 4–20%
  of each column with nothing raised and nothing warned. This project runs pandas
  3.0.5. `assert_full_coverage` checks the invariant rather than trusting it.

Gain: OOF 0.964709 → **0.967257** (run `020` → `022`), the largest single move
in the project.

---

## 12. Blending: rank average, not probability average

`src/blend.py` converts each model's output to average ranks scaled to (0, 1]
before averaging.

**Why.** AUC depends only on ordering. Averaging raw probabilities lets the
*shape* of each model's output distribution decide how much say it gets — a
model whose probabilities bunch near 0 and 1 moves a probability average far
more than one whose predictions sit in a narrow band, even if both rank rows
equally well. Ranking first strips that arbitrary scaling out.

**`--vs` compares a candidate blend to a *baseline blend*, not to its own best
member.** Without it there is no way to ask "does a third member help?" — the
built-in comparison only answers "does the blend beat its best member?", which
stays true whether the new member helps or hurts.

**The stacker is scored on held-out halves.** A logistic combiner fit on the OOF
matrix and scored on the same rows reads high, because its coefficients were
chosen using those labels. `logit_stack_score` fits on half the OOF rows and
scores on the other half over 5 shuffles, and scores the equal-weight rank
average on the *same* halves so the comparison stays paired. It works on logits
rather than probabilities because this target saturates — two models that both
say 0.999 can disagree by a lot in log-odds. The stacker's one advantage is that
it can give a member a **negative** coefficient, which a rank average cannot
express. Measured: null on this data.

**Adopted blend: `024 + 025 + 026`** (CatBoost + HistGBM-encoded + LightGBM),
OOF 0.968507, LB 0.96978. Three genuinely different ways of handling a
categorical level — a globally-smoothed target encoder, ordered target statistics
with per-row adaptive shrinkage, and gradient-sorted level splits with no target
statistic at all.

---

## 13. Hyperparameter search: subsample for *ranking*, full data for *magnitude*

`src/tuning.py` runs `RandomizedSearchCV` with `cv=3` on a 250,000-row
subsample, purely for runtime. **Nothing from the search is ever reported as a
result.** The winning parameters are handed back to the standard full-data
5-fold `run_cv`, and that number is what enters the log — measured on the same
folds, at the same size, as every other row.

**The subsample is a real cost, not a free saving.** At 250k rows the model is
data-limited, so the search optimises a slightly different problem than the one
it is selecting for. This has now misled twice, in the same direction:

| | subsample says | full data delivers | shrinkage |
|---|---|---|---|
| Phase 6 HistGBM | overstated | — | — |
| Iteration 1 CatBoost | +0.000212 | +0.000039 | **5×** |

The *ranking* transferred; the *magnitude* did not. **A subsample search is a
device for ranking candidates, never for sizing a gain.** The full-scale
re-measurement cost 25 minutes and saved 2.2 hours of compute on a null.

**Single-fold screening is for sign and consistency, not magnitude.** A
"+0.0002 on fold 1" screening bar would have discarded LightGBM, which turned
out to be the only blend gain in the loop (+0.000156, CI excluding zero). A
single fold has ~1/√5 the precision of the OOF; a fixed threshold set without
reference to that is arbitrary.

---

## 14. Submission discipline

- **Do not tune against the public leaderboard.** It is ~20% of the test rows —
  roughly 59,260 — and is being selected on noise by thousands of teams. Across
  seven finished Season 6 episodes, three had *zero* public-top-10 teams remain
  in the private top 10; in S6E7 the public winner finished private rank 440.
  Trust OOF, submit sparingly, record the gap, do not chase it.
- **Predict the leaderboard score before submitting.** Run `008` was called to
  within 0.0002. Run `012` was under-predicted by ~0.0009 by both methods, which
  is precisely what flagged the model-variance component of the gap. A prediction
  recorded in advance is a test; one written afterwards is a story.
- **Report the direction, not the magnitude.** Eleven submissions in, OOF gains
  have transferred to the leaderboard in the same direction *every time*, but
  the ratio ΔLB/ΔOOF has ranged from 0.875 to 1.47. Any future LB prediction is
  stated as a direction plus a range.
- **Every submission carries its OOF score in the `-m` message.**

---

## 15. Calibration is explicitly out of scope

AUC is invariant to strictly increasing transforms. `CalibratedClassifierCV`
applies exactly such a transform. It **cannot** change the score. Interesting as
an analysis aside; useless as an optimisation target, and it is written into the
project constraints so it does not get re-proposed.

---

## 16. Negative results are kept, and corrections are kept in place

The README's "what didn't work" section is not an appendix — for this project it
is the main output. Nulls kept deliberately:

- is-missing indicators for HistGBM (bit-identical OOF)
- missingness as signal (0.50038 — the generator masked independently of the label)
- native categorical declaration (identical to six decimals)
- the original 7,500-row source dataset (−0.00023 on one model, +0.00016 on
  another; opposite signs, both significant, both negligible)
- median imputation as an adversarial "values-only" control (it *encodes*
  missingness rather than erasing it — every imputed cell lands on the median and
  the discriminator counts the spike)
- an MLP blend member (−0.000186, CI excluding zero on the wrong side)
- the most-decorrelated available member (−0.000069)
- a stronger LightGBM (+0.000004; 1.6% of the solo gain reached the blend)

**Two corrections are preserved rather than edited away**, because a project
that silently rewrites its own claims is not evidence of anything:

1. The gap was claimed to be "a stable property of the pipeline". It is not — it
   scales with model variance. Falsified by run `012`.
2. Split counts read from fitted trees' internal node arrays returned confident
   nonsense (`age`, never declared categorical, with all splits flagged
   `is_categorical`). All such counts were **withdrawn** and replaced with
   permutation importance and ablation, which measure the model from the outside.
   *An explanation resting on a measurement you cannot verify is worse than no
   explanation, because it reads as authoritative.*

**A stopping rule was set in advance and honoured.** The self-paced loop ran
under "three consecutive iterations without a gain whose CI excludes zero and the
search stops". Iterations 3, 4 and 5 were nulls; the search stopped. Each null
came with a measurement explaining *why* it was null, and together they compose
into one mechanism — **getting stronger means getting more similar**
(Spearman(solo, ρ) = +0.886 across the HistGBM family), which is why the blend
saturated and why gains transfer at 2–30% rather than 100%.

---

## 17. Repository conventions

- **Competition data is never committed.** `data/` and `submissions/` are
  gitignored per Kaggle redistribution rules. `experiments/log.csv` and
  `experiments/blends.csv` are the tracked exceptions.
- **One commit per phase, minimum**, with a descriptive message. The incremental
  history is itself evidence the work was real.
- **Every phase gets a report under `reports/`**, generated or hand-written, with
  the hypothesis and the falsifier stated *before* the result.
- **Warning filters are narrow and documented at the point of use.** The
  `matmul` RuntimeWarning from numpy 2.x against Apple's Accelerate BLAS is
  spurious — reproduced on `rng.standard_normal((100000, 29)) @ w` with no NaN,
  inf or rank deficiency anywhere — and is silenced with the reasoning recorded
  inline so it is not mistaken for a fix later. Chasing it did surface two real
  preprocessing bugs.
- **No result is written down until it has been measured.** Not in the README,
  not in a report, not on a resume.

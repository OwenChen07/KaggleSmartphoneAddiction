# Kaggle Playground S6E8 — Predicting Smartphone Addiction

Tabular binary classification on 691,369 rows. The point of this repo is not the
score — it is the **validation methodology**: a fixed-fold CV harness where every
experiment is a paired comparison against every other, and where a claimed
improvement has to survive a bootstrap confidence interval before it gets
believed.

- **Task:** predict `addicted_label` (binary), metric **ROC AUC**
- **Data:** 691,369 train / 296,302 test rows; 9 numeric + 3 categorical features
- **Best OOF AUC:** **0.96726** (run `022` — target encoding + decimal lattice + imputation)
- **Public LB:** **0.96905** (run `022`), rank **827 of 2,982**

> Every model number below is out-of-fold cross-validation on the training set.
> `public_lb` and `gap` in `experiments/log.csv` are filled in only after a
> submission has actually scored; rows without a submission stay blank. Nothing
> here is projected.

### The OOF-to-LB gap

| run | model | features | OOF AUC | public LB | gap |
|---|---|---|---|---|---|
| `002` | HistGBM, numeric only | 9 | 0.954677 | 0.95576 | −0.001083 |
| `005` | HistGBM, numeric + categorical | 12 | 0.954704 | 0.95578 | −0.001076 |
| `008` | HistGBM + engineered features | 27 | 0.956091 | 0.95708 | −0.000989 |
| `012` | **tuned** HistGBM + engineered | 27 | 0.963192 | 0.96511 | −0.001918 |
| `017` | tuned, 5-seed average | 27 | 0.964060 | 0.96540 | −0.001340 |
| `019` | tuned + imputed columns (augment) | 32 | 0.963913 | **0.96607** | −0.002157 |
| `020` | imputed + 5-seed average | 32 | 0.964709 | 0.96610 | −0.001391 |
| `022` | **target encoding + lattice + imputation** | 69 | **0.967257** | **0.96905** | −0.001793 |

**This table previously claimed the gap was "a stable property of the
pipeline, not a per-model accident". That claim is now falsified, and the
correction is the most useful thing on this page.**

Across the first three submissions the gap held between −0.00099 and −0.00108
— three models, agreeing to about a ten-thousandth. Then the tuned model
nearly doubled it, to −0.001918.

The gap has two measured components, and neither is leakage.

**1. Fold-averaging, and it scales with model variance.** Test predictions are
the mean of all 5 fold models; each OOF row is scored by the single model that
did not train on it. Averaging reduces variance, so the leaderboard should sit
*above* OOF — the gap should be negative, as it is. Measured directly on an
80/20 same-population split, comparing on identical rows:

| model | single model | 5-model average | ensemble gain |
|---|---|---|---|
| default (`005`) | 0.953864 | 0.954453 | +0.000590 |
| tuned (`012`) | 0.962207 | 0.963220 | **+0.001012** |

The tuned model runs 69 leaves and 463 iterations against 31 and 100. More
capacity, more variance, more to gain from averaging — so a bigger gap. **The
gap tracks the model's variance, not the pipeline.**

**2. Covariate shift: the test set is genuinely easier.** Adversarial
validation (Phase 5) found train and test separable at AUC 0.565, entirely
because their *missingness rates* differ. Test has 2.55 points more
fully-observed rows, which score 0.963, and fewer rows missing all three top
features, which score 0.805. Reweighting the OOF rows by the adversarial
model's density ratio predicts +0.00124 for `005` and +0.00120 for `008`.

**The two components do not cleanly add**, and this is stated rather than
smoothed over. For run `012`, +0.001012 + 0.000931 = +0.001943 against an
observed +0.001918 — an almost exact match. For run `005`, +0.000590 +
0.001244 = +0.001834 against an observed +0.001076 — a substantial
over-prediction. A decomposition that works for one model and not another is
not yet understood. The public leaderboard is also scored on an unknown subset
of test rows, and AUC is a rank statistic that has no reason to be linear in
the row mix.

**Forward predictions, recorded before submitting.** Both submissions this
phase had their leaderboard scores predicted in advance:

| run | predicted (covariate shift) | predicted (gap holds) | actual | error |
|---|---|---|---|---|
| `008` | 0.95729 | 0.95717 | **0.95708** | +0.00021 / +0.00009 |
| `012` | 0.96412 | 0.96424 | **0.96511** | −0.00099 / −0.00087 |

Run `008` was called to within a ten-thousandth. Run `012` was **under**-predicted
by ~0.0009 by both methods — which is exactly what flagged the capacity effect
above, since both predictors assumed the old gap regime.

What has not changed: at no point does OOF *overstate* leaderboard
performance. That is the signature of leakage or overfitting, and there is
still no trace of it.

---

## Results

Full 5-fold stratified CV on all 691,369 rows. Identical folds for every row of
the table (`SEED = 42`), so the comparisons are paired.

| run | model | features | OOF AUC | fold std | fit (s) |
|---|---|---|---|---|---|
| `016` | tuned + external source data | 27 | 0.96335 | 0.00044 | 64 |
| `012` | **tuned HistGBM + engineered** *(model of record)* | 27 | **0.96319** | 0.00047 | 65 |
| `014` | tuned, search runner-up #3 | 27 | 0.96296 | 0.00051 | 69 |
| `013` | tuned, search runner-up #2 | 27 | 0.96258 | 0.00051 | 128 |
| `008` | HistGBM + 15 engineered features | 27 | 0.95609 | 0.00068 | 26 |
| `009` | engineered, pruned to 5 survivors | 17 | 0.95602 | 0.00078 | 22 |
| `011` | engineered minus the control feature | 26 | 0.95591 | 0.00072 | 23 |
| `015` | engineered + external source data | 27 | 0.95587 | 0.00081 | 23 |
| `005` | HistGBM, numeric + categorical | 12 | 0.95470 | 0.00078 | 21 |
| `006` | HistGBM + is-missing indicators | 21 | 0.95470 | 0.00078 | 25 |
| `010` | HistGBM, native categorical splits | 12 | 0.95470 | 0.00072 | 20 |
| `002` | HistGBM, numeric only (Phase 0) | 9 | 0.95468 | 0.00080 | 12 |
| `004` | Random forest (200 trees, leaf ≥ 50) | 12 | 0.94059 | 0.00064 | 176 |
| `003` | Logistic regression | 26 | 0.91379 | 0.00077 | 6 |
| `007` | is-missing pattern only | 12 | 0.50038 | 0.00074 | 3 |
| `001` | Dummy (class prior) | 12 | 0.50000 | 0.00000 | 2 |

Best blend, not a logged run: rank-average of `012`+`014` scores **0.963472**
(+0.000281 over `012`, CI `[+0.000245, +0.000321]`).

Reproduce with `python -m src.experiment --model all`.

At 691k rows the fold-to-fold spread is 0.0005–0.0008. The gap between the tree
ensembles and logistic regression (0.049) is ~60× that noise floor; tuning
(+0.0071) is ~15×; the engineered features (+0.0014) are ~2×. Everything else
in the table sits inside it.

## Method

`src/validation.py` is the whole argument of this project.

- **One fold assignment, globally.** `StratifiedKFold(shuffle=True,
  random_state=42)` is used by every experiment, so any two runs produce
  row-aligned OOF vectors and can be compared directly without refitting.
- **`clone()` per fold, preprocessing inside the pipeline.** No transformer is
  ever fit on data that includes the rows it will be scored on.
- **OOF AUC computed once over all rows**, not averaged across folds — and
  reported alongside the per-fold std, which is the noise floor any claimed
  improvement has to clear.
- **Paired bootstrap on the difference** (`src/compare.py`). Both models are
  resampled on the *same* bootstrap indices so shared row-level noise cancels.

```
$ python -m src.compare 004 005
A = 004 (rf: Random forest)          AUC 0.94059
B = 005 (histgbm)                    AUC 0.95470
  difference   +0.01411
  95% CI       [+0.01389, +0.01431]  -> B is better; interval excludes zero.
```

## What didn't work

Kept deliberately, because the negative results were more informative than the
positive ones.

**Explicit is-missing indicators added to HistGBM: exactly zero effect.** Not
"a small effect" — the OOF predictions were **bit-identical** (`np.array_equal`
→ `True`, max abs difference `0.0`). Bit-identical predictions are already
conclusive on their own: whatever the indicators did, it was nothing.

> **Correction.** This section previously also claimed the model "never once
> split on an indicator column", from walking the fitted trees' internal node
> arrays. That introspection was later found to be misread — see
> [`reports/native-categoricals.md`](reports/native-categoricals.md) — so the
> split counts have been withdrawn. The bit-identical OOF vectors are
> independent of it and stand.

The mechanism is still clear: HistGBM already learns a default routing direction for NaN natively,
so an is-missing indicator is information it has by construction and the split
offers zero additional gain. The 9 extra columns cost 4 seconds and bought
nothing.

**Missingness carries no signal at all.** Missing rates are high and uneven
(4.2% on `age` to 19.4% on `social_media_hours`), which looked like an
opportunity. A model trained on *nothing but* the 12 is-missing flags scores
**0.50038** — indistinguishable from chance. Target rate among missing vs
present rows differs by at most 0.0042 against a base rate of 0.7094. The data
is missing at random; the synthetic generator evidently masked values
independently of the label.

**The three categorical features are worth ~0.00003 AUC.** Dropping them
entirely (`baseline`, 9 features) scores 0.95468 against 0.95470 with them.
Notably the bootstrap CI is `[+0.00002, +0.00004]` — it *excludes zero*, so the
gain is statistically real. It is also completely irrelevant. With 691k rows
there is enough power to resolve differences three decimal places below
anything that matters, which is a good demonstration that "statistically
significant" and "worth having" are separate questions.

### Phases 4–6

**Pruning to the features that "survived" importance testing made it worse.**
Permutation importance on validation rows endorsed 5 of the 15 engineered
features and put the other 10 at or below 4e-5. Dropping those 10 cost
**−0.00007**, CI `[-0.00011, -0.00002]` — the interval excludes zero, so the
loss is real. Each of the ten measured ≈0 *with all the others present*;
collectively they were worth something. Permutation importance is a ranking
aid, not a pruning rule.

**A control feature that provably could not help, helped.** `sleep_deficit =
8 - sleep_hours` was added expressly to be worthless: a tree splits on
`x <= t`, so a monotonic transform of one column should give the identical
partition. Ablating it cost **+0.00018**, CI `[+0.00015, +0.00022]`. Running
it down: an *exact* copy of `sleep_hours` changes the OOF **bit-identically**
(CI `[0, 0]`), a `3.7×` rescale likewise, and only the order-**reversing**
`8 − x` does anything (+0.000191). `x <= t` sends the boundary tie-group left;
`8 − x <= 8 − t` sends it right, so where a threshold falls inside a group of
binned or tied values the two columns express partitions the other cannot
reach. HistGBM quantises to 255 bins and `sleep_hours` has 451 distinct
values, so 144 of its 241 occupied bins straddle a boundary in the reversed
column. **The original claim is true for increasing transforms and false for
decreasing ones.**

**Native categorical support: exactly nothing.** Declaring the categoricals to
HistGBM via `categorical_features` instead of ordinal-encoding them gives OOF
0.954704 — identical to run `005` at six decimals, CI `[-0.00002, +0.00002]`.
Permutation importance puts all three columns at +0.000000 under ordinal
encoding and +0.000003 under native. The encoding was never the binding
constraint: the categoricals are worth +0.00003 in total, and removing an
obstacle only helps if something was waiting behind it.

**Every blend with a weaker model lost.** None of the 26 subsets of
`{012, 008, 004, 005, 009}` beat the tuned model alone, and sweeping the blend
weight climbs monotonically toward "don't blend". The partners were both
highly rank-correlated with the tuned model (Spearman 0.985) *and* strictly
weaker. Blending averages skill as well as errors — **diversity only pays
among comparably strong models**. Re-measuring the search's 2nd and 3rd
candidates as runs `013`/`014` (comparable strength, different shapes) made
blending work: `012`+`014` gives +0.000281, CI `[+0.000245, +0.000321]`. Still
0.59× the fold noise.

**The original source dataset was not worth adding.** Found it (7,500 rows,
exact schema match, base rate 0.7077 vs 0.7094). It has **zero** missing values
against 4–19% in the competition data, and KS statistics of 0.24–0.26 on three
activity columns — two orders of magnitude further from the training data than
the *test set* is (KS ≈ 0.002). Appending it to training folds only: **−0.00023**
on the Phase 4 model, **+0.00016** on the tuned model. Opposite signs, both
significant, both negligible. 7,500 rows is 1.1% of 691,369.

**Median imputation does not erase missingness — it encodes it.** A
"values-only" adversarial control that median-imputed still scored 0.561
against 0.565 for the full discriminator, because every imputed cell lands on
exactly the median and the classifier counts the spike. Complete-case analysis
(0.499) is what actually removes the channel. Kept as a worked example of a
control that does not control for what it claims.

**A mechanism check that produced confident nonsense.** Walking the fitted
trees' node arrays to count splits per feature returned `age` — never declared
categorical — with all its splits flagged `is_categorical`, and `gender` with
none, against a correct `is_categorical_` attribute. The readings were
misaligned. All such counts were withdrawn from the reports and replaced with
permutation importance and ablation, which measure the model from the outside.
An explanation resting on a measurement you cannot verify is worse than no
explanation, because it reads as authoritative.

**An unpinned dependency silently moved every number.** Regenerating run
`005`'s OOF gave 0.954815 against a logged 0.954704. The harness was fine:
scikit-learn 1.5.2, 1.6.1, 1.7.0 and 1.8.0 all reproduce the logged values to
six decimals, and only **1.9.0** diverges — it changed histogram bin placement,
so `n_bins_non_missing_` goes from `[18,255,255,255,255,255,231,166,255]` to
`[18,254,253,246,255,242,231,166,255]` on the same data and seed. Move the bin
edges and every candidate split moves. `requirements.txt` now pins
`>=1.5,<1.9`. **A comparison is only paired if both sides came from the same
environment** — fixing the seed is not enough.

**Random forest: 8× the fit time of HistGBM for 0.014 less AUC.** Kept in the
table as a measured data point rather than quietly dropped.

**One phantom bug, chased and dismissed.** The logistic runs emitted
`RuntimeWarning: divide by zero encountered in matmul` from inside the lbfgs
solver. The design matrix was clean (no NaN, no inf, no zero-variance column),
and the same warning reproduces on `rng.standard_normal((100000, 29)) @ w` —
it is a spurious floating-point flag from numpy 2.x built against Apple's
Accelerate BLAS, not a numerical problem. Silenced narrowly, with the reasoning
recorded at `src/experiment.py` so it is not mistaken for a fix later.

Two **real** bugs did surface while investigating it, both in my own
preprocessing: the indicator block emitted column names identical to its source
columns (breaking `get_feature_names_out`), and indicators over categoricals
were exact duplicates of the `__MISSING__` one-hot level — perfectly collinear.
Fixed in `src/preprocessing.py`; the logistic score was unchanged at 0.91379,
since L2 had been absorbing it.

## Data

Generated by `python -m src.eda` → [`reports/eda.md`](reports/eda.md).

Three features carry nearly all of the signal on their own:

| feature | univariate AUC |
|---|---|
| `daily_screen_time_hours` | 0.88955 |
| `weekend_screen_time` | 0.88099 |
| `social_media_hours` | 0.85778 |
| `work_study_hours` | 0.65488 |
| `gaming_hours` | 0.62196 |

Base rate is 0.7094 positive. `sample_submission.csv` ships a constant
0.7094243, i.e. exactly the base rate — which scores 0.5.

## Layout

```
src/data.py           loading + column definitions; NaN preserved
src/preprocessing.py  ColumnTransformer builders, returned unfitted
src/validation.py     run_cv, experiment log, paired bootstrap
src/models.py         model zoo — add a function + a dict entry
src/experiment.py     CLI runner
src/compare.py        paired bootstrap between two logged runs
src/eda.py            descriptive statistics -> reports/eda.md
src/features.py       engineered features, stateless, fit inside the fold
src/importance.py     permutation importance on validation rows
src/adversarial.py    train-vs-test discriminator
src/tuning.py         RandomizedSearchCV + full-data re-measurement
src/blend.py          rank-average blending over logged runs
experiments/log.csv   tracked record of every full-data run
experiments/oof/      OOF vectors, row-aligned across runs (gitignored)
```

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m src.experiment --model all          # full bake-off
python -m src.experiment --model histgbm --sample 50000   # fast iteration
python -m src.compare 002 005                 # is the difference real?
python -m src.importance --model histgbm_fe   # what is each feature worth?
python -m src.adversarial                     # is the train/test split clean?
python -m src.tuning --n-iter 40 --search-rows 250000
python -m src.blend 012 014                   # rank-average two runs
```

`experiments/oof/` is gitignored, so a fresh clone has the log but not the
vectors `src/compare.py` needs. Regenerate one without adding a log row:

```bash
python -m src.experiment --model histgbm --oof-as 005 --no-submission
```

It refuses to write unless the reproduced OOF AUC matches the logged value,
which doubles as a determinism check on the environment.

Sampled runs are refused entry to `experiments/log.csv` — a row in the log
always means a full-data run.

## Next

- [x] Gap tracking established — and then **falsified**: the gap is not a
      stable pipeline property, it scales with model variance
- [x] Feature engineering — ratios/interactions, validated with
      `permutation_importance` on validation folds (+0.0014)
- [x] Declare categoricals natively to HistGBM (no effect)
- [x] Adversarial validation — 0.565, entirely missingness
- [x] `RandomizedSearchCV` tuning (+0.0071), then a `rankdata` rank-average
      blend (+0.00028, below the noise floor)
- [x] Test concatenating the original source dataset (negligible, sign-flips
      by model)
- [ ] LightGBM / XGBoost through the same harness — **not attempted**; neither
      is in `requirements.txt` and adding a dependency was out of scope for
      this phase
- [ ] Resolve why the two gap components sum correctly for run `012` but
      over-predict for run `005`
- [ ] Sweep `max_leaf_nodes` with and without external data, to test the
      capacity explanation for its sign flip

## Reports

| report | what it covers |
|---|---|
| [`reports/learning-notes.md`](reports/learning-notes.md) | **start here** — the concepts, explained from scratch |
| [`reports/feature-engineering.md`](reports/feature-engineering.md) | Phase 4: which features survived and why |
| [`reports/adversarial-validation.md`](reports/adversarial-validation.md) | Phase 5: train/test shift and what it means for OOF |
| [`reports/tuning-and-ensembling.md`](reports/tuning-and-ensembling.md) | Phase 6: search space, winning params, blending |
| [`reports/native-categoricals.md`](reports/native-categoricals.md) | Side experiment A |
| [`reports/external-data.md`](reports/external-data.md) | Side experiment B |
| [`reports/eda.md`](reports/eda.md) | descriptive statistics |

# Kaggle Playground S6E8 — Predicting Smartphone Addiction

Tabular binary classification on 691,369 rows. The point of this repo is not the
score — it is the **validation methodology**: a fixed-fold CV harness where every
experiment is a paired comparison against every other, and where a claimed
improvement has to survive a bootstrap confidence interval before it gets
believed.

- **Task:** predict `addicted_label` (binary), metric **ROC AUC**
- **Data:** 691,369 train / 296,302 test rows; 9 numeric + 3 categorical features
- **Best OOF AUC so far:** **0.95470** (HistGradientBoostingClassifier, 5-fold)
- **First public LB score:** **0.95578** (run `005`) — an OOF-to-LB gap of **-0.00108**

> Every model number below is out-of-fold cross-validation on the training set.
> `public_lb` and `gap` in `experiments/log.csv` are filled in only after a
> submission has actually scored; rows without a submission stay blank. Nothing
> here is projected.

### The gap so far

| run | model | OOF AUC | public LB | gap |
|---|---|---|---|---|
| `005` | HistGBM, numeric + categorical | 0.954704 | 0.95578 | **-0.00108** |

The gap is *negative* — the leaderboard scores slightly **better** than
cross-validation. That is the direction you want, and it has a structural
cause rather than being luck: test predictions are the mean of all five fold
models, so each test row is scored by a 5-model ensemble, while each OOF row is
scored by the single model that did not train on it. Averaging five models
reduces variance, so the test score should sit a little above OOF.

Whether -0.00108 is even distinguishable from noise depends on how much of the
296,302-row test set is in the public split. Resampling the OOF predictions at
plausible split sizes puts the sampling noise of a single AUC estimate at
sd = 0.00073 for a 20% public split (±0.00146 at 2sd) and sd = 0.00047 for a
50% split (±0.00093). So at 20% the gap is inside the noise band entirely; at
50% it is marginally outside it. Either way there is no sign of the failure
this column exists to detect — OOF badly overstating true performance.

---

## Results

Full 5-fold stratified CV on all 691,369 rows. Identical folds for every row of
the table (`SEED = 42`), so the comparisons are paired.

| model | features | OOF AUC | fold std | fit (s) |
|---|---|---|---|---|
| `histgbm` — HistGBM, numeric + categorical | 12 | **0.95470** | 0.00078 | 21 |
| `histgbm_isna` — HistGBM + is-missing indicators | 21 | 0.95470 | 0.00078 | 25 |
| `baseline` — HistGBM, numeric only (Phase 0) | 9 | 0.95468 | 0.00080 | 12 |
| `rf` — Random forest (200 trees, leaf ≥ 50) | 12 | 0.94059 | 0.00064 | 176 |
| `logistic` — impute + scale + one-hot | 26 | 0.91379 | 0.00077 | 6 |
| `missingness_only` — is-missing pattern only | 12 | 0.50038 | 0.00074 | 3 |
| `dummy` — predicts the class prior | 12 | 0.50000 | 0.00000 | 2 |

Reproduce with `python -m src.experiment --model all`.

At 691k rows the fold-to-fold spread is ~0.0008, so the gap between the tree
ensembles and logistic regression (0.041) is roughly 50× the noise floor, while
the gap between the top three rows is well inside it.

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
→ `True`, max abs difference `0.0`). Inspecting the fitted trees explains why:
across every tree in the ensemble, the model **never once split on an indicator
column**. HistGBM already learns a default routing direction for NaN natively,
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
experiments/log.csv   tracked record of every full-data run
experiments/oof/      OOF vectors, row-aligned across runs
```

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m src.experiment --model all          # full bake-off
python -m src.experiment --model histgbm --sample 50000   # fast iteration
python -m src.compare 002 005                 # is the difference real?
```

Sampled runs are refused entry to `experiments/log.csv` — a row in the log
always means a full-data run.

## Next

Phases 4–6 are deliberately not done yet; the harness exists so each is a
drop-in experiment rather than a rewrite.

- [x] First submission recorded (run `005`, gap -0.00108); more runs needed before the gap is a trend rather than a point
- [ ] Feature engineering — ratios/interactions on the screen-time columns,
      validated with `permutation_importance` on validation folds
- [ ] Declare categoricals natively to HistGBM via `categorical_features`
      rather than ordinal-encoding them
- [ ] Adversarial validation — train/test discriminator, AUC ≈ 0.5 means a clean split
- [ ] LightGBM / XGBoost through the same harness
- [ ] `RandomizedSearchCV` tuning, then a `rankdata` rank-average blend
- [ ] Test concatenating the original source dataset as an ablation

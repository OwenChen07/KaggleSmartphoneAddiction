# Side experiment B — the original source dataset

**Found it. Adding it changes almost nothing: −0.00023 on the Phase 4 model,
+0.00016 on the tuned model. Both intervals exclude zero, both are far below
the fold noise, and they point in opposite directions.**

## Finding it

Playground competitions are generated from a real dataset, and concatenating
that original to the training data is standard and permitted. `kaggle datasets
list -s "smartphone addiction"` returns several candidates, four of which are
byte-identical in size (182,295) and are re-uploads of one another.

Used: [`jayjoshi37/smartphone-usage-and-addiction-prediction`](https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction)
— 7,500 rows, `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`.

The schema matches exactly. All nine numeric columns, all three categoricals,
and `addicted_label` are present under identical names. Base rate 0.7077
against the competition's 0.7094. This is the source.

### One column that must stay dropped

The file also carries `addiction_level`, which determines the target
**exactly**:

| `addiction_level` | label 0 | label 1 |
|---|---|---|
| Mild | 1,373 | 0 |
| Moderate | 0 | 2,874 |
| Severe | 0 | 2,434 |
| (missing) | \- | 819 rows |

Using it as a feature would be handing the model the answer. `FEATURE_COLS`
excludes it by construction, so this is already safe — noted in
`src/data.py:load_external` because it is exactly the kind of plausible-looking
column someone adds back after reading the CSV header.

## How it differs from the competition data

Two differences matter, and both are large.

**The original has no missing values at all.** Zero, in any column. The
competition data is 4–19% missing per column. So every appended row is a
fully-observed row, and Phase 5 established that the missingness *rate* is the
one axis on which the competition's own train and test files differ.

**Three columns are distributionally shifted**, by far more than train and
test differ from each other:

| column | external mean | competition mean | KS statistic |
|---|---|---|---|
| `work_study_hours` | 3.242 | 2.367 | **0.261** |
| `social_media_hours` | 3.273 | 2.471 | **0.242** |
| `gaming_hours` | 2.014 | 1.459 | **0.240** |
| `notifications_per_day` | 134.257 | 145.895 | 0.083 |
| `app_opens_per_day` | 97.832 | 102.637 | 0.055 |
| `weekend_screen_time` | 9.244 | 9.480 | 0.045 |
| `daily_screen_time_hours` | 7.500 | 7.641 | 0.043 |
| `sleep_hours` | 6.738 | 6.804 | 0.039 |
| `age` | 26.569 | 26.615 | 0.007 |

For scale: the adversarial test in Phase 5 found train and test KS statistics
of about **0.002**. The original dataset is two orders of magnitude further
from the competition training data than the test set is. Ranges are clipped
differently too — external `social_media_hours` runs 0.50–6.00 against
0.00–8.00 in the competition data.

The generator did not merely resample the original; it reshaped these
marginals.

## How it was tested

`run_cv` gained `extra_X`/`extra_y`, which append rows to **every fold's
training set only**. The external rows never enter a validation fold, so the
OOF vector still covers exactly the competition rows in the same order and
stays directly comparable to every other run in the log. Putting external rows
into validation would change what the AUC is measured over and quietly break
every paired comparison in the project.

## Result

| run | model | external? | OOF AUC | fold std |
|---|---|---|---|---|
| `008` | Phase 4 defaults | no | 0.956091 | 0.000680 |
| `015` | Phase 4 defaults | **yes** | 0.955865 | 0.000810 |
| `012` | tuned | no | 0.963192 | 0.000474 |
| `016` | tuned | **yes** | 0.963349 | 0.000443 |

```
compare 008 015 :  difference -0.00023   95% CI [-0.00028, -0.00017]  -> WORSE
compare 012 016 :  difference +0.00016   95% CI [+0.00008, +0.00025]  -> better
```

**Opposite signs, both significant, both negligible.** The Phase 4 model is
hurt by the extra rows; the tuned model is helped. Neither effect reaches even
half the fold-to-fold noise of the model it applies to.

## Reading it

7,500 rows is **1.1%** of the 691,369 training rows. Even perfectly matched
data at that proportion could not move a 691k-row fit much. These rows are not
perfectly matched — they are drawn from visibly different marginals on the
three activity columns and carry no missingness — so what little weight they
have is partly pulling in the wrong direction.

The sign flip between the two models is consistent with that reading, though
this is interpretation rather than something measured here: the higher-capacity
tuned model (69 leaves, 463 iterations) has more room to accommodate rows from
a slightly different distribution as their own region of feature space, while
the 31-leaf default model has to average them in with everything else. A test
that would settle it — sweeping `max_leaf_nodes` with and without the external
rows — was not run.

**Conclusion: not worth using.** Not because external data is a bad idea in
general — on a competition with 5,000 training rows, 7,500 extra rows would be
transformative — but because the ratio here is 1.1% and the distributional
match is poor. The experiment is worth having run, because "we did not bother"
and "we tried it and it moves the fourth decimal place" are different states of
knowledge.

## Reproducing

```bash
kaggle datasets download -d jayjoshi37/smartphone-usage-and-addiction-prediction \
  --unzip -p data/external/jayjoshi
python -m src.experiment --model histgbm_fe --external --no-submission   # run 015
python -m src.compare 008 015
```

`data/` is gitignored, so the external file is not committed either — Kaggle
redistribution rules apply to it exactly as they do to the competition data.

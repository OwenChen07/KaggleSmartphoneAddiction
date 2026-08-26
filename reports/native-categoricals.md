# Side experiment A — native categorical support

**Result: no measurable difference. OOF AUC 0.954704 either way.**

| run | model | OOF AUC | fold std | fit (s) |
|---|---|---|---|---|
| `005` | HistGBM, categoricals ordinal-encoded | 0.954704 | 0.000782 | 21.0 |
| `010` | HistGBM, categoricals declared natively | 0.954704 | 0.000722 | 19.7 |

```
compare 005 010 :  difference +0.00000   95% CI [-0.00002, +0.00002]
                -> indistinguishable: the interval includes zero.
```

## What was being tested

Everywhere else in this project the three categorical columns reach the model
through `OrdinalEncoder`, which turns `gender` into `Female=0, Male=1,
Other=2, __MISSING__=3`. That imposes an ordering the categories do not have.
A tree can then only cut that axis into **contiguous** ranges, so a partition
like `{Male}` vs `{Female, Other}` costs two splits instead of one — and that
partition is the one the data suggests, since the target rates are Female
0.7038, **Male 0.7232**, Other 0.7010.

`HistGradientBoostingClassifier(categorical_features=...)` removes the
constraint: the split finder is told which columns are unordered sets and
partitions the levels directly. The encoder still runs — HistGBM wants small
non-negative integer codes, not strings — but the estimator now knows what
those codes mean.

The Phase 3 result made this worth testing. The categoricals were measured at
**+0.00003 AUC** in total, which is suspiciously close to nothing for three
columns with visibly different target rates. If the ordinal encoding were
throwing their value away, native declaration was where it would show up.

## What happened

It did not show up. The two OOF vectors are not identical — Spearman
correlation 0.9989, 154 of 691,369 rows differ by more than 0.1 — but the
difference is pure churn: AUC moves by 4×10⁻⁷.

Permutation importance on validation rows (fold 1, 3 repeats) confirms neither
model extracts anything from these columns:

| model | `gender` | `stress_level` | `academic_work_impact` | total |
|---|---|---|---|---|
| `005` ordinal | +0.000000 | +0.000000 | +0.000000 | **+0.000000** |
| `010` native | +0.000002 | +0.000002 | +0.000000 | **+0.000003** |

## Why

The encoding was never the binding constraint. The categoricals are worth
about +0.00003 AUC *in total*, so the most a better encoding could recover is
some fraction of a number that is already 25× below the fold-to-fold noise.
Removing an obstacle only helps if something was waiting behind it.

The target rates make this concrete. The largest spread within any of the
three columns is `gender`, at 0.7232 − 0.7010 = **0.022** against a base rate
of 0.7094. Against `daily_screen_time_hours`, which separates the classes to a
univariate AUC of 0.890 on its own, a 2-point shift in class balance is not
information a 100-iteration ensemble with 31 leaves per tree needs help
reaching. It has ample capacity to spend two splits instead of one where it
matters.

## Reading the negative result properly

This is not evidence that native categorical handling is a bad idea in
general. It is a real advantage on features with **many** levels, where the
number of contiguous cuts available to an ordinal encoding is a vanishing
fraction of the 2^(k−1) − 1 possible partitions. Here k = 4 (three levels plus
`__MISSING__`), so ordinal encoding already reaches 3 of the 7 possible
partitions, and the missing ones are worth nothing because the feature itself
is worth nothing.

The experiment cost 20 seconds and closes a "what if the encoding is wasting
them?" question that would otherwise stay open. That is the whole value of it.

## An attempted mechanism check that did not work

I tried to count how often each model actually splits on the categorical
columns, by walking `predictor.nodes` and reading `feature_idx` and
`is_categorical`. The output was self-contradictory — `age`, a column never
declared categorical, came back with every one of its splits flagged
`is_categorical`, while `gender` came back with none, even though
`clf.is_categorical_` correctly reports `[False ×9, True, True, True]`. I was
misreading the node struct layout.

Rather than publish counts I could not verify, the section above uses
permutation importance instead, which measures the model's behaviour from the
outside and needs no assumptions about private data structures. **Noted here
because the discarded numbers were interesting-looking and wrong, which is
exactly the kind of thing that ends up in a README if you do not write down
that you threw it away.**

## Reproducing

```bash
python -m src.experiment --model histgbm_native_cat    # run 010
python -m src.compare 005 010
```

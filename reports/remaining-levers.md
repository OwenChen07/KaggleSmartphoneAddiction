# What is left, and what to leave alone

Second research pass, 2026-08-28, after reaching LB 0.96948 (rank 767/2,982).
Source: the public notebook library for this competition, read via the Kaggle
API. **The two claims this report acts on were re-verified on our data.**

---

## The strategic finding: our position is the safe one

Georgy Mamarin published the **public and private** boards for all seven
finished Season 6 episodes — 26,345 team rows — so what the private split does
is measurable rather than folklore.

**Public top-10 teams, where they landed privately:**

| episode | private ranks of the public top 10 |
|---|---|
| S6E2 | 570, 1036, 589, 606, 593, 69, 580, 563, 577, 584 |
| S6E6 | 379, 79, 52, 327, 496, 485, 517, 412, 507, 401 |
| S6E7 | 440, 259, 643, 310, 282, 136, 211, 1075, 662, 602 |
| S6E3 | 1, 2, 3, 10, 15, 13, 29, 9, 4, 14 |
| S6E5 | 5, 2, 9, 6, 1, 3, 18, 10, 4, 121 |

Across all 70: **38.6% stayed in the private top 10, and 50% fell out of the
private top 50.** It is episode-dependent — E3 and E5 were stable — but when
it goes wrong it goes very wrong.

**Mid-pack is where the odds are good.** For teams around the 15–16% mark
(n = 1,052 comparable rows): median percentile shift **−0.4%**, **P(improve) =
69.9%**, middle 50% of outcomes between −2.1% and +0.2%.

We are at **rank 767 of 2,982 — the 25.7th percentile**. That is squarely in
the population that holds or improves. Much of that expected gain is
mechanical: we rise because teams selected on the public split fall past us.

The metric helps too — ROC AUC boards are markedly more stable than balanced
accuracy (46.7% vs 16.7% public-top-10 survival).

**Conclusion: holding an honest CV-driven submission is roughly a 70% bet to
hold or improve. Chasing the last 0.0003 into the 0.9712 population is
negative expected value.** Everything below is therefore optional, and the
correct amount of effort may be zero.

---

## A diagnostic that killed an idea in two minutes

Before target-encoding our engineered features, the question is whether their
target rate is **non-smooth in the value** — a GBM already captures anything
smooth by splitting on the number, and target encoding only pays where
neighbouring values disagree.

Ratio of adjacent-value target-rate jumps to what sampling noise alone
predicts, measured on our data. **The controls are the point** — without
columns whose answer we already know, the numbers are unreadable:

| column | kind | ratio |
|---|---|---|
| `app_opens_per_day` | **control: known lookup** | **18.1×** |
| `age` | **control: known lookup** | **17.6×** |
| `notifications_per_day` | **control: known lookup** | **15.0×** |
| `daily_screen_time_hours` | raw continuous | 1.6× |
| `weekend_ratio` | derived ratio | 1.5× |
| `social_share` / `gaming_share` / `work_share` | derived ratios | 1.0–1.1× |
| `waking_load` | derived | 0.5× |
| `residual_screen` | derived | 0.4× |
| `tracked_screen` | derived | 0.4× |
| `unaccounted_screen` | derived (identity) | 0.3× |

**Our derived features are smoother than noise.** They are deterministic
functions of the raw columns, so neighbouring values share structure by
construction. Target-encoding them would add nothing, and the whole family of
"encode the engineered features too" ideas dies here for two minutes of
compute.

Note the shape of the result: the three lookup columns sit at 15–18× and
everything else at 0.3–1.6×. That separation is only legible because the
controls are in the table.

---

## Levers worth trying, in order

**We ship a two-model blend, not a forty-member stack.** That changes the
ranking: several levers are large for a single model and vanish inside a dense
ensemble, and we are much closer to the single-model case. The published
figures below give both where they are known.

| # | lever | reported solo | reported stacked | why it fits us |
|---|---|---|---|---|
| 1 | **transductive frequency encoding** | **+0.00032** | +0.000015 | one line; we are near-solo so expect closer to the solo figure |
| 2 | **10 encoding folds instead of 5** | +0.0001 | — | one parameter in `TargetFrequencyEncoder` |
| 3 | **trig features on the two lookup columns** | **+0.00017** | +0.00002 | cheap; near-solo again |
| 4 | **logit stacking instead of rank-average** | — | +0.0004 | a stacker can assign *negative* weights; rank-average cannot |
| 5 | **10-fold CV, final run only** | +0.0002 | +0.00003 | breaks log comparability — production run, never for comparisons |

### 1. Transductive frequency encoding — the best of these

Our frequency columns count occurrences **within the training fold** (~553k
rows). But test *features* are given to us at prediction time; only test
*labels* are hidden. Counting across train **and** test (987k rows) is
therefore legitimate and is a strictly better estimate of how common a value
is.

**The target encoding must stay fold-based and nested** — that one uses labels
and must never see test. Only the counts change.

### 3. Trig features, and why they work despite adding no information

`sin(2πx/k)` and `cos(2πx/k)` on `notifications_per_day` and
`app_opens_per_day`. The published account is instructive: three careful
diagnostics all said there was no periodic structure, and all three were
right — and the feature still helped, because a split on `sin(2πx/20)` selects
a **union of disjoint intervals** of x. Against a lookup that jumps 0.22
between neighbours, one such split buys far more than target encoding's
one-split-per-value.

The lesson is one this project has now met twice: *"the model already has this
information" does not imply "the model can afford to use it."*

---

## Do not do these — all measured negative by others

| idea | delta | note |
|---|---|---|
| **pseudo-labeling confident test rows** | **−0.0034** | worst result in the published ledger |
| tree depth 9–13 | to −0.0011 | our re-tune already chose 8 |
| denoising-autoencoder features (768) | −0.00071 | −0.00014 even compressed to 32 PCs |
| pairwise target encoding | −0.0004 | |
| multi-resolution target encoding | −0.0003 | |
| monotone constraints on screen columns | −0.0003 | |
| rank/rank-gauss at the stacker instead of logits | −0.00013 | |
| kNN features from the source dataset | −0.00003 | 0.90 AUC standalone, still negative |
| target-encoding the derived features | — | **killed by our own diagnostic above** |
| concatenating the source dataset | −0.0001 | we measured −0.00023 / +0.00016 |
| more seed-averaging | — | we measured 4% transfer; a blend is already an averaging device |

The pseudo-labeling row is worth dwelling on: it is the idea most likely to
occur to someone at this stage, and it is the single worst thing in the
ledger.

---

## Independent confirmations of our own findings

Five things this project derived on its own appear in the published ledger
with the same sign:

| ours | theirs |
|---|---|
| imputation must augment, not replace | ✅ same, same reasoning |
| NA-indicator features are worthless (0.50038 from flags alone) | −0.00001, "MCAR" |
| source dataset not worth adding | −0.0001 |
| seed-averaging is nearly dead once you average anything else | +0.00013 solo → +0.000013 stacked, "a stack is already an averaging device" |
| the CV→LB offset is not a constant and tracks variance | ✅ same, and they refit it as the pipeline grew |

---

## Recommendation

**The honest answer is that the work is close to done.** We are in the
percentile band that historically holds or improves, on the metric family that
is most stable, with a submission chosen by CV rather than by leaderboard
probing.

If you want to spend more: items 1–3 are perhaps **+0.0005 combined** for
under an hour, and none of them risks anything. Item 5 is a final production
run. Everything else on the "do not" list has been measured negative by
someone who ran it properly.

What would *not* be worth doing is another representation hunt. The last one
found the lookup keys and was worth +0.0039; there is no reason to expect a
second of that size, and the ledger above is the record of forty-odd people
not finding one.

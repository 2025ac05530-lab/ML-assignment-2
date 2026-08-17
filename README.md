
# Machine Learning Assignment 2
Name: **Shruti Basavaraj Desai**
BITS ID: **2025AC05530**

## a. Problem Statement

Given a credit card client's demographic profile, credit limit, and six months of
billing history — repayment status, bill statement amounts and payment amounts
for each month — predict whether that client will **default on their payment in
the following month**.

This is a supervised binary classification task over 30,000 labelled client
records, framed from the lender's point of view. The six months of history all
precede the month being predicted, so the model is forecasting forward rather
than explaining a contemporaneous outcome. The practical use is early risk
flagging: a client predicted to default can be routed to collections outreach,
a credit limit review, or a restructured payment plan before the missed payment
occurs.

**Why accuracy is the wrong headline metric.** The two error types carry very
different costs. A false negative — a defaulter the model fails to flag — means
an unrecovered balance, potentially the full credit limit. A false positive — a
paying client the model flags — costs an outreach attempt and some goodwill,
and if it triggers a limit reduction it risks driving away a profitable customer,
but the loss is bounded and much smaller. On top of this asymmetry, only 22.12%
of clients default, so a model that predicts "no default" for every single client
scores 77.88% accuracy while catching zero defaulters and being useless.

That imbalance also compresses the accuracy column into uselessness as a
discriminator: four of the five models trained here land within a 2.5-point band
(0.7928–0.8173), while their recall spans 0.2442 to 0.6654 — a spread roughly
17× wider. Accuracy is reported for completeness, but the models are judged on:

| Metric | Why it is used here |
|---|---|
| **Recall** | Directly measures the expensive error — what fraction of real defaulters were caught |
| **MCC** | Balanced single number that stays honest under class imbalance, unlike accuracy or F1 |
| **AUC** | Threshold-independent quality of the risk *ranking*, which is what a lender actually needs in order to choose an operating point |
| Precision / F1 | Reported to expose the cost of high recall, and to show where models trade one error for the other |

Because the ratio of the two error costs is a business input rather than a
property of the data, section (e) states that ratio explicitly and shows which
model minimises expected cost in each regime.

**Note on the feature set.** `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are retained
because the assignment is a modelling exercise on the dataset as published. In a
real credit deployment these are protected attributes with legal restrictions in
many jurisdictions, and a fairness audit would be required before use.


## b. Dataset Description

- **Name:** Default of Credit Card Clients Dataset
- **Kaggle URL:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- **Original source:** UCI Machine Learning Repository — Taiwan, 2005
- **Shape:** 30,000 × 25
- **Features used:** 23 raw predictors → 26 model features after one-hot encoding
- **Target variable:** `default_payment_next_month`
- **Task type:** Binary classification

**Classes:**
| Value | Meaning | Count | % |
|---|---|---|---|
| 0 | No default next month | 23,364 | 77.88% |
| 1 | Defaulted next month | 6,636 | 22.12% |

A 3.52:1 imbalance. Not severe enough to require resampling or class weighting,
but severe enough that the majority-class baseline (77.88% accuracy) has to be
quoted alongside every accuracy figure — see section (d).

**Feature groups:**
| Group | Columns | Description |
|---|---|---|
| Demographic | LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE | Credit limit and client attributes |
| Repayment history | PAY_0, PAY_2–PAY_6 | Payment status each month; observed range -2 to 8, where negative = paid off / no balance and positive = months delayed (see data quality note) |
| Bill amounts | BILL_AMT1–BILL_AMT6 | Monthly bill statement amounts (NT$) |
| Payment amounts | PAY_AMT1–PAY_AMT6 | Amount paid each month (NT$) |

Note: the repayment columns are labelled PAY_0, PAY_2…PAY_6 — there is no PAY_1
in the original dataset. Retained as-is rather than renamed.

**Missing values:** None — `isnull().sum()` returns 0 for all 25 columns across all 30,000 rows.

**Data quality observations:**
- **Undocumented category codes in `EDUCATION`.** The data dictionary defines
  1 = graduate school, 2 = university, 3 = high school, 4 = others. The data also
  contains 0 (14 rows), 5 (280 rows) and 6 (51 rows), which are unlabelled.
  All three were folded into category 4 ("others"), 345 rows in total, taking
  that category from 123 to 468 — about 1.6% of the dataset.
- **Undocumented category code in `MARRIAGE`.** Defined values are 1 = married,
  2 = single, 3 = others; the data contains 54 rows coded 0. These were folded
  into category 3 ("others"), taking it from 323 to 377.
- **`PAY_*` contains -2 and 0**, neither of which appears in the official
  data dictionary (which lists -1 = paid duly and 1–9 = months of delay). These
  are commonly read as "no consumption" and "revolving credit / paid minimum".
  They were left untouched, since the columns are treated as ordinal severity
  scores where more negative = healthier. The observed maximum is 8, not the 9
  the dictionary allows for.
- **Negative `BILL_AMT` values** appear in the bill columns — the minima for
  `BILL_AMT4`, `BILL_AMT5` and `BILL_AMT6` are -170,000, -81,334 and -339,603
  respectively. These represent overpayments or credit balances rather than data
  errors, and were left as-is.
- **Heavy right skew in the monetary columns.** `PAY_AMT2` ranges from 0 to
  1,684,259 against a median of 2,009 — the mean sits far above the median in
  every amount column. This matters most for the distance-based and Gaussian
  models (kNN and Naive Bayes), whose assumptions are sensitive to it.
- **`AGE`** spans 21–79 with a median of 34, so the client base skews young.

**Preprocessing applied:**
- Dropped `ID` — a row identifier with no predictive value
- Renamed target column to remove dots for easier handling
  (`default.payment.next.month` → `default_payment_next_month`)
- Consolidated the undocumented `EDUCATION` and `MARRIAGE` codes into the
  existing "others" categories, as described above
- SEX / EDUCATION / MARRIAGE are categorical concepts stored as integers.
  Decision: **`EDUCATION` and `MARRIAGE` were one-hot encoded** (`drop_first=True`,
  giving `EDUCATION_2/3/4` and `MARRIAGE_2/3`, with `EDUCATION_1` and
  `MARRIAGE_1` held out as reference categories to avoid the dummy variable
  trap) because their integer codes carry no meaningful ordering — leaving them
  as integers would tell the linear and distance-based models that "high school"
  (3) sits exactly twice as far from "graduate school" (1) as "university" (2)
  does, which is not a real relationship.
  **`SEX` was retained as an integer** because it is already binary (1/2);
  one-hot encoding it with `drop_first=True` would yield `SEX_2 = SEX - 1`, an
  affine shift of the original column that StandardScaler removes anyway.
  This took the feature count from 23 to 26.
- StandardScaler applied inside a Pipeline, fit on training data only, to prevent
  test-set information leaking into the scaling parameters
- 80/20 stratified train-test split, `random_state=42` for reproducibility
  (24,000 train / 6,000 test, class balance preserved at 77.88/22.12 in both)
- The held-out test set was written to `test_data.csv` (6,000 × 27 — the 26
  features plus the target) so that evaluation and any downstream inference run
  against exactly the same rows the models never saw

**Why this dataset was chosen:**
It is a genuine, well-documented real-world credit risk problem rather than a
toy dataset, and it exercises every part of the assignment brief. It is large
enough (30,000 rows) for a clean 80/20 split to give stable metrics, mixes
continuous, ordinal and categorical features so that encoding and scaling
decisions actually matter, and carries a natural class imbalance that forces the
use of MCC, F1 and AUC instead of accuracy alone. It also has real data quality
issues — undocumented category codes, negative bill amounts, extreme skew — that
require deliberate cleaning choices rather than a straight read-and-fit.

## c.GitHub Repository Link
https://github.com/2025ac05530-lab/ML-assignment-2

## d. Models Used

All five models were trained inside an identical
`Pipeline([('scaler', StandardScaler()), ('clf', ...)])` on the same 24,000-row
training split, and evaluated on the same held-out 6,000-row test set at the
default 0.5 decision threshold.

Precision, recall and F1 are reported **for the positive class** (`default = 1`),
which is the class of interest here. AUC is computed from
`predict_proba(X_test)[:, 1]` and is therefore threshold-independent, unlike
every other column in the table.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7099 | **0.6938** | 0.2442 | 0.3612 | 0.3308 |
| Decision Tree | **0.8173** | 0.7418 | 0.6629 | 0.3542 | 0.4617 | **0.3896** |
| kNN | 0.7928 | 0.7021 | 0.5499 | 0.3489 | 0.4269 | 0.3200 |
| Naive Bayes | 0.6665 | 0.7239 | 0.3619 | **0.6654** | **0.4688** | 0.2807 |
| Random Forest | 0.8150 | **0.7549** | 0.6437 | 0.3662 | 0.4669 | 0.3863 |

Best value in each column in bold. No model wins more than two columns, and the
six metrics point at three different winners — which is the whole substance of
section (e).

**Hyperparameters:** Logistic Regression `max_iter=1000`; Decision Tree
`max_depth=5`; kNN `n_neighbors=5`; Gaussian Naive Bayes at defaults; Random
Forest `n_estimators=100`. `random_state=42` wherever applicable.

**Majority-class baseline** (predict "no default" for every client): 0.7788
accuracy, 0.0 recall, undefined precision (no positive predictions), and an
undefined MCC that `sklearn` returns as 0.0. Any model whose accuracy is not
comfortably clear of 0.7788 is adding no value over doing nothing.

**Confusion matrices** (test set: 4,673 non-defaulters, 1,327 defaulters):

| ML Model | Correct non-defaults (TN) | False alarms (FP) | Missed defaulters (FN) | Caught defaulters (TP) |
|---|---|---|---|---|
| Logistic Regression | 4,530 | 143 | 1,003 | 324 |
| Decision Tree | 4,434 | 239 | 857 | 470 |
| kNN | 4,294 | 379 | 864 | 463 |
| Naive Bayes | 3,116 | 1,557 | 444 | 883 |
| Random Forest | 4,404 | 269 | 841 | 486 |

Every row sums to 4,673 non-defaulters and 1,327 defaulters, and all six metrics
in the table above are reproducible from these four counts.

The raw counts expose what the summary metrics compress. Logistic Regression
raises only 143 false alarms but lets 1,003 of 1,327 defaulters through — 75.6%
of them. Naive Bayes misses only 444 but wrongly accuses 1,557 paying customers,
a third of the entire non-defaulting population. Random Forest sits between them
at 841 missed and 269 false alarms. This is a business trade-off about the
relative cost of the two error types, not a single "which model is best"
question, and section (e) resolves it by stating a cost criterion explicitly.


## e. Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision of all five (0.6938) but by far the worst recall (0.2442) — it catches only about a quarter of actual defaulters. Its 0.8090 accuracy is only 3.0 percentage points above the do-nothing baseline of 0.7788, which is the clearest sign that accuracy is the wrong headline metric here. The lowest AUC in the set (0.7099) suggests the decision boundary between defaulters and non-defaulters is genuinely non-linear, so a linear model in the original feature space cannot separate them well. Useful as a calibrated, interpretable floor, not as the final model. |
| Decision Tree | Best accuracy (0.8173) and best MCC (0.3896) of the five, from a single tree capped at `max_depth=5`. The depth cap is doing real work: it stops the tree memorising the training data and forces splits onto the few strongly predictive features. Inspecting `feature_importances_` confirms this directly — `PAY_0` alone accounts for 0.680 of the tree's splitting decisions and `PAY_2` a further 0.138, so two of the 26 features carry 82% of the model. It is specifically the *recent* repayment statuses that matter: `PAY_3` drops to 0.011 and `PAY_4` to 0.008. It roughly doubles Logistic Regression's recall for only a small precision give-back, which is exactly the trade the business case wants. Strongest interpretability-to-performance ratio in the comparison. |
| kNN | Weakest of the tree/linear group — 0.7928 accuracy sits almost on the baseline, and 0.5499 precision means nearly half its default flags are wrong. Predictable given the data: with 26 dimensions, heavily skewed monetary features and a mix of binary dummies and continuous amounts, Euclidean distance stops being a meaningful notion of "similar client" even after standardisation. Its low AUC (0.7021) confirms the neighbourhood vote is a poor probability estimator here. |
| Naive Bayes | The outlier. Highest recall by a wide margin (0.6654 — it catches two thirds of defaulters), and it nominally tops the F1 column at 0.4688, though that is a rounding-error edge over Random Forest (0.4669) and Decision Tree (0.4617) rather than a real win — the top three are effectively tied on F1, and they arrive there by opposite routes. But its precision of 0.3619 means roughly two of every three clients it flags will not actually default. Its 0.6665 accuracy is the only score in the table *below* the majority-class baseline, and its MCC (0.2807) is the worst of the five. The cause is its core assumption: the features are anything but conditionally independent (the six BILL_AMT columns are near-collinear month to month), so its probabilities are badly skewed toward the positive class. Worth noting that if the cost of a missed defaulter were extreme, this aggressive-flagging behaviour could still be the right choice — but the same effect is better achieved by lowering the threshold on a stronger model. |
| Random Forest | Best AUC (0.7549) and best recall among the high-precision models (0.3662), with accuracy (0.8150) and MCC (0.3863) essentially tied with the single Decision Tree. The highest AUC means it ranks clients by risk better than anything else in the table, which matters because it is threshold-independent — the operating point can be tuned after the fact without retraining. That it only matches a depth-5 tree at the 0.5 threshold is explained by the feature importances measured on that tree: two features carry 82% of the signal and the top five carry roughly 93%, so there is little structure left for 100 trees to find that one shallow tree has not already captured. |
| **Overall Winner** | **Random Forest**, on the strength of its class-leading AUC (0.7549) while staying within 0.004 of the Decision Tree on both accuracy (gap 0.0023) and MCC (gap 0.0033). Because the differences at the 0.5 threshold are small, the tiebreaker is ranking quality: a lender needs to sort clients by risk and pick an operating point that matches its collections capacity, and the Random Forest's probability ordering is the most reliable one available. **Honourable mention to the Decision Tree** — it wins outright on accuracy and MCC, trains in a fraction of the time and can be read off as a set of rules, so it is the better choice if the model has to be explained to a credit committee or a regulator. The broader observation is that all five models are constrained by the same ceiling — no model exceeds 0.755 AUC — and the feature importances explain why. The tree is not really using six months of history: it uses last month heavily (`PAY_0`, 0.680), the month before weakly (`PAY_2`, 0.138), and almost nothing else. The predictive signal in this dataset is both concentrated and shallow, which is precisely the regime in which extra model complexity buys nothing. The strongest levers available from here are therefore not model choice but threshold tuning and feature engineering on the repayment-history columns — trend and volatility features across `PAY_*`, or utilisation ratios of `BILL_AMT` against `LIMIT_BAL` — to create signal the current representation does not expose. |

**Evidence — Decision Tree feature importances** (`models['Decision Tree'].named_steps['clf'].feature_importances_`):

| Feature | Importance |
|---|---|
| PAY_0 | 0.680 |
| PAY_2 | 0.138 |
| PAY_AMT3 | 0.044 |
| PAY_5 | 0.033 |
| BILL_AMT1 | 0.032 |

Top two features: 82% of the model. Top five: ~93%. Remaining 21 features: ~7%.

**Stating the cost criterion.** Ranking quality is the tiebreaker above, but the
choice of *operating point* depends on the relative cost of the two error types.
Writing `r` for the cost of a missed defaulter divided by the cost of a false
alarm, total expected cost is `FN × r + FP`, and the confusion matrices give the
crossover points directly:

| Cost ratio `r` | Lowest-cost model at threshold 0.5 |
|---|---|
| r < 1.9 | Decision Tree |
| 1.9 < r < 3.2 | Random Forest |
| r > 3.2 | Naive Bayes |

Two conclusions follow. First, **kNN is strictly dominated** by the Decision Tree
— more misses (864 vs 857) *and* more false alarms (379 vs 239) — so it is never
the right choice at any cost ratio. Logistic Regression is dominated by Random
Forest for any r > 0.78, i.e. everywhere except the implausible regime where
annoying a paying customer costs more than writing off a defaulted balance.

Second, for a lender r is realistically well above 3.2 — an unrecovered balance
runs to thousands of NT$ while an outreach call costs very little — which appears
to hand the win to Naive Bayes. It does not, and this is the key point: Naive
Bayes achieves that recall through miscalibration, not discrimination. Its
inflated posteriors act as a de facto lowered threshold. The disciplined way to
buy the same recall is to take the model with the best ranking (Random Forest,
AUC 0.7549 vs 0.7239) and lower its threshold deliberately, which yields higher
precision at any given recall than Naive Bayes can offer. Hence: **Random Forest,
operated below the 0.5 threshold**, chosen on the criterion of ranking quality
because that is the only property the threshold cannot be tuned to fix.
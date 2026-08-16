
# Machine Learning Assignment 2
Name: Shruti Basavaraj Desai
BITS ID: 2025AC05530

## a. Problem Statement
Given a credit card client's demographic profile, credit limit, six months of
repayment status, six months of bill statement amounts and six months of payment
amounts, predict whether that client will default on their payment next month.

This is a binary classification problem framed from the lender's point of view.
The practical use is early risk flagging: a client predicted to default can be
routed to collections outreach, a credit limit review, or a restructured payment
plan before the missed payment actually happens.

The business cost is asymmetric. A false negative (a defaulter we fail to flag)
means unrecovered debt, while a false positive (a paying client we flag) costs
only some wasted outreach and a mildly annoyed customer. Because of this, and
because only 22% of clients default, plain accuracy is a misleading score — a
model that predicts "no default" for everyone already scores 77.88%. Recall,
F1 and MCC are the metrics that actually separate the models here, with AUC used
to judge the quality of the underlying probability ranking independent of the
0.5 threshold.




## b. Dataset Description

**Name:** Default of Credit Card Clients Dataset
**Kaggle URL:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
**Original source:** UCI Machine Learning Repository — Taiwan, 2005

**Shape:** 30000 × 25     
**Features used:** 23
**Target variable:** `default_payment_next_month`  
**Task type:** Binary classification


**Classes:**
| Value | Meaning | Count | % |
|---|---|---|---|
| 0 | No default next month | 23364 | 77.88% |
| 1 | Defaulted next month | 6636 | 22.12% |
                                    

**Feature groups:**
| Group | Columns | Description |
|---|---|---|
| Demographic | LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE | Credit limit and client attributes |
| Repayment history | PAY_0, PAY_2–PAY_6 | Payment status over 6 months (-1 = paid duly, positive = months delayed) |
| Bill amounts | BILL_AMT1–BILL_AMT6 | Monthly bill statement amounts (NT$) |
| Payment amounts | PAY_AMT1–PAY_AMT6 | Amount paid each month (NT$) |

Note: the repayment columns are labelled PAY_0, PAY_2…PAY_6 — there is no PAY_1
in the original dataset. Retained as-is rather than renamed.

**Missing values:** None

**Data quality observations:**

**Preprocessing applied:**
- Dropped `ID` — a row identifier with no predictive value
- Renamed target column to remove dots for easier handling
- SEX / EDUCATION / MARRIAGE are categorical concepts stored as integers.
  Decision: [one-hot encoded / retained as integers] because [your reasoning]
- StandardScaler applied inside a Pipeline, fit on training data only, to prevent
  test-set information leaking into the scaling parameters
- 80/20 stratified train-test split, random_state=42 for reproducibility

**Why this dataset was chosen:**

## c.GitHub Repository Link
https://github.com/2025ac05530-lab/ML-assignment-2

## d. Models Used

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7099 | 0.6938 | 0.2442 | 0.3612 | 0.3308 |
| Decision Tree | 0.8173 | 0.7418 | 0.6629 | 0.3542 | 0.4617 | 0.3896 |
| kNN | 0.7928 | 0.7021 | 0.5499 | 0.3489 | 0.4269 | 0.3200 |
| Naive Bayes | 0.6665 | 0.7239 | 0.3619 | 0.6654 | 0.4688 | 0.2807 |
| Random Forest | 0.8150 | 0.7549 | 0.6437 | 0.3662 | 0.4669 | 0.3863 |


## e. Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest | |
| **Overall Winner** | |
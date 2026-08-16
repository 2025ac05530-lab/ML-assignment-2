
# Machine Learning Assignment 2
Name: Shruti Basavaraj Desai
BITS ID: 2025AC05530

## a. Problem Statement



## b. Dataset Description

**Name:** Default of Credit Card Clients Dataset
**Kaggle URL:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
**Original source:** UCI Machine Learning Repository — Taiwan, 2005

**Shape:** [rows] × [columns]        ← from df.shape, Cell 2
**Features used:** [n] (after dropping ID)
**Target variable:** `[exact name]`  ← confirm by scrolling right
**Task type:** Binary classification

**Classes:**
| Value | Meaning | Count | % |
|---|---|---|---|
| 0 | No default next month | [ ] | [ ] |
| 1 | Defaulted next month | [ ] | [ ] |
                                     ← from value_counts(), Cell 5

**Feature groups:**
| Group | Columns | Description |
|---|---|---|
| Demographic | LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE | Credit limit and client attributes |
| Repayment history | PAY_0, PAY_2–PAY_6 | Payment status over 6 months (-1 = paid duly, positive = months delayed) |
| Bill amounts | BILL_AMT1–BILL_AMT6 | Monthly bill statement amounts (NT$) |
| Payment amounts | PAY_AMT1–PAY_AMT6 | Amount paid each month (NT$) |

Note: the repayment columns are labelled PAY_0, PAY_2…PAY_6 — there is no PAY_1
in the original dataset. Retained as-is rather than renamed.

**Missing values:** [state what isnull().sum() shows, Cell 4]

**Data quality observations:**
- EDUCATION is documented as 1–4 but the file also contains values [list what you find].
  Handling: [your decision]
- MARRIAGE is documented as 1–3 but also contains [list what you find].
  Handling: [your decision]
- LIMIT_BAL appears partly in scientific notation in the raw CSV; read as float64.

**Preprocessing applied:**
- Dropped `ID` — a row identifier with no predictive value
- Renamed target column to remove dots for easier handling
- SEX / EDUCATION / MARRIAGE are categorical concepts stored as integers.
  Decision: [one-hot encoded / retained as integers] because [your reasoning]
- StandardScaler applied inside a Pipeline, fit on training data only, to prevent
  test-set information leaking into the scaling parameters
- 80/20 stratified train-test split, random_state=42 for reproducibility

**Why this dataset was chosen:**

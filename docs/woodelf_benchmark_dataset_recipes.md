# WOODELF benchmark dataset recipes

This file gives **10 separate, runnable Python recipes**. Each recipe does:

1. download the data,
2. preprocess it,
3. split train/test,
4. train a sensible tree-ensemble model.

I biased the choices toward **large tabular datasets** and toward **models commonly associated with the dataset or source**.

## Common setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost tensorflow-datasets kagglehub pyarrow
```

Notes:
- Some datasets are **big**. Expect multi-GB RAM use.
- The Kaggle-based recipes require `kagglehub` access and a logged-in Kaggle account.
- For very large experiments, you may want to train on a subset first and then rerun on the full data.

---

## 1) HIGGS -> XGBoost (following the XGBoost paper setup closely)

**Why this choice**
- The XGBoost KDD'16 paper benchmarks on **Higgs-1M** / **Higgs-10M**.
- In the paper, the common setting is **max_depth = 8**, **shrinkage = 0.1**, and **500 trees**.
- HIGGS is dense and numeric, so preprocessing is intentionally minimal.

**Source-inspired choices**
- Dataset: UCI HIGGS.
- Model recipe: Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System* (KDD 2016), Section 6.3.

```python
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

# UCI file format: first column is label, then 28 float features.
cols = ["label"] + [f"f{i}" for i in range(28)]
df = pd.read_csv(URL, header=None, names=cols)

# Optional: mimic the classic 1M-instance benchmark from the XGBoost paper.
# df = df.iloc[:1_000_000].copy()

X = df.drop(columns="label")
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, pred))
```

---

## 2) Airline Delay -> LightGBM (using LightGBM on BTS-style flight-delay data)

**Why this choice**
- The LightGBM paper explicitly includes a **Flight Delay** benchmark and notes that it contains many one-hot-like features.
- For a modern practical recipe, I recommend **LightGBM with categorical columns kept as categorical**, instead of manually one-hot encoding everything.

**Source-inspired choices**
- Data origin: BTS On-Time / Flight Delay data.
- Model family: LightGBM paper benchmarked Flight Delay.
- Practical preprocessing: use pandas categorical dtype and let LightGBM handle them.

**Important**
- The official BTS download flow is awkward to automate reliably.
- This recipe uses a **Kaggle mirror sourced from BTS monthly files**.

```python
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import kagglehub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Kaggle mirror of BTS-style monthly flight data.
# Requires Kaggle credentials.
path = Path(kagglehub.dataset_download("hrishitpatil/flight-data-2024"))

# The dataset contains monthly CSVs. Concatenate them.
parts = sorted(path.glob("*.csv"))
df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)

# A practical target: departure delayed by 15+ minutes.
# Adjust this if your mirror already includes a target.
if "DepDelay" in df.columns:
    df["target_delay15"] = (df["DepDelay"].fillna(0) >= 15).astype(int)
elif "DEP_DELAY" in df.columns:
    df["target_delay15"] = (df["DEP_DELAY"].fillna(0) >= 15).astype(int)
else:
    raise ValueError("Could not find a departure-delay column.")

# Keep a moderately standard set of predictive columns if present.
want = [
    "Year", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "Distance",
    "DepDelay", "TaxiOut", "TaxiIn", "AirTime",
]

cols = [c for c in want if c in df.columns]
df = df[cols + ["target_delay15"]].copy()

# Remove obvious leakage at prediction time.
for leakage in ["DepDelay", "TaxiOut", "TaxiIn", "AirTime"]:
    if leakage in df.columns:
        df.drop(columns=leakage, inplace=True)

# Time engineering.
for c in ["CRSDepTime", "CRSArrTime"]:
    if c in df.columns:
        vals = df[c].fillna(0).astype(int)
        df[c + "_hour"] = vals // 100
        df[c + "_minute"] = vals % 100
        df.drop(columns=c, inplace=True)

# Missing values.
# LightGBM can handle NaNs in numerics; categoricals become pandas category.
y = df.pop("target_delay15")

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=255,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols,
)

pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, pred))
```

---

## 3) MSLR-WEB10K -> XGBoost ranker (following XGBoost's official LTR direction)

**Why this choice**
- MSLR-WEB10K is one of the canonical public learning-to-rank datasets.
- XGBoost has an official learning-to-rank tutorial and supports `rank:ndcg`.

**Source-inspired choices**
- Dataset: Microsoft MSLR-WEB10K.
- Model family: XGBoost ranking, objective `rank:ndcg`.
- Split: use the official fold split packaged by TensorFlow Datasets.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score


def tfds_split_to_frame(split_name: str):
    ds = tfds.load("mslr_web/10k_fold1", split=split_name, as_supervised=False)
    rows = []
    for ex in tfds.as_numpy(ds):
        features = ex["features"].astype(np.float32)
        label = int(ex["label"])
        qid = ex["query_id"].decode() if isinstance(ex["query_id"], bytes) else str(ex["query_id"])
        rows.append((qid, label, *features.tolist()))
    cols = ["qid", "label"] + [f"f{i}" for i in range(136)]
    return pd.DataFrame(rows, columns=cols)

train_df = tfds_split_to_frame("train")
valid_df = tfds_split_to_frame("vali")
test_df  = tfds_split_to_frame("test")

feature_cols = [c for c in train_df.columns if c.startswith("f")]

X_train = train_df[feature_cols]
y_train = train_df["label"]
qid_train = train_df.groupby("qid").size().to_numpy()

X_valid = valid_df[feature_cols]
y_valid = valid_df["label"]
qid_valid = valid_df.groupby("qid").size().to_numpy()

X_test = test_df[feature_cols]
y_test = test_df["label"]

model = XGBRanker(
    objective="rank:ndcg",
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.5,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    qid=qid_train,
    eval_set=[(X_valid, y_valid)],
    eval_qid=[qid_valid],
    verbose=False,
)

# Simple query-wise NDCG@10 evaluation.
test_df = test_df.copy()
test_df["pred"] = model.predict(X_test)

ndcgs = []
for _, g in test_df.groupby("qid"):
    ndcgs.append(ndcg_score([g["label"].to_numpy()], [g["pred"].to_numpy()], k=10))

print("Mean NDCG@10:", float(np.mean(ndcgs)))
```

---

## 4) Covertype -> Random Forest (classic strong baseline)

**Why this choice**
- Covertype is a classic tree benchmark.
- Random Forest is historically associated with this style of tabular multiclass classification.
- The dataset is already numeric and clean.

**Source-inspired choices**
- Dataset loader: `sklearn.datasets.fetch_covtype`.
- Model family: Random Forest, following the classic Breiman-style ensemble choice.

```python
from __future__ import annotations

from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = fetch_covtype(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

---

## 5) Poker Hand -> CatBoost (natural fit for categorical card features)

**Why this choice**
- The ten input features are really categorical card descriptors: suit/rank for 5 cards.
- CatBoost handles categorical features natively and is a very reasonable tree-ensemble choice here.

**Source-inspired choices**
- Dataset: UCI Poker Hand.
- Preprocessing: keep suit/rank columns as categorical rather than pretending they are continuous.

```python
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data"
test_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"

cols = [
    "S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "label"
]

train_df = pd.read_csv(train_url, header=None, names=cols)
test_df  = pd.read_csv(test_url, header=None, names=cols)

df = pd.concat([train_df, test_df], ignore_index=True)
for c in cols[:-1]:
    df[c] = df[c].astype(str)

X = df.drop(columns="label")
y = df["label"].astype(int)

# User asked for train-test splitting, so we resplit the merged data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_features = list(range(X.shape[1]))

model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=500,
    depth=8,
    learning_rate=0.1,
    eval_metric="Accuracy",
    random_seed=42,
    verbose=False,
)

model.fit(X_train, y_train, cat_features=cat_features)
pred = model.predict(X_test).astype(int).ravel()
print("Accuracy:", accuracy_score(y_test, pred))
```

---

## 6) Santander Customer Transaction Prediction -> LightGBM

**Why this choice**
- This dataset is a popular large tabular binary-classification benchmark from Kaggle.
- Features are already numeric, so a plain LightGBM baseline is clean and strong.

**Source-inspired choices**
- Dataset: Kaggle Santander competition.
- Model: LightGBM is one of the standard competition workhorses for this task.
- There is no single paper-standard hyperparameter set everybody uses, so this is a pragmatic strong baseline.

```python
from __future__ import annotations

from pathlib import Path
import kagglehub
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

path = Path(kagglehub.competition_download("santander-customer-transaction-prediction"))
train_path = path / "train.csv"

df = pd.read_csv(train_path)

# Kaggle columns are usually: ID_code, target, var_0 ... var_199
X = df.drop(columns=[c for c in ["ID_code", "target"] if c in df.columns])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, pred))
```

---

## 7) IEEE-CIS Fraud Detection -> LightGBM

**Why this choice**
- This is a large, messy, realistic fraud dataset with many missing values and categorical columns.
- LightGBM handles this type of tabular data very well.

**Source-inspired choices**
- Dataset: Kaggle IEEE-CIS Fraud Detection.
- Practical preprocessing follows standard competition practice: merge identity onto transaction data, derive simple time features, keep object columns categorical, keep NaNs.

```python
from __future__ import annotations

from pathlib import Path
import kagglehub
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

path = Path(kagglehub.competition_download("ieee-fraud-detection"))

train_tr = pd.read_csv(path / "train_transaction.csv")
train_id = pd.read_csv(path / "train_identity.csv")

df = train_tr.merge(train_id, on="TransactionID", how="left")

y = df["isFraud"].astype(int)
X = df.drop(columns=[c for c in ["isFraud", "TransactionID"] if c in df.columns]).copy()

# Very common practical feature engineering: turn TransactionDT into coarse cyclic-ish calendar features.
if "TransactionDT" in X.columns:
    sec = X["TransactionDT"].fillna(0)
    X["DT_day"] = (sec // (24 * 3600)).astype("int32")
    X["DT_week"] = (sec // (7 * 24 * 3600)).astype("int32")
    X["DT_hour"] = ((sec // 3600) % 24).astype("int8")

# LightGBM can consume categoricals if they are category dtype.
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=255,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train, categorical_feature=cat_cols)
pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, pred))
```

---

## 8) SUSY -> XGBoost (same spirit as HIGGS)

**Why this choice**
- SUSY is another classic large physics dataset: huge, numeric, binary classification.
- Like HIGGS, it is a very natural fit for XGBoost.

**Source-inspired choices**
- Dataset: UCI SUSY.
- Model: use the same XGBoost-style recipe as the HIGGS benchmark family.

```python
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

URL = "https://mlphysics.ics.uci.edu/data/susy/SUSY.csv.gz"

# First column is label, then 18 numerical features.
cols = ["label"] + [f"f{i}" for i in range(18)]
df = pd.read_csv(URL, header=None, names=cols)

X = df.drop(columns="label")
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, pred))
```

---

## 9) KDD Cup 2010 (Bridge to Algebra / Algebra) -> CatBoost

**Why this choice**
- This is a large historical benchmark with many categorical/text-like fields after tabularization.
- CatBoost is the most painless tree-ensemble baseline when you want to keep many discrete columns categorical.

**Important**
- The original KDD Cup 2010 data is **not as easy to auto-download** as the others; it typically requires registration / manual access.
- So this recipe assumes you already downloaded one of the development sets and saved it locally.

**Source-inspired choices**
- Dataset: KDD Cup 2010 Educational Data Mining Challenge.
- Model: CatBoost for mixed mostly-categorical tabular fields.

```python
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# Example local file after manual download / extraction.
# Replace with the actual file you downloaded.
DATA_PATH = Path("./kddcup2010/algebra_2008_2009_train.csv")

df = pd.read_csv(DATA_PATH, sep="\t")

# The challenge used correctness-like targets. Update this to your exact file's target column.
possible_targets = ["Correct First Attempt", "correct", "label", "target"]
target_col = next((c for c in possible_targets if c in df.columns), None)
if target_col is None:
    raise ValueError(f"Could not identify target column among {possible_targets}")

y = df[target_col].astype(int)
X = df.drop(columns=[target_col]).copy()

# Convert high-cardinality string-like columns to categorical.
cat_cols = []
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype(str)
        cat_cols.append(c)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=500,
    depth=8,
    learning_rate=0.1,
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

model.fit(X_train, y_train, cat_features=cat_features_idx)
pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, pred))
```

---

## 10) YearPredictionMSD -> XGBoost regressor (respecting the official split)

**Why this choice**
- This is a large classic regression benchmark.
- The dataset description explicitly asks users to respect the official split:
  first **463,715** rows for train, last **51,630** rows for test.
- XGBoost is a strong and very standard tree ensemble for this scale.

**Source-inspired choices**
- Dataset: UCI YearPredictionMSD.
- Split: exactly the official UCI split.

```python
from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile
import requests
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"

resp = requests.get(URL, timeout=120)
resp.raise_for_status()
with ZipFile(BytesIO(resp.content)) as zf:
    name = zf.namelist()[0]
    with zf.open(name) as f:
        df = pd.read_csv(f, header=None)

# First column is target year, remaining 90 features.
y = df.iloc[:, 0].astype(float)
X = df.iloc[:, 1:].astype(float)

# Official split from the UCI description.
X_train = X.iloc[:463_715].copy()
y_train = y.iloc[:463_715].copy()
X_test = X.iloc[463_715:].copy()
y_test = y.iloc[463_715:].copy()

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred, squared=False)
print("RMSE:", rmse)
```

---

# Sources I matched against

These are the main sources I aligned to when possible:

1. **XGBoost KDD 2016 paper**
   - Uses HIGGS and Yahoo LTRC.
   - Common benchmark setting reported there: **500 trees, max depth 8, shrinkage 0.1**.

2. **LightGBM NeurIPS 2017 paper**
   - Explicitly includes **Flight Delay** among its benchmark datasets.
   - Motivates LightGBM for large sparse / one-hot-heavy tabular problems.

3. **Official dataset pages / official loaders**
   - UCI HIGGS
   - UCI SUSY
   - UCI Poker Hand
   - UCI YearPredictionMSD
   - Microsoft MSLR
   - scikit-learn `fetch_covtype`
   - Kaggle competition pages for Santander and IEEE-CIS Fraud Detection

# A practical recommendation for your WOODELF paper

If the goal is **benchmarking explanation time**, I would not run all explanation methods on the full raw training sets immediately. I would do this:

1. Train the full model on the full training set.
2. Evaluate explanations on fixed test batches, e.g. `n = 1, 10, 100, 1000` predictions.
3. Vary the background size, e.g. `m = 10, 100, 1000, 10000`.
4. Save the trained models and the exact train/test/background indices.

That will make the SHAP-vs-WOODELF scaling story much cleaner and much easier to reproduce.

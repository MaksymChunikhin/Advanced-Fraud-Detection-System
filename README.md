# IEEE-CIS Fraud Detection

A machine-learning project for detecting fraudulent online transactions, based on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle competition.
It is a single, clean end-to-end pipeline (three notebooks) plus a live monitoring dashboard,
built for learning and portfolio purposes.

## About

The goal is to estimate the probability that a transaction is fraudulent, using banking and
user-behavioural data, and to turn that model into something you can actually watch working.

Technologies and approaches used:

- Exploratory Data Analysis (EDA)
- Feature engineering (time-normalized D-columns, aggregations, frequency encoding)
- Time-aware validation (no leakage from the future)
- LightGBM · XGBoost · CatBoost (Logistic Regression as a floor)
- Hyperparameter tuning with Optuna
- SHAP explainability
- A real-time monitoring dashboard (Streamlit)

The main focus was on high-quality data preparation, informative features and an honest
validation strategy — not on maximally stacked ensembles.

## Competition & Dataset

Organized by the IEEE Computational Intelligence Society together with the payment-services
company **Vesta**.

- **Goal:** predict the probability that an online transaction is fraud (the `isFraud` target).
- **Metric:** ROC-AUC.
- **Scale:** ~590K training / ~506K test transactions; **471 features** after preparation —
  transaction, card, address, email, device and identity information (mostly anonymized).
- **Imbalance:** only ~3.5% of transactions are fraud, which makes the validation strategy
  and metric choice especially important.

Data is **not included** in the repository due to file size — download it from the competition
page. Raw files used: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`,
`test_identity.csv`.

## Project Structure

The project is split into 3 notebooks; data is passed between them as parquet.

| Notebook | What it does |
|---|---|
| `notebooks/fraud_01_data_prep_features.ipynb` | Load & merge transaction/identity tables, EDA, memory optimization, missing-value handling, encoding, **feature engineering** → `data/processed/*.parquet`. |
| `notebooks/fraud_02_modeling_tuning.ipynb` | Baselines (LogReg / LightGBM / XGBoost / CatBoost), **Optuna tuning** of the three GBMs, final models, ensemble check, **SHAP**, threshold tuning → `models/*.pkl`. |
| `notebooks/fraud_03_inference_submission.ipynb` | Score the test set, build single-model and rank-blend submissions → `submissions/*.csv`. |

Other folders: `data/processed/` (parquet), `models/` (saved models, Optuna studies, best
params), `submissions/` (Kaggle files), `app/` (Streamlit dashboard).

## Pipeline

### 1. Data preparation & features (notebook 01)

- Merging the transaction and identity tables, memory optimization, missing-value handling
  and categorical encoding.
- Feature engineering: **time-normalized D-columns** (`D − transaction_day`), aggregation
  features (e.g. mean `TransactionAmt` per card), **frequency encoding**, time, email and
  interaction features.

### 2. Validation strategy

Transactions are ordered by time and the competition test set is the *future*. A random
K-Fold mixes past and future and overstates the score, so we use a **time-aware holdout**:
train on the earliest 80%, validate on the most recent 20%.

### 3. Models & tuning (notebook 02)

Baselines on the time-aware holdout (ROC-AUC):

| Model | Baseline ROC-AUC |
|---|---|
| XGBoost | 0.92631 |
| LightGBM | 0.92463 |
| CatBoost | 0.89642 |
| Logistic Regression | 0.81995 |

The three gradient-boosting models clearly beat logistic regression — the relationships in
the data are non-linear. They were then tuned with **Optuna** (30 trials, TPE, the same
time-aware split, early stopping):

| Model | Baseline | Tuned | Gain |
|---|---|---|---|
| LightGBM | 0.92463 | **0.93797** | +0.01334 |
| XGBoost | 0.92631 | 0.93580 | +0.00949 |
| CatBoost | 0.89628 | 0.92802 | +0.03174 |

After tuning all three improved and converged — a sign we are near the limit of a single
model on the current features.

### 4. Explainability — SHAP

SHAP on the best model (LightGBM) shows the strongest drivers are **activity counters**
(`C13`, `C1`, `C14`, `C11`), the **transaction amount** and our amount-per-card feature
(`card1_TransactionAmt_mean`), **time deltas** (`D1_norm`, `D2`, `D4`) and our **frequency
features** (`P_emaildomain_frequency`, `card1_frequency`). Importantly, the features we
engineered ourselves land in the top — the feature work genuinely paid off.

### 5. Threshold tuning

The model outputs a probability; the fraud / not-fraud decision uses a threshold. At 0.5 it
catches only part of the fraud (low recall). The operating point should be chosen by business
cost — a missed fraud vs a false block — not by the default 0.5. The threshold only affects
the operational decision, not the ROC-AUC score.

### 6. Inference & submission (notebook 03)

The saved models score `test.parquet`; submissions are written as **probabilities** (ROC-AUC
cares about order). The ensemble uses a **rank-average** of model predictions, which is robust
to different probability scales.

## Results

On the holdout, combining models barely helps: the `lgb+xgb` blend (0.93793) is essentially
equal to LightGBM alone (0.93797), and adding the weaker CatBoost lowers it (0.93721).

Kaggle leaderboard (late submission, ROC-AUC, sorted by Private):

| Submission | Public LB | Private LB |
|---|---|---|
| **LightGBM + XGBoost** (rank-blend) | 0.948949 | **0.919338** |
| **LightGBM (single — final model)** | **0.949379** | 0.919059 |
| XGBoost | 0.946904 | 0.917151 |
| LightGBM + XGBoost + CatBoost | 0.947121 | 0.915877 |
| CatBoost | 0.934277 | 0.897988 |

The `lgb+xgb` rank-blend gives the best **private** score (0.919338), but the single tuned
LightGBM is within ~0.0003 of it (0.919059) and even wins on the **public** board. Since the
gain from blending is negligible, we adopt the **single tuned LightGBM** as the final model —
it is simpler, faster and more stable to serve; the blend is kept as an optional alternative.

For context, the top of the private leaderboard:

| Rank | Team | Private LB |
|---|---|---|
| 1 | FraudSquad | 0.945884 |
| 2 | 2 uncles and 3 puppies | 0.944210 |

The remaining gap to the leaders is expected: top solutions relied on very aggressive
user-identification (UID) feature engineering and large stacked ensembles. This project
deliberately stays a clean, readable end-to-end pipeline.

## Demo & Service

### Real-time monitoring dashboard (Streamlit) — working

An interactive dashboard that replays pre-scored transactions
(`submissions/submission_lgb.csv` joined with the raw transaction fields) as if they arrive
live. When a transaction's fraud probability crosses the threshold, a toast alert fires and
the transaction is routed to a **"fraud control"** review queue, with live statistics
(processed, flagged, fraud rate, frozen amount, average probability) and a probability stream.

![Real-time fraud monitoring dashboard](docs/images/app_screen.jpg)

```bash
streamlit run app/streamlit_app.py
```

It joins the model output with the raw transaction fields, so the demo needs no extra
inference code to run. The score source (single LightGBM, lgb+xgb blend, …) and the threshold
can be switched live from the sidebar.

### Inference API (FastAPI) — roadmap

A REST inference service (`/predict`) is a planned next step. It is **not implemented yet**:
it requires porting the feature-engineering pipeline from notebook 01 into a reusable module
(`src/`) so raw transactions can be scored on the fly.

## Tech Stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · Optuna · SHAP ·
Matplotlib · Plotly · Streamlit · joblib · pyarrow.

## Key Takeaways

- **Features.** The gain that *transferred* to the leaderboard came from **time-normalizing
  the D-columns**. Entity-keyed UID/frequency features improved validation but did **not**
  transfer to the test set — a classic CV↔LB gap.
- **Validation.** A time-aware holdout is more honest than random K-Fold, because the test set
  is the future and random CV overstates the score.
- **Ensemble.** Blending all three models *hurt* (the weak CatBoost drags it down); a curated
  `lgb+xgb` blend was marginally best on private, but a single tuned LightGBM is essentially
  on par — strength *and* diversity matter, and complexity is not free.
- **GPU.** LightGBM runs on **CPU**: its CUDA build degrades on NaN/sparse features
  (≈0.84 vs ≈0.94). XGBoost and CatBoost run on GPU.
- **Explainability.** SHAP confirmed that the hand-built features (per-card amount averages,
  frequencies) are among the most influential.

**Main conclusion:** in fraud detection, the quality of feature engineering and the validation
strategy often matter more than model complexity.

## Environment

```bash
conda env create -f environment.yml
conda activate fraud
```

Python 3.10; GPU (CUDA) for XGBoost/CatBoost. Run the notebooks in order: 01 → 02 → 03.

## Author

Maksym Chunikhin

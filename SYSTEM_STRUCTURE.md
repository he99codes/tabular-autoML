# AutoML System — Complete Working Structure

---

## OVERVIEW: How the System Flows

```
CSV File
   │
   ▼
[STEP 1] CLI Parsing         → reads --data --target --task --time_budget etc.
   │
   ▼
[STEP 2] Dataset Analyzer    → inspects rows, cols, types, missing, imbalance
   │
   ▼
[STEP 3] Train/Val/Test Split → 70% / 15% / 15%
   │
   ▼
[STEP 4] Preprocessing       → impute → scale / encode → tensor
   │
   ▼
[STEP 5] Feature Engineering → original + log + sqrt + polynomial → SelectKBest
   │
   ▼
[STEP 6] Model Training Loop
   ├── sklearn models  → Optuna HPO → final fit → evaluate
   └── pytorch models  → Optuna HPO → full train loop → evaluate
   │
   ▼
[STEP 7] Leaderboard         → rank all models by primary score
   │
   ▼
[STEP 8] Best Model Selected → highest primary score wins
   │
   ▼
[STEP 9] Explainability      → SHAP or feature_importances_
   │
   ▼
[STEP 10] Save Artifacts     → model + pipeline + leaderboard.csv
```

---

## CASE A: CLASSIFICATION TASK

```bash
python main.py --data data.csv --target churn --task classification
```

### Models that run (in this order):

```
1. RandomForest        (RandomForestClassifier)
2. GradientBoosting    (GradientBoostingClassifier)
3. LogisticRegression  (LogisticRegression)
4. XGBoost             (XGBClassifier)          ← only if xgboost installed
5. LightGBM            (LGBMClassifier)          ← only if lightgbm installed
6. FeedforwardNN       (PyTorch)                 ← only if torch installed
7. ResidualMLP         (PyTorch)                 ← only if torch installed
```

### Metrics computed per model:

```
accuracy   = correct predictions / total
precision  = TP / (TP + FP)       → binary: direct | multiclass: macro average
recall     = TP / (TP + FN)       → binary: direct | multiclass: macro average
f1         = 2 * (P * R) / (P + R)
roc_auc    = area under ROC curve  → binary: normal | multiclass: OvR macro
```

### How best model is selected (classification):

```
primary_score = roc_auc   (if available)
              = f1         (fallback, when predict_proba not available)

→ Model with HIGHEST primary_score wins
```

Example leaderboard:
```
Rank  Model               accuracy  f1      roc_auc   → primary_score
 1    LogisticRegression  0.907     0.909   0.972      0.972   ← WINNER
 2    GradientBoosting    0.880     0.868   0.958      0.958
 3    FeedforwardNN       0.871     0.855   0.941      0.941
 4    RandomForest        0.853     0.836   0.945      0.945
```

---

## CASE B: REGRESSION TASK

```bash
python main.py --data data.csv --target price --task regression
```

### Models that run (in this order):

```
1. RandomForest        (RandomForestRegressor)
2. GradientBoosting    (GradientBoostingRegressor)
3. Ridge               (Ridge regression)
4. LinearRegression    (LinearRegression)
5. XGBoost             (XGBRegressor)            ← only if xgboost installed
6. LightGBM            (LGBMRegressor)            ← only if lightgbm installed
7. FeedforwardNN       (PyTorch)                  ← only if torch installed
8. ResidualMLP         (PyTorch)                  ← only if torch installed
```

### Metrics computed per model:

```
RMSE = sqrt( mean( (y_pred - y_true)² ) )   → lower is better
MAE  = mean( |y_pred - y_true| )             → lower is better
R²   = 1 - SS_residual / SS_total            → higher is better (1.0 = perfect)
```

### How best model is selected (regression):

```
primary_score = -RMSE    (negated so that higher = better, like classification)

→ Model with LOWEST RMSE = HIGHEST primary_score = WINS
```

Example leaderboard:
```
Rank  Model            rmse      mae      r2      primary_score
 1    Ridge            34.44    26.92    0.954    -34.44    ← WINNER
 2    LinearReg        35.36    27.69    0.951    -35.36
 3    FeedforwardNN    61.22    48.11    0.857    -61.22
 4    RandomForest    100.81    77.93    0.607   -100.81
```

---

## FEATURE ENGINEERING — Full Breakdown

This is what happens step by step when your preprocessed matrix hits the
FeatureEngineer. Example: you have 15 preprocessed features.

```
Input:  X shape = (N rows, 15 features)
```

### Step 1 — Original features kept as-is:
```
[f0, f1, f2, ..., f14]     → 15 columns
```

### Step 2 — Log Transform (use_log=True):
```
For each column:
  X_shifted = X - X.min(axis=0) + 0.000001   ← shift all values to positive
  log_feat  = log1p(X_shifted)                ← log(1 + x)

Result: 15 new log-transformed columns
  [log(f0), log(f1), ..., log(f14)]           → +15 columns = 30 total
```

### Step 3 — Sqrt Transform (use_sqrt=True):
```
For each column:
  X_shifted = X - X.min(axis=0)     ← shift to non-negative
  sqrt_feat = sqrt(X_shifted)

Result: 15 new sqrt-transformed columns
  [√f0, √f1, ..., √f14]             → +15 columns = 45 total
```

### Step 4 — Polynomial Features (use_polynomial=True, degree=2):
```
Runs on first 20 columns only (cap to avoid memory explosion)
With 15 features, runs on all 15.

PolynomialFeatures(degree=2, include_bias=False) generates:
  - Original terms:    f0, f1, ..., f14         (15 cols — REMOVED, already have them)
  - Squared terms:     f0², f1², ..., f14²       (15 cols)  ← NEW
  - Cross terms:       f0·f1, f0·f2, ..., f13·f14 (105 cols) ← NEW

Formula: new_cols = N*(N+1)/2 = 15*16/2 = 120

Example with just 3 features A, B, C:
  Input:  [A, B, C]
  After poly:
    A²   = A*A
    A·B  = A*B
    A·C  = A*C
    B²   = B*B
    B·C  = B*C
    C²   = C*C
  → 6 new columns (originals removed since already in base set)

With 15 features → +120 new polynomial columns = 165 total
```

### Step 5 — SelectKBest (select_k=60):
```
All 165 features are scored against target y using:
  - f_regression  (for regression tasks)   → F-statistic
  - f_classif     (for classification)     → ANOVA F-value

Top 60 features with highest scores are KEPT.
All others are DROPPED.

Output: X shape = (N rows, 60 features)
```

### Summary Table:

```
Stage                   Columns   Cumulative
─────────────────────────────────────────────
Original preprocessed      15         15
+ Log transforms           15         30
+ Sqrt transforms          15         45
+ Polynomial (degree-2)   120        165
─────────────────────────────────────────────
After SelectKBest(k=60)   ——→         60   ← final feature matrix
```

### Why you "didn't see" polynomial features:

The polynomial features ARE generated, but then SelectKBest filters the
full 165-feature matrix down to 60. The output is indexed as:
  feature_0, feature_1, ..., feature_59

These are not labeled as "poly" or "log" — they're just the top 60
regardless of type. If a polynomial feature ranked high, it's in there.
If it ranked low, SelectKBest dropped it. The system doesn't show which
type each surviving feature came from.

---

## MODEL SELECTION — Detailed Decision Tree

```
For each model in the candidate pool:
│
├── Does it have a search space? (len(search_space) > 0)
│   ├── YES → Run Optuna HPO
│   │         │
│   │         ├── Is Optuna installed?
│   │         │   ├── YES → Bayesian optimization (TPE sampler)
│   │         │   │         Each trial: suggest params → fit → score val → report
│   │         │   └── NO  → Random search fallback
│   │         │             Each trial: random sample → fit → score val → track best
│   │         │
│   │         └── Returns best_params dict
│   │
│   └── NO  → Skip HPO (e.g. LinearRegression has no hyperparams)
│             Use default params directly
│
├── Final fit on FULL train set with best_params
│
├── Predict on TEST set (never seen during HPO)
│
├── Compute metrics (accuracy/f1/auc OR rmse/mae/r2)
│
└── Compute primary_score → add to Leaderboard


After ALL models finish:
│
├── Sort leaderboard by primary_score DESCENDING
├── Rank 1 = BEST MODEL
└── best_model = leaderboard[0]["_model"]
```

---

## HPO SEARCH SPACES — Every Parameter

### RandomForest (classification AND regression):
```
n_estimators     int    [50, 500]      → number of trees
max_depth        int    [3, 20]        → max tree depth
min_samples_split int   [2, 20]        → min samples to split a node
min_samples_leaf  int   [1, 10]        → min samples in leaf node
```

### GradientBoosting (classification AND regression):
```
n_estimators     int    [50, 300]      → boosting rounds
max_depth        int    [2, 8]         → tree depth (shallower than RF)
learning_rate    float  [0.001, 0.3]   → log scale (shrinkage)
subsample        float  [0.5, 1.0]     → fraction of samples per tree
```

### LogisticRegression (classification only):
```
C               float  [0.0001, 10.0]  → log scale, inverse regularization
```

### Ridge (regression only):
```
alpha           float  [0.001, 100.0]  → log scale, regularization strength
```

### LinearRegression: NO hyperparams → skips HPO entirely

### XGBoost (if installed):
```
n_estimators     int    [50, 500]
max_depth        int    [2, 10]
learning_rate    float  [0.001, 0.3]   → log scale
subsample        float  [0.5, 1.0]
colsample_bytree float  [0.5, 1.0]    → fraction of features per tree
```

### LightGBM (if installed):
```
n_estimators     int    [50, 500]
max_depth        int    [2, 10]
learning_rate    float  [0.001, 0.3]   → log scale
num_leaves       int    [16, 256]      → controls model complexity
subsample        float  [0.5, 1.0]
```

### FeedforwardNN (PyTorch):
```
lr              float  [0.0001, 0.01]  → log scale (Adam learning rate)
hidden_dim      choice [64, 128, 256]  → units per hidden layer
n_layers        int    [2, 4]          → number of hidden layers
dropout         float  [0.1, 0.5]      → dropout probability
batch_size      choice [32, 64, 128]   → mini-batch size
weight_decay    float  [1e-6, 0.001]   → L2 regularization
```

### ResidualMLP (PyTorch):
```
lr              float  [0.0001, 0.01]  → log scale
hidden_dim      choice [64, 128, 256]  → ALL residual blocks share this width
n_blocks        int    [2, 5]          → number of residual blocks
dropout         float  [0.1, 0.5]
batch_size      choice [32, 64, 128]
weight_decay    float  [1e-6, 0.001]
```

---

## PYTORCH ARCHITECTURE DETAILS

### FeedforwardNN:
```
Input(n_features)
    │
    ▼
[Linear(n, hidden_dim) → BatchNorm1d → ReLU → Dropout]  ← repeated n_layers times
    │
    ▼
Linear(hidden_dim, output_dim)
    │
    ▼
Output: logits (classification) OR scalar (regression)
```

### ResidualMLP:
```
Input(n_features)
    │
    ▼
Linear(n, hidden_dim) → BatchNorm1d → ReLU      ← input projection
    │
    ▼
┌── ResidualBlock ──────────────────────┐
│  x_in ──────────────────────────────┐ │
│      │                              │ │
│  Linear → BN → ReLU → Dropout       │ │    ← repeated n_blocks times
│      │                              │ │
│  Linear → BN                        │ │
│      │                              │ │
│  ReLU( x_block + x_in ) ←──────────┘ │   ← skip connection
└───────────────────────────────────────┘
    │
    ▼
Linear(hidden_dim, output_dim)
```

The skip connection (x_in + x_block) is what makes it "residual" —
gradients flow directly backward through the addition, preventing
vanishing gradient in deeper networks.

---

## TRAINING ENGINE — Each Epoch

```
For each epoch (max 80):
│
├── TRAIN PHASE
│   For each mini-batch:
│   ├── Forward: output = model(X_batch)
│   ├── Loss:
│   │     classification → CrossEntropyLoss(output, y_batch)
│   │     regression     → MSELoss(output.squeeze(), y_batch)
│   ├── Backward: loss.backward()
│   ├── Gradient clip: clip_grad_norm_(max=1.0)   ← safety guard
│   └── Step: Adam optimizer updates weights
│
├── VALIDATION PHASE
│   ├── model.eval() + torch.no_grad()
│   ├── Compute val_loss on full val set
│   └── model.train()
│
├── LR SCHEDULER
│   └── ReduceLROnPlateau: if val_loss doesn't improve for 5 epochs
│       → halve the learning rate
│
└── EARLY STOPPING
    └── If val_loss doesn't improve for 10 epochs:
        ├── Restore best weights seen so far
        └── Stop training
```

---

## PREPROCESSING DECISION TREE

```
For each column:
│
├── Is it numeric dtype? (int, float)
│   └── YES → Impute(median) → StandardScaler
│
├── Is it object/categorical dtype?
│   ├── nunique ≤ 15  → Impute(most_frequent) → OneHotEncoder
│   │                   Creates binary 0/1 columns for each category
│   │
│   └── nunique > 15  → Impute(most_frequent) → OrdinalEncoder (if no category_encoders)
│                        OR TargetEncoder (if category_encoders installed)
│                        Maps categories to integers or target-mean values
│
└── Is it text? (avg_string_len > 30 AND nunique > 50)
    └── TfidfVectorizer(max_features=50, ngram_range=(1,2))
        Creates 50 TF-IDF score columns from text
```

---

## WHAT HAPPENS WITH TIME BUDGET

```
--time_budget=300  (5 minutes)

Time allocation per model (approximate):
  Each sklearn model:  ~20% of remaining budget, capped at 60s for HPO
  Each pytorch model:  ~30% of remaining budget, capped at 90s for HPO

If budget expires mid-loop:
  ├── Current model finishes its current HPO trial
  ├── No new models are started
  └── Whatever finished goes to leaderboard → best is selected

No time budget (default):
  └── Every model runs to completion with all 15 HPO trials
```

---

## EXPLAINABILITY — How It Decides

```
After best model is selected:

Is best model a tree-based sklearn model?
  (RandomForest, GradientBoosting, XGBoost, LightGBM)
  └── YES → use model.feature_importances_ directly (fast, no SHAP needed)
            OR shap.TreeExplainer if SHAP installed (exact, tree-native)

Is best model a linear sklearn model?
  (LogisticRegression, Ridge, LinearRegression)
  └── YES → use abs(model.coef_) directly
            OR shap.LinearExplainer if SHAP installed

Is best model a PyTorch model?
  └── YES → requires SHAP installed
            shap.KernelExplainer with 50 background samples (approximate)
            → if SHAP not installed: no importance available for neural nets

Output: dict of {feature_name: importance_score}
        sorted descending → top 15 printed as bar chart
```

---

## OUTPUT FILES

```
./automl_output/
├── best_model.joblib          ← sklearn model (if sklearn won)
├── best_model.pt              ← PyTorch model weights (if neural net won)
├── preprocessing.joblib       ← fitted ColumnTransformer (imputers + scalers + encoders)
├── feature_engineering.joblib ← fitted FeatureEngineer (poly + log + sqrt + selector)
└── leaderboard.csv            ← all models with all metrics, ranked
```

To reload and use the best model later:
```python
import joblib
import torch

# For sklearn winner:
model = joblib.load("automl_output/best_model.joblib")
prep  = joblib.load("automl_output/preprocessing.joblib")
feat  = joblib.load("automl_output/feature_engineering.joblib")

X_new = prep.transform(new_df)
X_new = feat.transform(X_new)
preds = model.predict(X_new)

# For PyTorch winner:
model = torch.load("automl_output/best_model.pt")
model.eval()
```

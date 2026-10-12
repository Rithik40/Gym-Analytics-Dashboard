"""
train_all_models.py
Trains and tunes 4 models (Random Forest, Extra Trees, Gradient Boosting, Stacking)
alongside the existing XGBoost. Evaluates all 5 with regression metrics and saves
everything to disk.

Run:  python train_all_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib, json, optuna

from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, explained_variance_score)
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
DATASET_PATH = r"D:\Users\Rithik\Documents\6th Semester\Data Visualization\Lab\Dataset\gym_members_exercise_tracking3.csv"
print(f"Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Drop duplicate session duration column if present
if "Session_Duration (mins)" in df.columns:
    df = df.drop(columns=["Session_Duration (mins)"])

X = df.drop(columns=["Calories_Burned"])
y = df["Calories_Burned"]

# Encode categoricals using existing label encoders
label_encoders = joblib.load("label_encoders.joblib")
for col in ["Gender", "Workout_Type", "Diet_Type"]:
    X[col] = label_encoders[col].transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

# ── 2. Metrics Helper ────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    evs  = float(explained_variance_score(y_true, y_pred))
    return {
        "r2":   round(r2,   4),
        "mae":  round(mae,  2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "evs":  round(evs,  4),
    }

def print_metrics(name, m):
    print(f"  {name:<22} R²={m['r2']:.4f}  MAE={m['mae']:.1f} kcal  "
          f"RMSE={m['rmse']:.1f} kcal  MAPE={m['mape']:.2f}%  EVS={m['evs']:.4f}")

all_metrics = {}

# ── 3. XGBoost (existing — re-evaluate only) ────────────────────────────────
print("── XGBoost (existing model) ─────────────────────────────────────────")
xgb_model = joblib.load("xgb_calories_model.joblib")
xgb_pred  = xgb_model.predict(X_test)
xgb_metrics = compute_metrics(y_test, xgb_pred)
all_metrics["XGBoost"] = xgb_metrics
print_metrics("XGBoost", xgb_metrics)

# Best params from original Optuna run (train_model.ipynb)
xgb_params = {
    "n_estimators":    200,
    "learning_rate":   0.08383559651229099,
    "max_depth":       4,
    "subsample":       0.9230641116176058,
    "colsample_bytree":0.8049987865922269,
    "reg_alpha":       0.20462535330061787,
    "reg_lambda":      0.3210834295193818,
}

# ── 4. Random Forest ─────────────────────────────────────────────────────────
print("\n── Random Forest (Optuna 20 trials) ────────────────────────────────")
def rf_objective(trial):
    m = RandomForestRegressor(
        n_estimators    = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth       = trial.suggest_int("max_depth", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features    = trial.suggest_float("max_features", 0.3, 1.0),
        random_state=42, n_jobs=-1)
    return -cross_val_score(m, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error").mean()

rf_study = optuna.create_study(direction="minimize")
rf_study.optimize(rf_objective, n_trials=20)
rf_best  = rf_study.best_params
rf_model = RandomForestRegressor(**rf_best, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_metrics = compute_metrics(y_test, rf_model.predict(X_test))
all_metrics["Random Forest"] = rf_metrics
joblib.dump(rf_model, "rf_calories_model.joblib")
print_metrics("Random Forest", rf_metrics)
print(f"  Best params: {rf_best}")

# ── 5. Extra Trees ───────────────────────────────────────────────────────────
print("\n── Extra Trees (Optuna 20 trials) ──────────────────────────────────")
def et_objective(trial):
    m = ExtraTreesRegressor(
        n_estimators    = trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth       = trial.suggest_int("max_depth", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features    = trial.suggest_float("max_features", 0.3, 1.0),
        random_state=42, n_jobs=-1)
    return -cross_val_score(m, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error").mean()

et_study = optuna.create_study(direction="minimize")
et_study.optimize(et_objective, n_trials=20)
et_best  = et_study.best_params
et_model = ExtraTreesRegressor(**et_best, random_state=42, n_jobs=-1)
et_model.fit(X_train, y_train)
et_metrics = compute_metrics(y_test, et_model.predict(X_test))
all_metrics["Extra Trees"] = et_metrics
joblib.dump(et_model, "et_calories_model.joblib")
print_metrics("Extra Trees", et_metrics)
print(f"  Best params: {et_best}")

# ── 6. Gradient Boosting ────────────────────────────────────────────────────
print("\n── Gradient Boosting (Optuna 20 trials) ────────────────────────────")
def gb_objective(trial):
    m = GradientBoostingRegressor(
        n_estimators  = trial.suggest_int("n_estimators", 100, 400, step=50),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth     = trial.suggest_int("max_depth", 3, 8),
        subsample     = trial.suggest_float("subsample", 0.5, 1.0),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        max_features  = trial.suggest_float("max_features", 0.3, 1.0),
        random_state=42)
    return -cross_val_score(m, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error").mean()

gb_study = optuna.create_study(direction="minimize")
gb_study.optimize(gb_objective, n_trials=20)
gb_best  = gb_study.best_params
gb_model = GradientBoostingRegressor(**gb_best, random_state=42)
gb_model.fit(X_train, y_train)
gb_metrics = compute_metrics(y_test, gb_model.predict(X_test))
all_metrics["Gradient Boosting"] = gb_metrics
joblib.dump(gb_model, "gb_calories_model.joblib")
print_metrics("Gradient Boosting", gb_metrics)
print(f"  Best params: {gb_best}")

# ── 7. Stacking (RF + ET + GB + XGB  →  Ridge meta) ─────────────────────────
print("\n── Stacking (uses best params from above, Ridge meta-learner) ───────")
estimators = [
    ("xgb", xgb.XGBRegressor(**xgb_params, random_state=42)),
    ("rf",  RandomForestRegressor(**rf_best, random_state=42, n_jobs=-1)),
    ("et",  ExtraTreesRegressor(**et_best,  random_state=42, n_jobs=-1)),
    ("gb",  GradientBoostingRegressor(**gb_best, random_state=42)),
]
stack_model = StackingRegressor(
    estimators=estimators, final_estimator=Ridge(), cv=5, n_jobs=-1)
stack_model.fit(X_train, y_train)
stack_metrics = compute_metrics(y_test, stack_model.predict(X_test))
all_metrics["Stacking"] = stack_metrics
joblib.dump(stack_model, "stack_calories_model.joblib")
print_metrics("Stacking", stack_metrics)

# ── 8. Identify Best Model & Save Metrics ────────────────────────────────────
best_model_name = max(
    (k for k in all_metrics if k != "best_model"),
    key=lambda k: all_metrics[k]["r2"])
all_metrics["best_model"] = best_model_name

with open("model_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "="*60)
print(f"  ✅ All models trained and saved!")
print(f"  🏆 Best model: {best_model_name}  "
      f"(R²={all_metrics[best_model_name]['r2']:.4f})")
print("="*60)
print("\nFull Leaderboard:")
ranked = sorted(
    [(k, v) for k, v in all_metrics.items() if k != "best_model"],
    key=lambda x: x[1]["r2"], reverse=True)
for i, (name, m) in enumerate(ranked, 1):
    tag = " 🏆" if name == best_model_name else ""
    print(f"  #{i} {name:<22}{tag}")
    print(f"     R²={m['r2']}  MAE={m['mae']} kcal  "
          f"RMSE={m['rmse']} kcal  MAPE={m['mape']}%  EVS={m['evs']}")

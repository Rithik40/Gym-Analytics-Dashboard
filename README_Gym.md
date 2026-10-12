# 🏋️ Gym Analytics Dashboard
### Ensemble ML + Explainable AI · Calorie Burn Predictor

A production-ready, end-to-end machine learning web application that predicts **calories burned per workout session** based on physiological metrics and lifestyle data. Built with a 5-model ensemble, Optuna-tuned hyperparameters, SHAP-based explainability, and an interactive Flask dashboard.

---

## 🚀 Features

- **Multi-Model Prediction** — Choose from 5 trained regressors at inference time: XGBoost, Random Forest, Extra Trees, Gradient Boosting, and a Stacking ensemble.
- **Optuna Hyperparameter Tuning** — Each model is automatically optimized with 20 Bayesian trials using Optuna, minimizing cross-validated MSE.
- **SHAP Explainability** — Per-prediction SHAP value attribution so you understand *why* the model made its estimate.
- **Feature Importance Charts** — Visualize which input features drive model predictions most (for tree-based models).
- **Sensitivity Analysis** — Automatically sweeps ±10% on every numeric input to show how each feature affects the predicted output.
- **Live Model Leaderboard** — All 5 models' metrics (R², MAE, RMSE, MAPE, EVS) displayed side-by-side for comparison.
- **Interactive Dashboard** — Chart.js-powered visualizations embedded in a clean Flask frontend.
- **Input Validation** — Server-side range checks on all 16 input features with descriptive error messages.

---

## 📊 Model Performance

| Model | R² | MAE (kcal) | RMSE (kcal) | MAPE (%) | EVS |
|---|---|---|---|---|---|
| **XGBoost** 🏆 | **0.9965** | **12.18** | **16.97** | **1.38** | **0.9966** |
| Stacking | 0.9965 | 12.35 | 16.98 | 1.40 | 0.9966 |
| Gradient Boosting | 0.9947 | 15.24 | 21.06 | 1.68 | 0.9947 |
| Random Forest | 0.9705 | 37.55 | 49.57 | 4.17 | 0.9708 |
| Extra Trees | 0.9667 | 41.48 | 52.67 | 4.66 | 0.9669 |

> Evaluated on a held-out 20% test split (random_state=42). XGBoost is automatically selected as the default model.

---

## 🗂️ Project Structure

```
Gym Project/
│
├── app.py                      # Flask application — routes, inference, SHAP, sensitivity
├── train_all_models.py         # Training script — Optuna tuning for all 5 models
├── predict.py                  # Standalone prediction utility
├── train_model.ipynb           # Exploratory training notebook (XGBoost baseline)
│
├── model_metrics.json          # Auto-generated metrics + best model identifier
├── label_encoders.joblib       # Saved LabelEncoders for Gender, Workout_Type, Diet_Type
│
├── xgb_calories_model.joblib   # Trained XGBoost model
├── rf_calories_model.joblib    # Trained Random Forest model
├── et_calories_model.joblib    # Trained Extra Trees model
├── gb_calories_model.joblib    # Trained Gradient Boosting model
├── stack_calories_model.joblib # Trained Stacking ensemble
├── calories_model.joblib       # Legacy single-model file
│
├── templates/
│   └── index.html              # Jinja2 frontend with Chart.js visualizations
├── static/
│   └── styles.css              # Application styles
│
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- The gym dataset CSV (see [Dataset](#-dataset) section)

### 1. Clone / Download the project

```bash
# Navigate to the project directory
cd "Gym Project"
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models *(skip if `.joblib` files already exist)*

Update the `DATASET_PATH` variable in `train_all_models.py` to point to your local copy of the dataset, then run:

```bash
python train_all_models.py
```

This will:
- Load and preprocess the dataset
- Run Optuna (20 trials each) for RF, ET, and GB
- Train all 5 models and save them as `.joblib` files
- Save `model_metrics.json` with the leaderboard

### 5. Run the Flask application

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 📦 Dataset

| Field | Details |
|---|---|
| **Name** | Gym Members Exercise Tracking |
| **Source** | Kaggle (`gym_members_exercise_tracking3.csv`) |
| **Target** | `Calories_Burned` (continuous, regression) |
| **Train / Test Split** | 80% / 20% (stratified by random_state=42) |

### Input Features (16 total)

| Feature | Type | Range / Values |
|---|---|---|
| Age | Numeric | 1 – 120 |
| Gender | Categorical | Male, Female |
| Weight (kg) | Numeric | 0 – 300 |
| Height (m) | Numeric | 0 – 3 |
| Max BPM | Numeric | 100 – 240 |
| Avg BPM | Numeric | 60 – 200 |
| Resting BPM | Numeric | 30 – 120 |
| Workout Type | Categorical | Yoga, Cardio, Strength, HIIT |
| Fat Percentage | Numeric | 0 – 50 |
| Water Intake (liters) | Numeric | 0 – 10 |
| Workout Frequency (days/week) | Numeric | 1 – 7 |
| Experience Level | Ordinal | 1 (Beginner), 2 (Intermediate), 3 (Advanced) |
| BMI | Numeric | 10 – 50 |
| Diet Type | Categorical | High Protein, Paleo, Keto, Mediterranean, Balanced, Vegan, Low Carb, Gluten-Free, Vegetarian, Low Fat, Intermittent Fasting |
| Sleep Duration (hours) | Numeric | 0 – 24 |
| Session Duration (minutes) | Numeric | 0 – 600 |

---

## 🧠 ML Pipeline

```
Raw CSV
   │
   ▼
LabelEncoding (Gender, Workout_Type, Diet_Type)
   │
   ▼
Train / Test Split  (80% / 20%, random_state=42)
   │
   ├──► XGBoost      (Optuna-tuned in train_model.ipynb, re-evaluated here)
   ├──► Random Forest (Optuna 20 trials → best params → fit)
   ├──► Extra Trees   (Optuna 20 trials → best params → fit)
   ├──► Gradient Boosting (Optuna 20 trials → best params → fit)
   └──► Stacking      (XGB + RF + ET + GB → Ridge meta-learner, CV=5)
   │
   ▼
Evaluate: R², MAE, RMSE, MAPE, EVS
   │
   ▼
Save: *.joblib + model_metrics.json
   │
   ▼
Flask App  →  SHAP · Feature Importance · Sensitivity Analysis
```

### Stacking Ensemble Architecture

The stacking model uses all four base learners as level-0 estimators with their Optuna-optimized hyperparameters, and a **Ridge regression** meta-learner trained on their out-of-fold predictions (5-fold CV).

---

## 📈 Explainability Features

### SHAP Values
- **XGBoost**: Uses native `predict(pred_contribs=True)` for exact SHAP values via DMatrix.
- **Other tree models**: Uses `shap.TreeExplainer` for fast, model-specific attributions.
- Output: Per-feature signed contribution showing exactly how each input nudged the prediction above or below the baseline.

### Feature Importance
- Displayed as a bar chart using `model.feature_importances_` (available for all tree-based models except Stacking).

### Sensitivity Analysis
- For every numeric feature, the model is re-run with the feature perturbed by ±10% (min delta of ±0.5).
- Displays the resulting calorie change as an up/down delta chart.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| ML Models | XGBoost, scikit-learn (RF, ET, GB, Stacking) |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Frontend Visualizations | Chart.js |
| Templating | Jinja2 |

---

## 📄 Dependencies

Key packages (see `requirements.txt` for full list):

```
flask
scikit-learn
xgboost
optuna
shap
pandas
numpy
joblib
```

---

## 🔧 Configuration

All model file paths are declared at the top of `app.py` in the `MODEL_FILES` registry — add or swap models by editing that dict. No other changes needed.

```python
MODEL_FILES = {
    "XGBoost":           "xgb_calories_model.joblib",
    "Random Forest":     "rf_calories_model.joblib",
    "Extra Trees":       "et_calories_model.joblib",
    "Gradient Boosting": "gb_calories_model.joblib",
    "Stacking":          "stack_calories_model.joblib",
}
```

---

## 👤 Author

**Rithik B**  
Integrated M.Sc. Data Science · Amrita Vishwa Vidyapeetham  
rithikrithik222@gmail.com

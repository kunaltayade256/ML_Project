from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = Path("data/ai4i2020.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Machine failure"

LEAKAGE_COLUMNS = [
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH.resolve()}\n"
        "Put your downloaded CSV in the data folder "
        "and name it ai4i2020.csv."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required = FEATURES + [TARGET]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Found columns: {df.columns.tolist()}"
    )

df = df.drop_duplicates().reset_index(drop=True)

# The main target must be binary.
df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Machine failure must contain only 0 and 1.")

df[TARGET] = df[TARGET].astype(int)

# Keep only model features and target.
# IDs and failure subtype labels are excluded.
X = df[FEATURES].copy()
y = df[TARGET].copy()

for col in NUMERIC_FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X["Type"] = X["Type"].astype("string").str.strip()

if not set(X["Type"].dropna().unique()).issubset({"L", "M", "H"}):
    raise ValueError("Unexpected values in Type column.")

print("\nDATASET SUMMARY")
print("-" * 50)
print("Rows:", len(df))
print("Features:", FEATURES)
print("Failure counts:")
print(y.value_counts().sort_index())
print("Failure rate:", round(y.mean() * 100, 3), "%")
print("Missing feature values:", int(X.isna().sum().sum()))

if y.value_counts().min() < 3:
    raise ValueError("Not enough samples in one target class.")

# --------------------------------------------------
# 3. Split data into train / validation / test
# --------------------------------------------------

# Stratification preserves the rare failure proportion.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print("\nSPLIT SIZES")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

def make_preprocessor(scale_numeric):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())
class_weight_ratio = negative / max(positive, 1)

models = {
    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor(False)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", make_preprocessor(False)),
        ("classifier", XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=class_weight_ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "SVM": Pipeline([
        ("preprocessor", make_preprocessor(True)),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
}

# --------------------------------------------------
# 6. Train and compare using validation data
# --------------------------------------------------

comparison = {}
trained_models = {}
best_model_name = None
best_val_ap = -1.0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    val_ap = average_precision_score(y_val, val_scores)

    # Choose threshold using validation F1.
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_scores
    )

    if len(thresholds):
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        threshold = float(thresholds[np.argmax(f1_values)])
    else:
        threshold = 0.5

    val_predictions = (val_scores >= threshold).astype(int)

    comparison[name] = {
        "validation_average_precision": float(val_ap),
        "validation_f1": float(
            f1_score(y_val, val_predictions, zero_division=0)
        ),
        "validation_threshold": threshold,
    }
    trained_models[name] = model

    print("Validation average precision:", round(val_ap, 4))
    print("Validation F1:", round(
        comparison[name]["validation_f1"], 4
    ))
    print("Validation threshold:", round(threshold, 4))

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_model_name = name

# --------------------------------------------------
# 7. Evaluate selected model once on the test set
# --------------------------------------------------

best_model = trained_models[best_model_name]
best_threshold = comparison[best_model_name][
    "validation_threshold"
]

test_scores = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_scores >= best_threshold).astype(int)

test_metrics = {
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(precision_score(
        y_test, test_predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, test_predictions, zero_division=0
    )),
    "f1": float(f1_score(
        y_test, test_predictions, zero_division=0
    )),
    "average_precision": float(
        average_precision_score(y_test, test_scores)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_scores)),
    "confusion_matrix": confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).tolist(),
}

print("\nSELECTED MODEL:", best_model_name)
print("\nFINAL TEST METRICS")
for metric, value in test_metrics.items():
    print(metric, ":", value)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# --------------------------------------------------
# 8. Save the model, metrics, and test predictions
# --------------------------------------------------

joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")

metadata = {
    "model_name": best_model_name,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target": TARGET,
    "threshold": best_threshold,
    "validation_comparison": comparison,
    "test_metrics": test_metrics,
    "dataset_rows": int(len(df)),
    "failure_rate_percent": float(y.mean() * 100),
}

(ARTIFACT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

test_output = X_test.reset_index(drop=True).copy()
test_output["actual_failure"] = y_test.reset_index(drop=True)
test_output["failure_score"] = test_scores
test_output["predicted_failure"] = test_predictions

test_output.to_csv(
    ARTIFACT_DIR / "test_predictions.csv", index=False
)

# Save feature importance for tree models.
classifier = best_model.named_steps["classifier"]
preprocessor = best_model.named_steps["preprocessor"]

if hasattr(classifier, "feature_importances_"):
    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        ARTIFACT_DIR / "feature_importance.csv",
        index=False
    )

print("\nTraining complete!")
print("Model saved to artifacts/best_model.joblib")
print("Metrics saved to artifacts/metadata.json")
print("Test predictions saved to artifacts/test_predictions.csv")
print("Validation comparison:", comparison)
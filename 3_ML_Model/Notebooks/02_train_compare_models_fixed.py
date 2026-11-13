"""
Step 2: Train and compare three machine learning models.

The questionnaire responses are text labels (for example, "Agree").
This script converts them to the corresponding 1-to-5 Likert values
before training the models.

Important:
The dataset and target categories are synthetic and illustrative.
Results demonstrate a machine learning workflow only. They do not
validate real-world or clinical wellbeing prediction.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "3_ML_Model" / "Processed_Data"
MODEL_DIR = PROJECT_ROOT / "3_ML_Model" / "Model_Outputs"
EVAL_DIR = PROJECT_ROOT / "3_ML_Model" / "Model_Evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "illustrative_rule_based_band"

# Convert the five questionnaire response labels into numeric values.
LIKERT_MAPPING = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5
}

print("Loading prepared training and testing data...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET_COLUMN]
y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET_COLUMN]

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Input features: {X_train.shape[1]}")

print("\nConverting questionnaire responses from text to numeric values...")
X_train = X_train.replace(LIKERT_MAPPING)
X_test = X_test.replace(LIKERT_MAPPING)

# Confirm that all questionnaire values are numeric.
if X_train.isna().any().any() or X_test.isna().any().any():
    raise ValueError(
        "Some questionnaire responses could not be converted. "
        "Please check the response labels in the dataset."
    )

print("Response conversion completed.")
print("Example numeric responses:")
print(X_train.head())

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=6,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

results = []
reports = {}

print("\nTraining and evaluating models...")

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    }
    results.append(metrics)

    reports[model_name] = classification_report(
        y_test, predictions, zero_division=0
    )

    safe_name = model_name.lower().replace(" ", "_")
    joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )
    matrix_df.to_csv(EVAL_DIR / f"{safe_name}_confusion_matrix.csv")

results_df = pd.DataFrame(results).sort_values(
    by="F1_macro", ascending=False
)
results_df.to_csv(EVAL_DIR / "model_comparison.csv", index=False)

with open(EVAL_DIR / "classification_reports.txt", "w", encoding="utf-8") as file:
    for model_name, report in reports.items():
        file.write(f"\n{'=' * 65}\n")
        file.write(f"{model_name}\n")
        file.write(f"{'=' * 65}\n")
        file.write(report)
        file.write("\n")

best_model_name = results_df.iloc[0]["Model"]
summary = {
    "best_model_by_macro_f1": best_model_name,
    "selection_metric": "F1_macro",
    "note": (
        "This comparison uses synthetic data and an illustrative "
        "rule-derived target. It is not evidence of clinical validity "
        "or real-world predictive performance."
    )
}
with open(EVAL_DIR / "model_selection_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nBEST MODEL BY MACRO F1:")
print(best_model_name)
print(f"\nTrained models saved to: {MODEL_DIR}")
print(f"Evaluation results saved to: {EVAL_DIR}")
print("\nSTEP 2 COMPLETED SUCCESSFULLY!")

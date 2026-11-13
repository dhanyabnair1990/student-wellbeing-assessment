"""
Step 2: Train and compare three machine learning models.

Important:
This project uses synthetic questionnaire data and an illustrative,
rule-based target category. Model results demonstrate a machine learning
workflow; they do not validate real-world or clinical wellbeing prediction.
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

# Find the project folder from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "3_ML_Model" / "Processed_Data"
MODEL_DIR = PROJECT_ROOT / "3_ML_Model" / "Model_Outputs"
EVAL_DIR = PROJECT_ROOT / "3_ML_Model" / "Model_Evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "illustrative_rule_based_band"

print("Loading prepared training and testing data...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET_COLUMN]
y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET_COLUMN]

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Input features: {X_train.shape[1]}")

# Define the three models.
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

    # Save each trained model.
    safe_name = model_name.lower().replace(" ", "_")
    joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    # Save confusion matrix for this model.
    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )
    matrix_df.to_csv(EVAL_DIR / f"{safe_name}_confusion_matrix.csv")

    print(f"Completed: {model_name}")

# Save comparison metrics.
results_df = pd.DataFrame(results).sort_values(
    by="F1_macro", ascending=False
)
results_df.to_csv(EVAL_DIR / "model_comparison.csv", index=False)

# Save detailed classification reports.
with open(EVAL_DIR / "classification_reports.txt", "w", encoding="utf-8") as file:
    for model_name, report in reports.items():
        file.write(f"\n{'=' * 65}\n")
        file.write(f"{model_name}\n")
        file.write(f"{'=' * 65}\n")
        file.write(report)
        file.write("\n")

# Save the name of the model with the highest macro F1 score.
best_model_name = results_df.iloc[0]["Model"]
summary = {
    "best_model_by_macro_f1": best_model_name,
    "selection_metric": "F1_macro",
    "note": (
        "This is a comparison on synthetic data with an illustrative "
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

print("\nFiles saved:")
print(f"Trained models: {MODEL_DIR}")
print(f"Evaluation results: {EVAL_DIR}")
print("\nSTEP 2 COMPLETED SUCCESSFULLY!")

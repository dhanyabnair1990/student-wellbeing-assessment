"""
Step 1: Prepare the student wellbeing data for machine learning.

Important:
- The dataset is synthetic.
- The target is an illustrative, rule-based category.
- This model will learn the scoring rules used to create that category.
  It is not a clinically validated wellbeing assessment.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Locate the main project folder automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "1_Dataset"
    / "Raw_Data"
    / "student_wellbeing_synthetic_2000.csv"
)

TARGET_COLUMN = "illustrative_rule_based_band"
QUESTION_COLUMNS = [f"Q{i:02d}" for i in range(1, 21)]

print("Loading the student wellbeing dataset...")
print(f"Dataset path: {DATA_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Please check that "
        "'student_wellbeing_synthetic_2000.csv' is inside "
        "1_Dataset/Raw_Data."
    )

df = pd.read_csv(DATA_FILE)

# Check that all required columns exist.
required_columns = QUESTION_COLUMNS + [TARGET_COLUMN]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"These required columns are missing: {missing_columns}")

# Use only the 20 questionnaire responses as model inputs.
# Do not include pre-calculated scores or the target in X.
X = df[QUESTION_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

print("\nINPUT FEATURES")
print(f"Number of questionnaire features: {X.shape[1]}")
print(f"Feature columns: {', '.join(X.columns)}")

print("\nTARGET")
print(f"Target column: {TARGET_COLUMN}")
print("\nTarget category counts:")
print(y.value_counts())

# Check for missing values before splitting.
print("\nMISSING VALUES")
print(f"Missing values in input features: {int(X.isna().sum().sum())}")
print(f"Missing values in target: {int(y.isna().sum())}")

if X.isna().any().any() or y.isna().any():
    raise ValueError(
        "Missing values were found. We should handle them before training."
    )

# Keep the same category proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the prepared training and testing data.
OUTPUT_DIR = PROJECT_ROOT / "3_ML_Model" / "Processed_Data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_frame(name=TARGET_COLUMN).to_csv(
    OUTPUT_DIR / "y_train.csv", index=False
)
y_test.to_frame(name=TARGET_COLUMN).to_csv(
    OUTPUT_DIR / "y_test.csv", index=False
)

print("\nDATA SPLIT COMPLETED")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Training data saved to: {OUTPUT_DIR}")
print("\nSTEP 1 COMPLETED SUCCESSFULLY!")

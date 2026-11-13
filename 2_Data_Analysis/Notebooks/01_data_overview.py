# ============================================================
# AI-POWERED STUDENT WELLBEING ASSESSMENT AND SUPPORT SYSTEM
# Step 1: Data Overview
# ============================================================

import pandas as pd
from pathlib import Path

print("=" * 65)
print("STUDENT WELLBEING ASSESSMENT - DATA OVERVIEW")
print("=" * 65)

# 1. Locate the main project folder
project_folder = Path(__file__).resolve().parents[2]

# 2. Locate the dataset folder
dataset_folder = project_folder / "1_Dataset"

# Search for the dataset inside 1_Dataset and its subfolders
matching_files = list(
    dataset_folder.rglob("student_wellbeing_synthetic_2000.csv")
)

if not matching_files:
    raise FileNotFoundError(
        "Dataset not found. Please check the 1_Dataset folder."
    )

file_path = matching_files[0]

print("\nDataset location:")
print(file_path)

# 3. Load the dataset
df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

# 4. Display first five records
print("\nFIRST FIVE RECORDS:")
print(df.head())

# 5. Dataset dimensions
print("\nDATASET DIMENSIONS:")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 6. Column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# 7. Dataset information
print("\nDATASET INFORMATION:")
df.info()

# 8. Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 9. Summary statistics
print("\nSUMMARY STATISTICS:")
print(df.describe(include="all"))

print("\n" + "=" * 65)
print("DATA OVERVIEW COMPLETED SUCCESSFULLY")
print("=" * 65)
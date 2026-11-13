# ============================================================
# AI-POWERED STUDENT WELLBEING ASSESSMENT
# Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 65)
print("STUDENT WELLBEING - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# ------------------------------------------------------------
# 1. FIND AND LOAD THE DATASET
# ------------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

dataset_name = "student_wellbeing_synthetic_2000.csv"

matching_files = list(project_root.rglob(dataset_name))

if not matching_files:
    raise FileNotFoundError(
        f"Dataset not found: {dataset_name}. "
        "Please check the 1_Dataset folder."
    )

dataset_path = matching_files[0]

df = pd.read_csv(dataset_path)

print("\n1. DATASET LOADED SUCCESSFULLY")
print("Dataset location:", dataset_path)
print("Number of records:", df.shape[0])
print("Number of columns:", df.shape[1])


# ------------------------------------------------------------
# 2. CHECK DATA QUALITY
# ------------------------------------------------------------

print("\n2. DATA QUALITY CHECK")

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes.value_counts())


# ------------------------------------------------------------
# 3. DEMOGRAPHIC ANALYSIS
# ------------------------------------------------------------

print("\n3. DEMOGRAPHIC ANALYSIS")

print("\nAge distribution:")
print(df["age"].value_counts().sort_index())

print("\nClass distribution:")
print(df["class"].value_counts())

print("\nGender distribution:")
print(df["gender"].value_counts())


# ------------------------------------------------------------
# 4. WELLBEING DIMENSION ANALYSIS
# ------------------------------------------------------------

dimension_columns = [
    "emotional_wellbeing_score",
    "academic_wellbeing_score",
    "family_and_relationships_score",
    "friendships_and_social_wellbeing_score",
    "physical_wellbeing_score",
    "self-image_and_confidence_score",
    "digital_wellbeing_score"
]

dimension_names = [
    "Emotional",
    "Academic",
    "Family & Relationships",
    "Friendships & Social",
    "Physical",
    "Self-image & Confidence",
    "Digital"
]

dimension_averages = df[dimension_columns].mean()

print("\n4. AVERAGE SCORE BY WELLBEING DIMENSION")

for name, column in zip(dimension_names, dimension_columns):
    print(f"{name}: {dimension_averages[column]:.2f}")


# ------------------------------------------------------------
# 5. OVERALL WELLBEING ANALYSIS
# ------------------------------------------------------------

print("\n5. OVERALL WELLBEING ANALYSIS")

overall_average = df[
    "overall_wellbeing_index_illustrative"
].mean()

print("Average overall wellbeing index:", round(overall_average, 2))

print("\nIllustrative wellbeing category distribution:")

band_counts = df[
    "illustrative_rule_based_band"
].value_counts()

print(band_counts)


# ------------------------------------------------------------
# 6. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

output_folder = project_root / "2_Data_Analysis" / "Visualisations"

output_folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 7. CHART 1 - AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

age_counts = df["age"].value_counts().sort_index()

plt.bar(age_counts.index.astype(str), age_counts.values)

plt.title("Student Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(output_folder / "01_age_distribution.png")
plt.close()


# ------------------------------------------------------------
# 8. CHART 2 - WELLBEING DIMENSION COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

averages = dimension_averages.values

plt.barh(dimension_names, averages)

plt.title("Average Score Across Wellbeing Dimensions")
plt.xlabel("Average Score (1 to 5)")
plt.xlim(0, 5)
plt.tight_layout()

plt.savefig(output_folder / "02_wellbeing_dimension_comparison.png")
plt.close()


# ------------------------------------------------------------
# 9. CHART 3 - WELLBEING CATEGORY DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    band_counts.index,
    band_counts.values
)

plt.title("Distribution of Illustrative Wellbeing Categories")
plt.xlabel("Illustrative Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(output_folder / "03_wellbeing_category_distribution.png")
plt.close()


# ------------------------------------------------------------
# 10. CHART 4 - QUESTION RESPONSE PATTERNS
# ------------------------------------------------------------

question_columns = [
    f"Q{i:02d}_score" for i in range(1, 21)
]

question_averages = df[question_columns].mean()

plt.figure(figsize=(12, 6))

plt.bar(
    question_averages.index,
    question_averages.values
)

plt.title("Average Score Across 20 Questionnaire Statements")
plt.xlabel("Question")
plt.ylabel("Average Score (1 to 5)")
plt.ylim(0, 5)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(output_folder / "04_question_response_patterns.png")
plt.close()


# ------------------------------------------------------------
# 11. CHART 5 - CORRELATION BETWEEN DIMENSIONS
# ------------------------------------------------------------

correlation = df[dimension_columns].corr()

plt.figure(figsize=(10, 8))

plt.imshow(correlation, vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(dimension_names)),
    dimension_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(dimension_names)),
    dimension_names
)

plt.title("Correlation Between Wellbeing Dimensions")
plt.tight_layout()

plt.savefig(output_folder / "05_dimension_correlation.png")
plt.close()


# ------------------------------------------------------------
# 12. SAVE SUMMARY REPORTS
# ------------------------------------------------------------

dimension_summary = pd.DataFrame({
    "Wellbeing Dimension": dimension_names,
    "Average Score": averages
})

dimension_summary.to_csv(
    output_folder / "dimension_summary.csv",
    index=False
)

band_counts.to_csv(
    output_folder / "wellbeing_category_summary.csv"
)


# ------------------------------------------------------------
# 13. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nCharts and summary files saved in:")
print(output_folder)

print("\nFiles generated:")
print("1. Age distribution")
print("2. Wellbeing dimension comparison")
print("3. Wellbeing category distribution")
print("4. Question response patterns")
print("5. Dimension correlation")
print("6. Dimension summary CSV")
print("7. Wellbeing category summary CSV")

print("\nEDA PROCESS COMPLETED.")
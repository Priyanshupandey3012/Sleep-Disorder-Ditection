# ============================================================
#   SLEEP DISORDER DETECTION - DATA PREPROCESSING PIPELINE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
import sys
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

df = pd.read_excel("sleep_disorder_merged_dataset_v2.xlsx", sheet_name="Sleep_Dataset",header = 1)
df.columns = df.columns.str.replace('\n', '_').str.strip()
print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))


# ============================================================
# STEP 2: BASIC INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: BASIC INSPECTION")
print("=" * 60)

print("\n-- Data Types --")
print(df.dtypes)

print("\n-- Missing Values --")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ok")

print("\n-- Duplicates --")
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"  → Dropped. New shape: {df.shape}")

print("\n-- Class Distribution (Target) --")
print(df['Sleep_Disorder_Type'].value_counts())
print(df['Diagnosis_Confirmed'].value_counts())


# ============================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: HANDLE MISSING VALUES")
print("=" * 60)

# Numeric columns → fill with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' with median: {median_val}")

# Categorical columns → fill with mode
# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        # Special case: Sleep_Disorder_Type NaN means "None" (no disorder)
        if col == 'Sleep_Disorder_Type':
            df[col].fillna('None', inplace=True)
            print(f"  Filled 'Sleep_Disorder_Type' NaN with 'None' (no disorder)")
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' with mode: {mode_val}")

print("Missing values handled ok")


# ============================================================
# STEP 4: DROP UNNECESSARY COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: DROP UNNECESSARY COLUMNS")
print("=" * 60)

# Patient_ID is just an identifier, not a feature
# Blood_Pressure_mmHg is already split into Systolic_BP and Diastolic_BP
drop_cols = ['Patient_ID', 'Blood_Pressure_mmHg']
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped: {drop_cols}")
print(f"Remaining columns: {df.shape[1]}")


# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 60)

# -- 5a. Sleep Efficiency Score (sleep quality + duration combined)
df['Sleep_Efficiency_Score'] = (
    (df['Sleep_Duration_hrs'] / 9.5) * 0.5 +
    (df['Quality_of_Sleep_1_10'] / 10) * 0.5
) * 100
df['Sleep_Efficiency_Score'] = df['Sleep_Efficiency_Score'].round(2)
print("Created: Sleep_Efficiency_Score ok")

# -- 5b. Clinical Risk Score (AHI + inverse SaO2 + Stress)
df['Clinical_Risk_Score'] = (
    (df['AHI_Score'] / 60) * 0.4 +
    ((100 - df['SaO2_Level_pct']) / 15) * 0.4 +
    (df['Stress_Level_1_10'] / 10) * 0.2
) * 100
df['Clinical_Risk_Score'] = df['Clinical_Risk_Score'].round(2)
print("Created: Clinical_Risk_Score ok")

# -- 5c. Activity Index (steps + physical activity combined)
df['Activity_Index'] = (
    (df['Daily_Steps'] / 15000) * 0.5 +
    (df['Physical_Activity_min_day'] / 90) * 0.5
) * 100
df['Activity_Index'] = df['Activity_Index'].round(2)
print("Created: Activity_Index ok")

# -- 5d. Blood Pressure Category
def classify_bp(row):
    s, d = row['Systolic_BP'], row['Diastolic_BP']
    if s < 120 and d < 80:
        return 'Normal'
    elif s < 130 and d < 80:
        return 'Elevated'
    elif s < 140 or d < 90:
        return 'High_Stage1'
    else:
        return 'High_Stage2'

df['BP_Category'] = df.apply(classify_bp, axis=1)
print("Created: BP_Category ok")

# -- 5e. Age Group
def age_group(age):
    if age < 30:
        return 'Young'
    elif age < 45:
        return 'Middle'
    elif age < 60:
        return 'Senior'
    else:
        return 'Elderly'

df['Age_Group'] = df['Age'].apply(age_group)
print("Created: Age_Group ok")

# -- 5f. Wearable Risk Flag (low SpO2 or high movement at night)
df['Wearable_Risk_Flag'] = (
    (df['Wearable_SpO2_pct'] < 92) | (df['Wearable_Movement_Actigraphy'] > 7)
).astype(int)
print("Created: Wearable_Risk_Flag ok")

print(f"\nNew shape after feature engineering: {df.shape}")


# ============================================================
# STEP 6: ENCODE CATEGORICAL VARIABLES
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

le = LabelEncoder()

# -- Binary encoding
df['Gender_Encoded'] = le.fit_transform(df['Gender'])          # Male=1, Female=0
print("Binary encoded: Gender")

# -- Label encoding (ordinal)
bmi_order = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
df['BMI_Encoded'] = df['BMI_Category'].map(bmi_order)
print("Ordinal encoded: BMI_Category")

bp_order = {'Normal': 0, 'Elevated': 1, 'High_Stage1': 2, 'High_Stage2': 3}
df['BP_Encoded'] = df['BP_Category'].map(bp_order)
print("Ordinal encoded: BP_Category")

age_order = {'Young': 0, 'Middle': 1, 'Senior': 2, 'Elderly': 3}
df['Age_Group_Encoded'] = df['Age_Group'].map(age_order)
print("Ordinal encoded: Age_Group")

# -- One-hot encoding (nominal)
df = pd.get_dummies(df, columns=['Occupation'], prefix='Occ', drop_first=False)
print("One-hot encoded: Occupation")

# -- Target encoding
disorder_map = {
    'None': 0,
    'Insomnia': 1,
    'Sleep Apnea': 2,
    'Narcolepsy': 3,
    'Restless Leg Syndrome': 4
}
df['Disorder_Label'] = df['Sleep_Disorder_Type'].map(disorder_map)
print("Label encoded: Sleep_Disorder_Type → Disorder_Label")

# Drop original string columns (already encoded)
drop_str_cols = ['Gender', 'BMI_Category', 'BP_Category', 'Age_Group', 'Sleep_Disorder_Type']
df.drop(columns=drop_str_cols, inplace=True)
print(f"\nDropped original string columns: {drop_str_cols}")
print(f"Shape after encoding: {df.shape}")


# ============================================================
# STEP 7: FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: FEATURE SCALING")
print("=" * 60)

# Columns to NOT scale (binary flags, encoded ordinals, targets)
skip_scale = [
    'Diagnosis_Confirmed', 'Disorder_Label',
    'Gender_Encoded', 'BMI_Encoded', 'BP_Encoded',
    'Age_Group_Encoded', 'Wearable_Risk_Flag'
] + [col for col in df.columns if col.startswith('Occ_')]

scale_cols = [col for col in df.select_dtypes(include=[np.number]).columns
              if col not in skip_scale]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

print(f"Scaled {len(scale_cols)} columns using StandardScaler:")
for c in scale_cols:
    print(f"  → {c}")


# ============================================================
# STEP 8: DEFINE FEATURES & TARGETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: DEFINE FEATURES & TARGETS")
print("=" * 60)

# Check for any unmapped disorder labels
print("Unique disorder values:", df['Disorder_Label'].unique())
print("NaN count in Disorder_Label:", df['Disorder_Label'].isnull().sum())

# Fill any remaining NaN in Disorder_Label with 0 (None = no disorder)
df['Disorder_Label'] = df['Disorder_Label'].fillna(0).astype(int)
df['Diagnosis_Confirmed'] = df['Diagnosis_Confirmed'].fillna(0).astype(int)

print("After fix - NaN count:", df['Disorder_Label'].isnull().sum())
print("Disorder_Label distribution:")
print(df['Disorder_Label'].value_counts().sort_index())

# Multi-class target (5 disorder types)
X = df_scaled.drop(columns=['Diagnosis_Confirmed', 'Disorder_Label'])
y_multiclass = df_scaled['Disorder_Label'].fillna(0).astype(int)
y_binary     = df_scaled['Diagnosis_Confirmed'].fillna(0).astype(int)

print(f"\nFeatures (X) shape     : {X.shape}")
print(f"Multiclass target shape: {y_multiclass.shape} | Classes: {y_multiclass.nunique()}")
print(f"Binary target shape    : {y_binary.shape}     | Classes: {y_binary.nunique()}")


# ============================================================
# STEP 9: TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train_mc, y_test_mc = train_test_split(
    X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
)
_, _, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Training set  : {X_train.shape}")
print(f"Test set      : {X_test.shape}")
print(f"\nClass distribution in training set:")
print(pd.Series(y_train_mc).value_counts().sort_index()
      .rename({v: k for k, v in disorder_map.items()}))


# ============================================================
# STEP 10: HANDLE CLASS IMBALANCE (SMOTE)
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: HANDLE CLASS IMBALANCE (SMOTE)")
print("=" * 60)

print("Before SMOTE:")
print(pd.Series(y_train_mc).value_counts().sort_index())

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_mc)

print("\nAfter SMOTE:")
print(pd.Series(y_train_bal).value_counts().sort_index())
print(f"\nBalanced training set shape: {X_train_bal.shape}")


# ============================================================
# STEP 11: SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: SAVE PROCESSED FILES")
print("=" * 60)

# Save full processed dataframe
df_scaled.to_csv("sleep_processed_full.csv", index=False)
print("Saved: sleep_processed_full.csv ok")

# Save train/test splits
X_train_bal_df = pd.DataFrame(X_train_bal, columns=X.columns)
X_train_bal_df['Disorder_Label'] = y_train_bal
X_train_bal_df.to_csv("sleep_train.csv", index=False)
print("Saved: sleep_train.csv ok")

X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['Disorder_Label'] = y_test_mc.values
X_test_df.to_csv("sleep_test.csv", index=False)
print("Saved: sleep_test.csv ok")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Total records        : 1000")
print(f"  Final features       : {X.shape[1]}")
print(f"  Training samples     : {X_train_bal.shape[0]}  (after SMOTE)")
print(f"  Test samples         : {X_test.shape[0]}")
print(f"  Task 1 (Binary)      : Disorder present? (0/1)")
print(f"  Task 2 (Multiclass)  : Which disorder? (0–4)")
print(f"\n  Output files:")
print(f"    → sleep_processed_full.csv")
print(f"    → sleep_train.csv")
print(f"    → sleep_test.csv")
print("=" * 60)
print("  ok Ready for EDA & Model Training!")
print("=" * 60)

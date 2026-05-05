# preprocessing.py

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

# ============================================
# 1. Load Data
# ============================================
df = pd.read_csv(r'E:\Uni\AI\AI-Income-Project/Data/raw/trainincome_data.csv')

# ============================================
# 2. Initial Data Exploration
# ============================================
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
df.info()

print("\nMissing values:\n", df.isnull().sum())
print("\nInitial duplicates:", df.duplicated().sum())

# Display unique values for each column
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].unique())

# ============================================
# 3. Data Cleaning - Remove extra spaces
# ============================================
object_cols = df.select_dtypes(include=['object', 'string']).columns
for col in object_cols:
    df[col] = df[col].str.strip()

# ============================================
# 4. Remove Duplicates
# ============================================
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nDuplicates after drop: {df.duplicated().sum()}")
print(f"Shape after dropping duplicates: {df.shape}")

# ============================================
# 5. Handle Missing/Invalid Values ('?')
# ============================================
df.replace('?', np.nan, inplace=True)

for col in ['workclass', 'occupation', 'native-country']:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Missing values after fill: {df.isnull().values.any()}")

# ============================================
# 6. Handle Outliers
# ============================================
num_cols = df.select_dtypes(include='int64').columns
print(f"\nNumeric columns: {num_cols.tolist()}")

def count_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers.shape[0]

for col in num_cols:
    count = count_outliers_iqr(df, col)
    print(f"{col}: {count}")

# Apply log1p transformation to capital-gain and capital-loss
df['capital-gain'] = np.log1p(df['capital-gain'])
df['capital-loss'] = np.log1p(df['capital-loss'])

# Cap outliers for age and hours-per-week
for col in ['age', 'hours-per-week']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# Drop fnlwgt column (no correlation with target)
if 'fnlwgt' in df.columns:
    df.drop('fnlwgt', axis=1, inplace=True)
    print("\nColumn 'fnlwgt' dropped.")

print(f"\nDuplicates after handling outliers: {df.duplicated().sum()}")

# ============================================
# 7. Correlation Analysis
# ============================================
print("\nCorrelation matrix:")
print(df.select_dtypes(include="number").corr())

# ============================================
# 8. One-Hot Encoding for Categorical Columns
# ============================================
one_hot_cols = [
    'sex',
    'workclass',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'native-country'
]

df = pd.get_dummies(df, columns=one_hot_cols)
print(f"\nShape after one-hot encoding: {df.shape}")

# ============================================
# 9. Ordinal Encoding for Education Column
# ============================================
education_order = [
    'Preschool',
    '1st-4th',
    '5th-6th',
    '7th-8th',
    '9th',
    '10th',
    '11th',
    '12th',
    'HS-grad',
    'Some-college',
    'Assoc-voc',
    'Assoc-acdm',
    'Bachelors',
    'Masters',
    'Prof-school',
    'Doctorate'
]

encoder = OrdinalEncoder(categories=[education_order])
df['education'] = encoder.fit_transform(df[['education']])
print(f"Education unique values after encoding: {df['education'].unique()}")

# ============================================
# 10. Target Encoding (Income)
# ============================================
le = LabelEncoder()
df['Income'] = le.fit_transform(df['Income '])
df = df.drop('Income ', axis=1)  # Drop original column

print(f"\nTarget distribution: {df['Income'].value_counts().to_dict()}")

# ============================================
# 11. Separate Features and Target
# ============================================
X = df.drop('Income', axis=1)
y = df['Income']

# ============================================
# 12. Standard Scaling for Numeric Columns
# ============================================
numeric_cols = ['age', 'education', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X[numeric_cols])
X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)

X_categorical = X.drop(columns=numeric_cols)
X_final = pd.concat([X_numeric_scaled, X_categorical], axis=1)

print(f"\nFinal X shape: {X_final.shape}")
print(f"Final y shape: {y.shape}")

# ============================================
# 13. Verify No Missing Values
# ============================================
print(f"\nMissing values in X_final: {X_final.isnull().sum().sum()}")
print(f"Duplicates in X_final: {X_final.duplicated().sum()}")
print(f"Duplicates in y: {y.duplicated().sum()}")

# ============================================
# 14. Handle Duplicates with Different Target Values
# ============================================
print("\n" + "="*60)
print("Checking for duplicates with different target values...")
print("="*60)

Xy = pd.concat([X_final, y], axis=1)
Xy_dup = Xy[Xy.duplicated(subset=X_final.columns, keep=False)]

if len(Xy_dup) > 0:
    print(f"Found {len(Xy_dup)} rows that are duplicates in features with different target values")
    
    # Show statistics of target per duplicate group
    dup_counts = Xy_dup.groupby(list(X_final.columns))['Income'].nunique()
    print(f"\nGroups with multiple target values: {len(dup_counts[dup_counts > 1])}")
    
    # For each duplicate group, take the most frequent target value
    Xy_clean = Xy.groupby(list(X_final.columns))['Income'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
    
    X_final = Xy_clean.drop('Income', axis=1)
    y = Xy_clean['Income']
    
    print(f"\nShape after handling duplicates: X_final={X_final.shape}, y={y.shape}")
    print(f"Remaining duplicates in X_final: {X_final.duplicated().sum()}")
else:
    print("No duplicates found in X_final")

# ============================================
# 15. Verify Normalization
# ============================================
print("\n" + "="*60)
print("Verification of normalization:")
print("="*60)
numeric_check = X_final[numeric_cols]
print("Means (should be 0):")
print(numeric_check.mean().round(6))
print("\nStds (should be 1):")
print(numeric_check.std().round(6))

# ============================================
# 16. Final Data Types Overview
# ============================================
print("\n" + "="*60)
print("Final data types overview:")
print("="*60)
print(f"Boolean columns: {len(X_final.select_dtypes(include='bool').columns)}")
print(f"Float columns: {len(X_final.select_dtypes(include='float64').columns)}")
print(f"Integer columns: {len(X_final.select_dtypes(include='int64').columns)}")
print(f"Total columns: {len(X_final.columns)}")

# Ensure no object columns remain
object_cols_final = X_final.select_dtypes(include='object').columns
if len(object_cols_final) == 0:
    print("\n No object columns remaining - all features are numeric")
else:
    print(f"\n Warning: Still have object columns: {object_cols_final.tolist()}")

print("\n" + "="*60)
print("Final preprocessing completed successfully!")
print("="*60)

# ============================================
# 17. Save Processed Data (Optional)
# ============================================
# X_final.to_csv('../Data/processed/X_preprocessed.csv', index=False)
# y.to_csv('../Data/processed/y_preprocessed.csv', index=False)
# print("\nData saved to ../Data/processed/")
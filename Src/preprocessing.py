# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
# from pathlib import Path
# import joblib
# import os

# def load_data():
#     """Load data"""
#     base_dir = Path(__file__).parent.parent  
#     possible_paths = [
#         base_dir / "Data/raw/trainincome_data.csv",
#         base_dir / "data/raw/trainincome_data.csv",
#         Path("Data/raw/trainincome_data.csv"),
#         Path("../Data/raw/trainincome_data.csv"),
#     ]
    
#     for p in possible_paths:
#         if p.exists():
#             print(f"File found sucessfully: {p}")
#             return pd.read_csv(p)
    
#     raise FileNotFoundError(f"""File not found!Check the path:
#     {base_dir}/Data/raw/trainincome_data.csv
#     """)


# def clean_data(df):
#     """ Data cleaning"""
#     # Handing spaces
#     object_cols = df.select_dtypes(include=['object', 'string']).columns
#     for col in object_cols:
#         df[col] = df[col].str.strip()
    
#     # Replace ? with NaN
#     df.replace('?', np.nan, inplace=True)
#     for col in ['workclass', 'occupation', 'native-country']:
#         if col in df.columns and df[col].isna().any():
#             df[col] = df[col].fillna(df[col].mode()[0])
    
#     # Drop fnlwgt
#     if 'fnlwgt' in df.columns:
#         df.drop('fnlwgt', axis=1, inplace=True)
    
#     # Remove duplicates
#     df = df.drop_duplicates().reset_index(drop=True)
#     print(f" After removing duplicates: {df.shape[0]} rows")
    
#     return df


# def handle_outliers(df):
#     """Handling Outliers"""
#     df['capital-gain'] = np.log1p(df['capital-gain'])
#     df['capital-loss'] = np.log1p(df['capital-loss'])
    
#     for col in ['age', 'hours-per-week']:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         df[col] = df[col].clip(lower, upper)
    
#     return df


# def encode_features(df):
#     """Encoding"""
#     # One-Hot Encoding
#     one_hot_cols = ['sex', 'workclass', 'marital-status', 
#                    'occupation', 'relationship', 'race', 'native-country']
#     df = pd.get_dummies(df, columns=one_hot_cols, dtype=bool)
    
#     # Ordinal Encoding for education
#     education_order = [
#         'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
#         '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc',
#         'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
#     ]
#     encoder = OrdinalEncoder(categories=[education_order])
#     df['education'] = encoder.fit_transform(df[['education']])
    
#     # Label Encoding for Target
#     le = LabelEncoder()
#     df['Income '] = le.fit_transform(df['Income '])
    
#     return df, le


# def scale_features(X):
#     """ Scaling"""
#     numeric_cols = ['age', 'education-num', 
#                    'capital-gain', 'capital-loss', 'hours-per-week']
    
#     scaler = StandardScaler()
#     X_numeric = scaler.fit_transform(X[numeric_cols])
#     X_numeric = pd.DataFrame(X_numeric, columns=numeric_cols, index=X.index)
    
#     X_final = pd.concat([X_numeric, X.drop(columns=numeric_cols)], axis=1)
    
#     return X_final, scaler


# def preprocess_pipeline():
#     """ Pipeline Full"""
#     print(" Start Preprocessing...\n")
    
#     df = load_data()
#     df = clean_data(df)
#     df = handle_outliers(df)
#     df, label_encoder = encode_features(df)
    
#     # Split
#     X = df.drop('Income ', axis=1)
#     y = df['Income ']
    
#     # Scaling
#     X_final, scaler = scale_features(X)
    
#     # ==============================
#     #  Handle conflicting duplicates 
#     # ==============================
#     Xy = pd.concat([X_final, y], axis=1)
    
#     #  majority voting
#     Xy_clean = Xy.groupby(list(X_final.columns))['Income '].agg(lambda x: x.mode()[0]).reset_index()
    
#     X_final = Xy_clean.drop('Income ', axis=1)
#     y = Xy_clean['Income ']
    
#     # ==============================
#     print(f"\n Done! final shape: {X_final.shape}")
#     print(f"No duplicates: {Xy_clean.duplicated().sum()}")
#     print(f"Distribution Target:\n{y.value_counts()}")
    
#     return X_final, y, scaler, label_encoder


# def save_artifacts(X, y, scaler, le, dir_path="Data/processed"):
#     """ Save files"""
#     path = Path(dir_path)
#     path.mkdir(parents=True, exist_ok=True)
    
#     X.to_csv(path / "X_processed.csv", index=False)
#     y.to_csv(path / "y_train.csv", index=False)
#     joblib.dump(scaler, path / "scaler.pkl")
#     joblib.dump(le, path / "label_encoder.pkl")
    
#     print(f"Files saved in : {path}")


# if __name__ == "__main__":
#     X, y, scaler, le = preprocess_pipeline()
#     save_artifacts(X, y, scaler, le)
    
    
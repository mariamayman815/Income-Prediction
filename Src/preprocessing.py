import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

def load_data(path=r'E:\Uni\AI\AI-Income-Project\Data\raw\trainincome_data.csv'):
    """Loading data"""
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Data cleaning"""
    # Useless spaces
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in object_cols:
        df[col] = df[col].str.strip()
    
    # Missing values
    df.replace('?', np.nan, inplace=True)
    for col in ['workclass', 'occupation', 'native-country']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Drop fnlwgt
    if 'fnlwgt' in df.columns:
        df.drop('fnlwgt', axis=1, inplace=True)
    
    # إRemove duplicates
    df = df.drop_duplicates()
    
    return df


def handle_outliers(df):
    """Handling  Outliers"""
    # Log transformation for capital-gain and capital-loss
    df['capital-gain'] = np.log1p(df['capital-gain'])
    df['capital-loss'] = np.log1p(df['capital-loss'])
    
    # Clipping -> age and hours-per-week
    for col in ['age', 'hours-per-week']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    
    return df


def encode_features(df):
    """Encoding variables"""
    # One-Hot Encoding
    one_hot_cols = [
        'sex', 'workclass', 'marital-status', 'occupation',
        'relationship', 'race', 'native-country'
    ]
    df = pd.get_dummies(df, columns=one_hot_cols)
    
    # Ordinal Encoding for education
    education_order = [
        'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
        '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc',
        'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
    ]
    
    encoder = OrdinalEncoder(categories=[education_order])
    df['education'] = encoder.fit_transform(df[['education']])
    
    # Label Encoding for Target
    le = LabelEncoder()
    df['Income '] = le.fit_transform(df['Income '])
    
    return df, le


def scale_features(X, numeric_cols):
    """Scaling numercal variables"""
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_cols])
    X_numeric_scaled = pd.DataFrame(X_numeric_scaled, 
                                  columns=numeric_cols, 
                                  index=X.index)
    
    X_categorical = X.drop(columns=numeric_cols)
    X_final = pd.concat([X_numeric_scaled, X_categorical], axis=1)
    
    return X_final, scaler


def preprocess_pipeline(path=r'E:\Uni\AI\AI-Income-Project\Data\raw\trainincome_data.csv'):
    """Full Pipeline """
    print(" Start Preprocessing...")
    
    # 1. Loading data
    df = load_data(path)
    
    # 2. data cleaning
    df = clean_data(df)
    
    # 3. handling Outliers
    df = handle_outliers(df)
    
    # 4. label encoding
    df, label_encoder = encode_features(df)
    
    # 5. Separate X و y
    X = df.drop('Income ', axis=1)
    y = df['Income ']
    
    # 6. Scaling
    numeric_cols = ['age', 'education', 'education-num', 
                   'capital-gain', 'capital-loss', 'hours-per-week']
    
    X_final, scaler = scale_features(X, numeric_cols)
    
    print(f" Done sucessfully!final data: {X_final.shape}")
    print(f"Numbers of features: {X_final.shape[1]}")
    
    return X_final, y, scaler, label_encoder


if __name__ == "__main__":
    X, y, scaler, le = preprocess_pipeline()
   


if __name__ == "__main__":
    X, y, scaler, le = preprocess_pipeline()
    
    print("\n" + "="*50)
    print("🔍 التحقق من جودة الـ Preprocessing:")
    print("="*50)
    
    # 1. التحقق من الـ Scaling
    numeric_cols = ['age', 'education', 'education-num', 
                   'capital-gain', 'capital-loss', 'hours-per-week']
    
    print("\nمتوسط الأعمدة الرقمية (يجب يكون قريب جدًا من 0):")
    print(X[numeric_cols].mean().round(4))
    
    print("\nالانحراف المعياري (يجب يكون قريب من 1):")
    print(X[numeric_cols].std().round(4))
    
    # 2. التحققات الأخرى
    print(f"\nعدد الـ NaN: {X.isnull().sum().sum()}")
    print(f"عدد التكرارات: {X.duplicated().sum()}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # 3. حفظ البيانات المعالجة (مهم)
    import joblib
    from pathlib import Path
    
    processed_dir = Path("Data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    X.to_csv(processed_dir / "X_train.csv", index=False)
    y.to_csv(processed_dir / "y_train.csv", index=False)
    
    joblib.dump(scaler, processed_dir / "scaler.pkl")
    joblib.dump(le, processed_dir / "label_encoder.pkl")
    
    print(f"\n✅ تم حفظ البيانات المعالجة في: {processed_dir}")
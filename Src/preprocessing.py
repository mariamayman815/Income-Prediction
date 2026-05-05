import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_clean_data(train_path, test_path):
    df = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    datasets = [df, test]
    for data in datasets:
        object_cols = data.select_dtypes(include=['object', 'string']).columns
        for col in object_cols: # removing initial space 
            data[col] = data[col].str.strip()

        data.replace('?', np.nan, inplace=True)
        for col in ['workclass', 'occupation', 'native-country']:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        if 'Income ' in data.columns:
            data['Income '] = data['Income '].str.replace('.', '', regex=False) # removing dots 
        
    df = df.drop_duplicates().reset_index(drop=True)
    test = test.drop_duplicates().reset_index(drop=True)

    for data in [df, test]:
        if 'fnlwgt' in data.columns:
            data.drop('fnlwgt', axis=1, inplace=True)
    return df, test

def handle_outliers_and_features(df, test):
    
    # 1. Log Transformation
    for data in [df, test]:
        data['capital-gain'] = np.log1p(data['capital-gain'])
        data['capital-loss'] = np.log1p(data['capital-loss'])
    
    # 2. Outlier Clipping 
    for col in ['age', 'hours-per-week']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)
        
    return df, test

def encode_and_scale(df, test):
    
    # 1. Ordinal Encoding 
    education_order = [
        'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', 
        '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 
        'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
    ]
    ord_enc = OrdinalEncoder(categories=[education_order])
    df['education'] = ord_enc.fit_transform(df[['education']])
    test['education'] = ord_enc.transform(test[['education']])
    
    # 2. One-Hot Encoding
    one_hot_cols = ['sex', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Train
    train_ohe = pd.DataFrame(ohe.fit_transform(df[one_hot_cols]), 
                columns=ohe.get_feature_names_out(one_hot_cols), index=df.index)
    df = pd.concat([df.drop(columns=one_hot_cols), train_ohe], axis=1)
    
    # Test
    test_ohe = pd.DataFrame(ohe.transform(test[one_hot_cols]), 
                            columns=ohe.get_feature_names_out(one_hot_cols), index=test.index)
    test = pd.concat([test.drop(columns=one_hot_cols), test_ohe], axis=1)
    
    # 3. Target Mapping
    df['Income '] = df['Income '].map({'<=50K': 0, '>50K': 1})
    test['Income '] = test['Income '].map({'<=50K': 0, '>50K': 1})
    
    return df, test

def prepare_final_sets(df, test):
    
    x_train = df.drop('Income ', axis=1)
    y_train = df['Income ']
    x_test = test.drop('Income ', axis=1)
    y_test = test['Income ']
    
    # Split Validation
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_final)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    
    x_train_df = pd.DataFrame(x_train_scaled, columns=x_train_final.columns)
    x_val_df = pd.DataFrame(x_val_scaled, columns=x_val.columns)
    x_test_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    
    return x_train_df, x_val_df, x_test_df, y_train_final, y_val, y_test

if __name__ == "__main__":
    df_raw, test_raw = load_and_clean_data('../Data/raw/trainincome_data.csv', '../Data/raw/test_data.csv')
    df_feat, test_feat = handle_outliers_and_features(df_raw, test_raw)
    df_final, test_final = encode_and_scale(df_feat, test_feat)
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_final_sets(df_final, test_final)

    
# df = pd.read_csv('../Data/raw/trainincome_data.csv')
# test = pd.read_csv('../Data/raw/test_data.csv')
# # Step 1: Handle missing values

# for col in df.columns:
#     print("Column:", col)
#     print(df[col].unique())
#     print("------")

# for col in test.columns:
#     print("Column:", col)
#     print(test[col].unique())
#     print("------")

# object_cols = df.select_dtypes(include=['object', 'string']).columns
# for col in object_cols:
#     df[col] = df[col].str.strip()

# object_cols = test.select_dtypes(include=['object', 'string']).columns
# for col in object_cols:
#     test[col] = test[col].str.strip()


# df = df.drop_duplicates().reset_index(drop=True)
# test = test.drop_duplicates().reset_index(drop=True)

# df.duplicated().sum()
# test.duplicated().sum()


# for i in df.select_dtypes(include="number").columns:
#     sns.histplot(data=df,x=i)
#     plt.show()

# for i in df.select_dtypes(include="number").columns:
#     sns.boxplot(data=df,x=i)
#     plt.show() 

# df.plot.scatter(x='age', y='hours-per-week')

# df.plot.scatter(x='age', y='education-num')

# df.plot.scatter(x='hours-per-week', y='education-num')


# # there is no correlation between the features
# df.select_dtypes(include="number").corr()

# sns.heatmap(df.select_dtypes(include="number").corr(), annot=True)

# plt.figure(figsize=(20, 6))
# sns.countplot(x='education', hue='Income ', data=df)

# # handling garbage values
# df.replace('?', np.nan, inplace=True)
# for col in ['workclass', 'occupation', 'native-country']:
#     df[col] = df[col].fillna(df[col].mode()[0])


# test.replace('?', np.nan, inplace=True)
# for col in ['workclass', 'occupation', 'native-country']:
#     test[col] = test[col].fillna(test[col].mode()[0])

# (df == "unknown").sum()
# (test == "unknown").sum()


# num_cols=df.select_dtypes(include='int64').columns
# print(num_cols)

# def outliers_iqr(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
    
#     outliers = df[(df[column] < lower) | (df[column] > upper)]
#     return print(column,":",outliers.shape[0])
      

# for col in num_cols :
#     outliers_iqr(df,col)
    

# df['capital-gain']=np.log1p(df['capital-gain'])
# df['capital-loss']=np.log1p(df['capital-loss'])

# test['capital-gain']=np.log1p(test['capital-gain'])
# test['capital-loss']=np.log1p(test['capital-loss'])


# sns.boxplot(data=df,x=df['capital-gain'])

# sns.boxplot(data=df,x=df['capital-loss'])

# Q1 = df['age'].quantile(0.25)
# Q3 = df['age'].quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# df['age'] = df['age'].clip(lower, upper)

# Q1 = df['age'].quantile(0.25)
# Q3 = df['age'].quantile(0.75)
# IQR = Q3 - Q1 # using training data to avoid data leakage

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# test['age'] = test['age'].clip(lower, upper)


# Q1 = df['hours-per-week'].quantile(0.25)
# Q3 = df['hours-per-week'].quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# df['hours-per-week'] = df['hours-per-week'].clip(lower, upper)


# Q1 = df['hours-per-week'].quantile(0.25)
# Q3 = df['hours-per-week'].quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# test['hours-per-week'] = test['hours-per-week'].clip(lower, upper)

# print("Max capital-gain in train:", df['capital-gain'].max())
# print("Max capital-gain in test:", test['capital-gain'].max())


# sns.boxplot(data=df,x=df['age'])


# sns.boxplot(data=df,x=df['hours-per-week'])


# df.duplicated().sum()
# test.duplicated().sum() # there is one after outlier handling cause values became closer to each other

# if 'fnlwgt' in df.columns:
#     df.drop('fnlwgt', axis=1, inplace=True)
#     print("Column 'fnlwgt' has been dropped.")
# else:
#     print("Column 'fnlwgt' not found - it was already dropped.")

# test.drop('fnlwgt', axis=1, inplace=True)

# test.duplicated().sum()
# df.columns.tolist()


# df=df.drop_duplicates()
# test = test.drop_duplicates()

# one_hot_cols = [
#     'sex',
#     'workclass',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'native-country'
# ]
# ordinal_cols=['education']

# encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# train_encoded_parts = encoder.fit_transform(df[one_hot_cols])

# encoded_cols_names = encoder.get_feature_names_out(one_hot_cols)
# train_encoded_df = pd.DataFrame(train_encoded_parts, columns=encoded_cols_names, index=df.index)

# df = pd.concat([df, train_encoded_df], axis=1)

# df.drop(columns=one_hot_cols, inplace=True)

# test_encoded_parts = encoder.transform(test[one_hot_cols]) 

# test_encoded_df = pd.DataFrame(test_encoded_parts, columns=encoded_cols_names, index=test.index)

# test = pd.concat([test, test_encoded_df], axis=1)

# test.drop(columns=one_hot_cols, inplace=True)

# education_order = [
#     'Preschool',
#     '1st-4th',
#     '5th-6th',
#     '7th-8th',
#     '9th',
#     '10th',
#     '11th',
#     '12th',
#     'HS-grad',
#     'Some-college',
#     'Assoc-voc',
#     'Assoc-acdm',
#     'Bachelors',
#     'Masters',
#     'Prof-school',
#     'Doctorate'
# ]
# encoder=OrdinalEncoder(categories=[education_order])

# df['education']=encoder.fit_transform(df[['education']])
# test['education'] = encoder.transform(test[['education']])



# df['Income '] = df['Income '].str.strip().str.replace('.', '', regex=False)
# test['Income '] = test['Income '].str.replace('.', '', regex=False)

# x_train = df.drop('Income ', axis=1)
# y_train = df['Income ']

# y_train = y_train.map({'<=50K': 0, '>50K': 1})


# x_test = test.drop('Income ', axis=1)
# y_test = test['Income ']
# y_test = y_test.map({'<=50K': 0, '>50K': 1})

# x_train_final, x_val, y_train_final, y_val = train_test_split(
#     x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
# )


# scaler = StandardScaler()

# x_train_scaled = scaler.fit_transform(x_train_final)

# x_val_scaled = scaler.transform(x_val)
# x_test_scaled = scaler.transform(x_test)

# x_train_final_df = pd.DataFrame(x_train_scaled, columns=x_train_final.columns, index=x_train_final.index)

# x_val_final_df = pd.DataFrame(x_val_scaled, columns=x_val.columns, index=x_val.index)

# x_test_final_df = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

# print("Missing values in Train:", x_train_final_df.isnull().sum().sum())
# print("----------------------------------------------------------")
# print("Missing values in Val:", x_val_final_df.isnull().sum().sum())
# print("----------------------------------------------------------")
# print("Missing values in Test:", x_test_final_df.isnull().sum().sum())


# print("Data type of Train scaled:", x_train_scaled.dtype)
# print("----------------------------------------------------------")
# print("Data type of Train scaled:", x_val_scaled.dtype)
# print("----------------------------------------------------------")
# print("Data type of Train scaled:", x_test_scaled.dtype)


# print("Mean of X_train_scaled:", np.mean(x_train_scaled).round(2))
# print("Std of X_train_scaled:", np.std(x_train_scaled).round(2))
# print("----------------------------------------------------------")
# print("Mean of X_train_scaled:", np.mean(x_val_scaled).round(2))
# print("Std of X_train_scaled:", np.std(x_val_scaled).round(2))
# print("----------------------------------------------------------")
# print("Mean of X_train_scaled:", np.mean(x_test_scaled).round(2))
# print("Std of X_train_scaled:", np.std(x_test_scaled).round(2))


# print(f"X_train_scaled columns: {x_train_scaled.shape[1]}")
# print("----------------------------------------------------------")
# print(f"X_val_scaled columns: {x_val_scaled.shape[1]}")
# print("----------------------------------------------------------")
# print(f"X_test_scaled columns: {x_test_scaled.shape[1]}")


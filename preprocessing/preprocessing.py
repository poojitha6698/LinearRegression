import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(file_path):

    # Load Dataset
    df = pd.read_csv(file_path)

    # Check Missing Values
    print(df.isnull().sum())

    # Encoding Categorical Features
    le = LabelEncoder()

    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])

    return df
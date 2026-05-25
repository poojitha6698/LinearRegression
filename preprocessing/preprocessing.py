import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(path):

    df = pd.read_csv(path)

    le = LabelEncoder()

    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])

    return df
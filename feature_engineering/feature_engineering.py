def feature_engineering(df):

    X = df.drop('charges', axis=1)
    y = df['charges']

    return X, y
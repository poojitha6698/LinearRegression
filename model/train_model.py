import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_model(X, y):

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Linear Regression Model
    model = LinearRegression()

    # Train Model
    model.fit(X_train, y_train)

    # Save Model
    with open('model/linear_regression_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model, X_test, y_test
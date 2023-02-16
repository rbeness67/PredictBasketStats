import pandas as pd
from joblib import load


def predict(data):
    # Load model
    model = load("models/basketball_model.joblib")

    # Make prediction
    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data/test_data.csv")

    # Make prediction
    result = predict(data)

    print(result)

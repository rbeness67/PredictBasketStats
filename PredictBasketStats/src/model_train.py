import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from feature_eng import engineer_features

def train_model():
    print('enter')
    # Load processed data
    data = pd.read_csv("data/processed_data.csv")


    # Engineer features
    data = engineer_features(data)
    print('passed')
    # Split data into features and target
    X = data.drop(["HOME_TEAM_WINS"], axis=1)
    y = data["HOME_TEAM_WINS"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Save trained model to file
    joblib.dump(model, "models/model.pkl")


if __name__ == "__main__":
    train_model()

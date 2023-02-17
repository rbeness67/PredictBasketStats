import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import os

# Step 1: Start an MLflow run
mlflow.start_run()

# Step 2: Load data
data = pd.read_csv("data/processed_data.csv")
train, test = train_test_split(data)

# Step 3: Set the model parameters
n_estimators = 100
max_depth = 6
random_state = 42

# Step 4: Train the model
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
rf.fit(train[["feature1", "feature2", "feature3"]], train["target"])

# Step 5: Predict on the test set
predictions = rf.predict(test[["feature1", "feature2", "feature3"]])

# Step 6: Log model parameters and metrics
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_param("random_state", random_state)

mse = mean_squared_error(test["target"], predictions)
mae = mean_absolute_error(test["target"], predictions)
r2 = r2_score(test["target"], predictions)

mlflow.log_metric("mse", mse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)

# Step 7: Save the model to the MLflow server
mlflow.sklearn.log_model(rf, "random-forest-model")

# Step 8: End the MLflow run
mlflow.end_run()

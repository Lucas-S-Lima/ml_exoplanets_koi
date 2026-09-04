import sys
from pathlib import Path
import pickle
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from ml_core.facade import load_exoplanet_split, get_pickle_filepath, ROOT_DIR


get_pickle_filepath("exoplanets_split.pkl", root_dir=ROOT_DIR) 
x_koi_train, x_koi_test, y_koi_train, y_koi_test = load_exoplanet_split("exoplanets_split.pkl", ROOT_DIR)


logistic_regression = LogisticRegression(
    l1_ratio=0.0,
    C=3000,
    solver="lbfgs",
    max_iter=2000,
    random_state=42,
)

logistic_regression.fit(x_koi_train, y_koi_train)   

print("Logistic Regression model trained successfully.")
print("Intercept:", logistic_regression.intercept_)
print("Coefficients:", logistic_regression.coef_)

predictions = logistic_regression.predict(x_koi_test)
print("Predictions:", predictions)
print("Test accuracy:", logistic_regression.score(x_koi_test, y_koi_test))

print("Metrics:")
print("Accuracy:", accuracy_score(y_koi_test, predictions))
print("Classification Report:\n", classification_report(y_koi_test, predictions))

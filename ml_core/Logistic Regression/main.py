import pickle
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression


ROOT_DIR = Path(__file__).resolve().parents[2]

pkl_path = ROOT_DIR / "exoplanets.pkl"

data = joblib.load(pkl_path)

x_koi_train = data["x_train"]
x_koi_test = data["x_test"]

y_koi_train = data["y_train"]
y_koi_test = data["y_test"]


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

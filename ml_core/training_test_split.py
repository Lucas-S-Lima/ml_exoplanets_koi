import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.x_koi_scaling import x_known_scaled as x_koi_data
from ml_core.y_koi_encoded import y_known_encoded as y_koi_data


x_koi_train, x_koi_test, y_koi_train, y_koi_test = train_test_split(x_koi_data, y_koi_data, test_size=0.25, random_state=0, stratify=y_koi_data)











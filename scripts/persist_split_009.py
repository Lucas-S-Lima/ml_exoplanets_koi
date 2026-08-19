import joblib
from training_test_split_008 import x_koi_train, x_koi_test, y_koi_train, y_koi_test

data = {
    'x_train': x_koi_train,
    'x_test': x_koi_test,
    'y_train': y_koi_train,
    'y_test': y_koi_test,
}

joblib.dump(data, 'exoplanets_split.pkl')
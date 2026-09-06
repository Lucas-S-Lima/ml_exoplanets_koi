from pathlib import Path
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DEFAULT_FILENAME = "exoplanets_split.pkl"
ROOT_DIR = Path(__file__).resolve().parents[1]


def get_pickle_filepath(filename: str = DEFAULT_FILENAME, root_dir: Path | None = None) -> Path:

    root_dir = root_dir or ROOT_DIR

    return root_dir / filename


def load_exoplanet_split(filename: str = DEFAULT_FILENAME, root_dir: Path | None = None):

    pickle_path = get_pickle_filepath(filename, root_dir)
    raw = joblib.load(pickle_path)

    return raw["x_train"], raw["x_test"], raw["y_train"], raw["y_test"]


def logistic_regression_model(
    x_train, y_train, l1_ratio=0.0, C=3000, solver="lbfgs", max_iter=2000, random_state=42
):

    model = LogisticRegression(
        l1_ratio=l1_ratio,
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(x_train, y_train)
    return model


def logistic_regression_predict(model, x_test):
    return model.predict(x_test)


def get_logistic_regression_metrics(model, x_test, y_test) -> dict:
    predictions = logistic_regression_predict(model, x_test)

    return {
        "intercept": model.intercept_,
        "coefficients": model.coef_,
        "predictions": predictions,
        "test_score": model.score(x_test, y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }


def get_logistic_regression_confusion_matrix(model, x_test, y_test):
    predictions = logistic_regression_predict(model, x_test)
    return confusion_matrix(y_test, predictions)


def random_forest_model(
    x_train,
    y_train,
    n_estimators=200,
    criterion="entropy",
    class_weight="balanced_subsample",
    max_depth=30,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(x_train, y_train)
    return model


def random_forest_predict(model, x_test):
    return model.predict(x_test)


def get_random_forest_metrics(model, x_test, y_test) -> dict:
    predictions = random_forest_predict(model, x_test)

    return {
        "intercept": model.intercept_,
        "coefficients": model.coef_,
        "predictions": predictions,
        "test_score": model.score(x_test, y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }


def get_random_forest_confusion_matrix(model, x_test, y_test):
    predictions = random_forest_predict(model, x_test)
    return confusion_matrix(y_test, predictions)


def svm_model(
    x_train, y_train, kernel="rbf", C=400, gamma=0.05, class_weight=None, random_state=42
):

    model = SVC(
        kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, random_state=random_state
    )

    model.fit(x_train, y_train)
    return model


def svm_predict(model, x_test):
    return model.predict(x_test)


def get_svm_metrics(model, x_test, y_test) -> dict:
    predictions = svm_predict(model, x_test)

    return {
        "predictions": predictions,
        "test_score": model.score(x_test, y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }


def get_svm_confusion_matrix(model, x_test, y_test):
    predictions = svm_predict(model, x_test)
    return confusion_matrix(y_test, predictions)

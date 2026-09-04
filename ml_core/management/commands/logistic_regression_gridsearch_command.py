from sklearn.linear_model import LogisticRegression
from _base_gridsearch import BaseGridSearchCommand


class Command(BaseGridSearchCommand):
    help = "Busca os melhores hiperparâmetros para a Regressão Logística."
    algorithm_name = "Regressão Logística"
    default_expected_combinations = 42
    default_pkl_filename = "exoplanets_split.pkl"

    C_RANGE_AMPLO = [0.01, 0.1, 1, 10, 100, 300, 1000, 3000]
    C_RANGE_L1 = [0.01, 0.1, 1, 10, 100]

    def get_estimator(self):
        return LogisticRegression(random_state=0)

    def get_param_grid(self):
        return [
            {
                "l1_ratio": [0.0],
                "C": self.C_RANGE_AMPLO,
                "solver": ["lbfgs"],
                "class_weight": [None, "balanced"],
                "max_iter": [2000],
            },
            {
                "l1_ratio": [0.0],
                "C": self.C_RANGE_AMPLO,
                "solver": ["liblinear"],
                "class_weight": [None, "balanced"],
                "max_iter": [2000],
            },
            {
                "l1_ratio": [1.0],
                "C": self.C_RANGE_L1,
                "solver": ["liblinear"],
                "class_weight": [None, "balanced"],
                "max_iter": [2000],
            },
        ]

from sklearn.ensemble import RandomForestClassifier
from _base_gridsearch import BaseGridSearchCommand


class Command(BaseGridSearchCommand):
    help = "Busca os melhores hiperparâmetros para o Random Forest."
    algorithm_name = "Random Forest"
    default_expected_combinations = 216

    def get_estimator(self):
        return RandomForestClassifier(
            criterion="entropy",
            random_state=0,
            n_jobs=1,
        )

    def get_param_grid(self):
        return {
            "n_estimators": [150, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }
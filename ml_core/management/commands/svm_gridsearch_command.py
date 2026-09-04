import logging
import pandas as pd
from sklearn.svm import SVC
from _base_gridsearch import BaseGridSearchCommand

logger = logging.getLogger(__name__)


MAX_ITER_SEGURANCA = 50_000

C_RANGE_LINEAR = [0.1, 1, 10, 100, 300, 1000]
C_RANGE_RBF = [50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]
GAMMA_RANGE_RBF = ["scale", 0.03, 0.05, 0.07, 0.1, 0.2]
C_RANGE_POLY = [1, 10, 100, 300]
GAMMA_RANGE_POLY = ["scale", 0.1]
C_RANGE_SIGMOID = [1, 10, 100]
GAMMA_RANGE_SIGMOID = ["scale", 0.05, 0.1]


class Command(BaseGridSearchCommand):
    help = "Busca os melhores hiperparâmetros para o SVM"
    algorithm_name = "SVM"
    default_cv_splits = 3
    default_n_jobs = 10
    evaluate_test_set = True

    def get_estimator(self):
        return SVC(random_state=0)

    def get_param_grid(self):
        return [
            {
                "kernel": ["linear"],
                "C": C_RANGE_LINEAR,
                "class_weight": [None],
                "max_iter": [MAX_ITER_SEGURANCA],
            },
            {
                "kernel": ["rbf"],
                "C": C_RANGE_RBF,
                "gamma": GAMMA_RANGE_RBF,
                "class_weight": [None],
            },
            {
                "kernel": ["poly"],
                "C": C_RANGE_POLY,
                "gamma": GAMMA_RANGE_POLY,
                "degree": [2, 3],
                "coef0": [0, 1],
                "class_weight": [None],
            },
            {
                "kernel": ["sigmoid"],
                "C": C_RANGE_SIGMOID,
                "gamma": GAMMA_RANGE_SIGMOID,
                "coef0": [0, 1],
                "class_weight": [None],
            },
        ]

    def _extra_analysis(self, grid_search):
        df_resultados = pd.DataFrame(grid_search.cv_results_)

        colunas_kernel = [
            "rank_test_f1_macro",
            "mean_test_f1_macro",
            "std_test_f1_macro",
            "mean_test_accuracy",
            "mean_test_precision_macro",
            "mean_test_recall_macro",
            "mean_fit_time",
            "param_kernel",
            "param_C",
            "param_gamma",
            "param_degree",
            "param_coef0",
        ]

        melhor_por_kernel = (
            df_resultados.sort_values("rank_test_f1_macro")
            .groupby("param_kernel", as_index=False)
            .first()[colunas_kernel]
            .sort_values("mean_test_f1_macro", ascending=False)
        )

        logger.info(
            "Melhor configuração de cada kernel:\n%s",
            melhor_por_kernel.to_string(index=False),
        )

        comparacao_kernel = (
            df_resultados.groupby("param_kernel")["mean_test_f1_macro"]
            .agg(media="mean", maximo="max", desvio="std")
            .sort_values("maximo", ascending=False)
        )

        logger.info("Análise por kernel:\n%s", comparacao_kernel.to_string())

        comparacao_c = df_resultados.groupby(["param_kernel", "param_C"])["mean_test_f1_macro"].agg(
            media="mean", maximo="max", desvio="std"
        )

        logger.info("Análise de C (por kernel):\n%s", comparacao_c.to_string())

        mask_com_gamma = df_resultados["param_kernel"] != "linear"
        comparacao_gamma = (
            df_resultados[mask_com_gamma]
            .groupby(["param_kernel", "param_gamma"])["mean_test_f1_macro"]
            .agg(media="mean", maximo="max", desvio="std")
        )

        logger.info(
            "Análise de gamma (kernels rbf/poly/sigmoid):\n%s",
            comparacao_gamma.to_string(),
        )

        best_svm = grid_search.best_estimator_
        if hasattr(best_svm, "n_support_"):
            logger.info("Vetores de suporte do melhor modelo: %d", best_svm.n_support_.sum())

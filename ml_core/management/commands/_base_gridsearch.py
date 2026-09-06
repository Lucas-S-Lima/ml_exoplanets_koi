import logging
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from ml_core.utils import load_exoplanet_split

logger = logging.getLogger(__name__)


class BaseGridSearchCommand(BaseCommand):
    """Comando base para busca de hiperparâmetros via GridSearchCV.

    Subclasses devem sobrescrever:
      - algorithm_name: nome legível do algoritmo (usado nos logs)
      - get_estimator(): retorna o estimador (sem fit)
      - get_param_grid(): retorna o param_grid (dict ou list de dicts)

    Opcionalmente:
      - default_expected_combinations: nº de combinações esperadas,
        usado como validação extra (assert). Se None, a validação é
        pulada e o valor observado é usado como referência.
      - default_pkl_filename: nome do arquivo .pkl.
    """

    help = "Executa GridSearchCV para encontrar os melhores hiperparâmetros."
    algorithm_name: str = "Modelo"
    default_expected_combinations: int | None = None
    default_pkl_filename: str = "exoplanets_split.pkl"

    def add_arguments(self, parser):
        parser.add_argument(
            "--filename",
            type=str,
            default=self.default_pkl_filename,
            help="Nome do arquivo .pkl com os dados de treino/teste.",
        )
        parser.add_argument(
            "--root-dir",
            type=str,
            default=None,
            help="Diretório raiz onde o .pkl está localizado (opcional).",
        )
        parser.add_argument(
            "--cv-splits",
            type=int,
            default=5,
            help="Número de folds da validação cruzada estratificada.",
        )
        parser.add_argument(
            "--n-jobs",
            type=int,
            default=-1,
            help="Número de jobs paralelos no GridSearchCV.",
        )
        parser.add_argument(
            "--verbose-level",
            type=int,
            default=2,
            help="Nível de verbosidade do GridSearchCV (0-3).",
        )

    def get_estimator(self):
        raise NotImplementedError

    def get_param_grid(self):
        raise NotImplementedError

    def handle(self, *args, **options):
        self.options = options

        x_train, x_test, y_train, y_test = self._load_data()

        param_grid = self.get_param_grid()
        combinacoes = list(ParameterGrid(param_grid))

        n_combinacoes_esperadas = (
            self.default_expected_combinations
            if self.default_expected_combinations is not None
            else len(combinacoes)
        )
        self._validar_param_grid(combinacoes, n_combinacoes_esperadas)

        cv = StratifiedKFold(
            n_splits=options["cv_splits"],
            shuffle=True,
            random_state=0,
        )

        grid_search = GridSearchCV(
            estimator=self.get_estimator(),
            param_grid=param_grid,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            refit="f1_macro",
            cv=cv,
            n_jobs=options["n_jobs"],
            verbose=options["verbose_level"],
            error_score="raise",
        )

        logger.info("Iniciando Grid Search - %s", self.algorithm_name)

        inicio = time.perf_counter()
        grid_search.fit(x_train, y_train)
        fim = time.perf_counter()

        self._checar_execucao(grid_search, combinacoes, n_combinacoes_esperadas)
        self._relatorio_execucao(grid_search, fim - inicio, cv.n_splits)
        self._top_configuracoes(grid_search)
        self._modelo_final(grid_search)

        return f"Melhor f1_macro (CV): {grid_search.best_score_:.4f}"

    def _load_data(self):
        filename = self.options["filename"]
        root_dir = Path(self.options["root_dir"]) if self.options["root_dir"] else None

        logger.info("Carregando dados de: %s", filename)

        try:
            x_train, x_test, y_train, y_test = load_exoplanet_split(filename, root_dir)

        except FileNotFoundError as exc:
            logger.error("Arquivo de dados não encontrado: %s", exc)
            raise CommandError(str(exc)) from exc

        logger.info("x_train: %s | x_test: %s", x_train.shape, x_test.shape)
        return x_train, x_test, y_train, y_test

    def _validar_param_grid(self, combinacoes, n_esperadas):
        n_combinacoes = len(combinacoes)

        logger.info("Combinações no param_grid: %d", n_combinacoes)

        if n_combinacoes != n_esperadas:
            mensagem = (
                f"Esperadas {n_esperadas} combinações, mas foram encontradas {n_combinacoes}."
            )
            logger.error(mensagem)
            raise CommandError(mensagem)

        logger.info("As %d combinações foram confirmadas.", n_esperadas)

    def _checar_execucao(self, grid_search, combinacoes, n_esperadas):
        resultados = grid_search.cv_results_
        params_executados = resultados["params"]
        n_executadas = len(params_executados)

        esperadas_set = {tuple(sorted(p.items())) for p in combinacoes}
        executadas_set = {tuple(sorted(p.items())) for p in params_executados}

        faltantes = esperadas_set - executadas_set
        extras = executadas_set - esperadas_set

        logger.info(
            "Combinações executadas: %d (faltantes: %d, extras: %d)",
            n_executadas,
            len(faltantes),
            len(extras),
        )

        if n_executadas != n_esperadas:
            mensagem = (
                f"Esperadas {n_esperadas}, executadas {n_executadas}. "
                f"Faltantes: {len(faltantes)} | Extras: {len(extras)}"
            )
            logger.error(mensagem)
            raise CommandError(mensagem)

    def _relatorio_execucao(self, grid_search, tempo_total, n_folds):
        resultados = grid_search.cv_results_
        n_combinacoes = len(resultados["params"])

        mean_fit_times = np.asarray(resultados["mean_fit_time"])
        mean_score_times = np.asarray(resultados["mean_score_time"])

        n_cpus = os.cpu_count()
        memoria_maxima = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memoria_maxima_mb = (
            memoria_maxima / (1024**2) if platform.system() == "Darwin" else memoria_maxima / 1024
        )

        logger.info("Métricas de execução:")
        logger.info(
            "Tempo total          : %.2f s (%.2f min)",
            tempo_total,
            tempo_total / 60,
        )
        logger.info("Tempo médio de fit   : %.4f s", np.mean(mean_fit_times))
        logger.info("Tempo médio de score : %.4f s", np.mean(mean_score_times))
        logger.info("CPUs disponíveis     : %s", n_cpus)
        logger.info("Memória máxima       : %.2f MB", memoria_maxima_mb)
        logger.info("Combinações x folds  : %d x %d", n_combinacoes, n_folds)

    def _top_configuracoes(self, grid_search, top_n: int = 10):
        df_resultados = pd.DataFrame(grid_search.cv_results_)

        colunas_param = [c for c in df_resultados.columns if c.startswith("param_")]
        colunas_top = [
            "rank_test_f1_macro",
            "mean_test_f1_macro",
            "std_test_f1_macro",
            "mean_test_accuracy",
            "mean_test_precision_macro",
            "mean_test_recall_macro",
            "mean_fit_time",
            *colunas_param,
        ]

        top = df_resultados[colunas_top].sort_values("rank_test_f1_macro").head(top_n)

        logger.info("Top %d configurações (f1_macro):\n%s", top_n, top.to_string(index=False))

    def _modelo_final(self, grid_search):
        logger.info("Modelo final:")

        for parametro, valor in grid_search.best_params_.items():
            logger.info("%-20s: %s", parametro, valor)

        logger.info("F1 macro (CV): %.4f", grid_search.best_score_)

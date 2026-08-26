import os
import time
import resource
import platform
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    ParameterGrid,
)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

pkl_path = ROOT_DIR / "exoplanets.pkl"

print("\n" + "=" * 60)
print("CARREGAMENTO DOS DADOS")
print("=" * 60)

print(f"ROOT_DIR : {ROOT_DIR}")
print(f"PKL      : {pkl_path}")

if not pkl_path.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{pkl_path}"
    )

data = joblib.load(pkl_path)

x_koi_train = data["x_train"]
x_koi_test = data["x_test"]
y_koi_train = data["y_train"]
y_koi_test = data["y_test"]

print("\nDados carregados com sucesso.")

print(
    f"x_train: {x_koi_train.shape}"
)

print(
    f"y_train: {y_koi_train.shape}"
)

print(
    f"x_test : {x_koi_test.shape}"
)

print(
    f"y_test : {y_koi_test.shape}"
)

print("=" * 60)


# ============================================================
# AMBIENTE
# ============================================================

print("\n" + "=" * 60)
print("AMBIENTE DE EXECUÇÃO")
print("=" * 60)

print(
    f"Arquivo executado : {__file__}"
)

print(
    f"Python/scikit-learn: "
    f"{sklearn.__version__}"
)

print(
    f"GridSearchCV      : "
    f"{GridSearchCV}"
)

print(
    f"Implementação     : "
    f"{inspect.getfile(GridSearchCV)}"
)

print("=" * 60)


# ============================================================
# CONFIGURAÇÃO DO GRID
# ============================================================
# A partir do scikit-learn 1.8, o parâmetro 'penalty' foi descontinuado
# (removido na 1.10) em favor de 'l1_ratio':
#   l1_ratio=0 -> equivalente a penalty='l2'
#   l1_ratio=1 -> equivalente a penalty='l1'
#   0 < l1_ratio < 1 -> equivalente a penalty='elasticnet'
#   C=np.inf   -> equivalente a penalty=None
#
# O grid é dividido em três blocos por compatibilidade solver/penalidade
# e por custo computacional:
#   1) solver 'lbfgs'     + L2 (l1_ratio=0) -> rápido, range de C ampliado
#   2) solver 'liblinear'  + L2 (l1_ratio=0) -> rápido, range de C ampliado
#   3) solver 'liblinear'  + L1 (l1_ratio=1) -> lento (visto no log: ~6min/fit
#      com C=10 e C=100), mantido no range original para não explodir o tempo

C_RANGE_AMPLO = [0.01, 0.1, 1, 10, 100, 300, 1000, 3000]
C_RANGE_L1 = [0.01, 0.1, 1, 10, 100]

param_grid = [
    {
        "l1_ratio": [0.0],
        "C": C_RANGE_AMPLO,
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"],
        "max_iter": [2000],
    },
    {
        "l1_ratio": [0.0],
        "C": C_RANGE_AMPLO,
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
        "max_iter": [2000],
    },
    {
        "l1_ratio": [1.0],
        "C": C_RANGE_L1,
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
        "max_iter": [2000],
    },
]


# ============================================================
# VALIDAÇÃO DAS COMBINAÇÕES
# ============================================================

combinacoes = list(
    ParameterGrid(param_grid)
)

n_combinacoes_esperadas = len(
    combinacoes
)

print("\n" + "=" * 60)
print("VALIDAÇÃO DO PARAM_GRID")
print("=" * 60)

print(
    "Bloco 1 (lbfgs, L2)          :",
    len(list(ParameterGrid(param_grid[0])))
)

print(
    "Bloco 2 (liblinear, L2)      :",
    len(list(ParameterGrid(param_grid[1])))
)

print(
    "Bloco 3 (liblinear, L1)      :",
    len(list(ParameterGrid(param_grid[2])))
)

print(
    f"\nCombinações esperadas: "
    f"{n_combinacoes_esperadas}"
)

assert n_combinacoes_esperadas == 42, (
    f"ERRO: deveriam existir 42 combinações, "
    f"mas foram encontradas "
    f"{n_combinacoes_esperadas}."
)

print(
    "✓ As 42 combinações foram confirmadas."
)

print("=" * 60)


# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)


# ============================================================
# MÉTRICAS
# ============================================================

scoring = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]


# ============================================================
# ESTIMADOR
# ============================================================

estimator = LogisticRegression(
    random_state=0,
)


# ============================================================
# GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    scoring=scoring,
    refit="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    error_score="raise",
)


# ============================================================
# VERIFICAÇÃO ANTES DO FIT
# ============================================================

grid_antes_do_fit = list(
    ParameterGrid(
        grid_search.param_grid
    )
)

print("\n" + "=" * 60)
print("VERIFICAÇÃO ANTES DO FIT")
print("=" * 60)

print(
    f"Combinações no GridSearchCV: "
    f"{len(grid_antes_do_fit)}"
)

assert len(grid_antes_do_fit) == 42, (
    "ERRO: o GridSearchCV não recebeu "
    "as 42 combinações esperadas."
)

print(
    "✓ GridSearchCV recebeu 42 combinações."
)

print("=" * 60)


# ============================================================
# FITS ESPERADOS
# ============================================================

n_folds = cv.n_splits

n_fits_cv_esperados = (
    n_combinacoes_esperadas *
    n_folds
)

n_fits_total_esperados = (
    n_fits_cv_esperados +
    1
)


print("\n" + "=" * 60)
print("EXECUÇÃO ESPERADA")
print("=" * 60)

print(
    f"Combinações : "
    f"{n_combinacoes_esperadas}"
)

print(
    f"Folds       : "
    f"{n_folds}"
)

print(
    f"Fits CV     : "
    f"{n_fits_cv_esperados}"
)

print(
    f"Refit       : 1"
)

print(
    f"Total fits  : "
    f"{n_fits_total_esperados}"
)

print(
    "\nEsperado:"
)

print(
    f"Fitting {n_folds} folds for each of "
    f"{n_combinacoes_esperadas} candidates, totalling "
    f"{n_fits_cv_esperados} fits"
)

print("=" * 60)


# ============================================================
# EXECUÇÃO
# ============================================================

print("\n" + "=" * 60)
print("INICIANDO GRID SEARCH")
print("=" * 60)

inicio = time.perf_counter()

grid_search.fit(
    x_koi_train,
    y_koi_train,
)

fim = time.perf_counter()


# ============================================================
# RESULTADOS BRUTOS
# ============================================================

resultados = grid_search.cv_results_

params_executados = resultados[
    "params"
]

n_combinacoes_executadas = len(
    params_executados
)


# ============================================================
# VERIFICAÇÃO PÓS-EXECUÇÃO
# ============================================================

print("\n" + "=" * 60)
print("VERIFICAÇÃO PÓS-EXECUÇÃO")
print("=" * 60)

print(
    f"Combinações esperadas : "
    f"{n_combinacoes_esperadas}"
)

print(
    f"Combinações executadas: "
    f"{n_combinacoes_executadas}"
)

print("=" * 60)


# ============================================================
# LISTA DAS CONFIGURAÇÕES REALMENTE EXECUTADAS
# ============================================================

print("\n" + "=" * 60)
print("CONFIGURAÇÕES REALMENTE PRESENTES NO CV_RESULTS")
print("=" * 60)

for i, params in enumerate(
    params_executados,
    start=1,
):
    print(
        f"{i:3d}: {params}"
    )

print("=" * 60)


# ============================================================
# COMPARAÇÃO ENTRE ESPERADAS E EXECUTADAS
# ============================================================

esperadas_set = {
    tuple(sorted(params.items()))
    for params in combinacoes
}

executadas_set = {
    tuple(sorted(params.items()))
    for params in params_executados
}

faltantes = (
    esperadas_set -
    executadas_set
)

extras = (
    executadas_set -
    esperadas_set
)


print("\n" + "=" * 60)
print("COMPARAÇÃO DAS COMBINAÇÕES")
print("=" * 60)

print(
    f"Esperadas : "
    f"{len(esperadas_set)}"
)

print(
    f"Executadas: "
    f"{len(executadas_set)}"
)

print(
    f"Faltantes : "
    f"{len(faltantes)}"
)

print(
    f"Extras    : "
    f"{len(extras)}"
)


if faltantes:

    print(
        "\nCONFIGURAÇÕES FALTANTES:"
    )

    for i, params in enumerate(
        sorted(faltantes),
        start=1,
    ):
        print(
            f"{i:3d}: {dict(params)}"
        )


if extras:

    print(
        "\nCONFIGURAÇÕES EXTRAS:"
    )

    for i, params in enumerate(
        sorted(extras),
        start=1,
    ):
        print(
            f"{i:3d}: {dict(params)}"
        )

print("=" * 60)


# ============================================================
# ASSERT
# ============================================================

assert n_combinacoes_executadas == 42, (
    f"\nERRO CRÍTICO:\n"
    f"Esperadas: {n_combinacoes_esperadas}\n"
    f"Executadas: {n_combinacoes_executadas}\n"
    f"Faltantes: {len(faltantes)}\n"
    f"Extras: {len(extras)}"
)


# ============================================================
# RESULTADOS DO MODELO
# ============================================================

best_lr = (
    grid_search.best_estimator_
)

print("\n" + "=" * 60)
print("RESULTADOS DO GRID SEARCH")
print("=" * 60)

print(
    "Melhores parâmetros:"
)

print(
    grid_search.best_params_
)

print(
    f"\nMelhor f1_macro (CV): "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# MÉTRICAS DE EXECUÇÃO
# ============================================================

tempo_total = (
    fim - inicio
)

n_fits_cv = (
    n_combinacoes_executadas *
    n_folds
)

n_fits_total = (
    n_fits_cv +
    1
)


# ============================================================
# TEMPOS DO SCIKIT-LEARN
# ============================================================

mean_fit_times = np.asarray(
    resultados["mean_fit_time"]
)

mean_score_times = np.asarray(
    resultados["mean_score_time"]
)

tempo_medio_fit = (
    np.mean(mean_fit_times)
)

tempo_min_fit = (
    np.min(mean_fit_times)
)

tempo_max_fit = (
    np.max(mean_fit_times)
)

tempo_medio_score = (
    np.mean(mean_score_times)
)


# ============================================================
# CPU
# ============================================================

n_cpus = os.cpu_count()


# ============================================================
# MEMÓRIA
# ============================================================

memoria_maxima = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss


if os.name == "posix":

    if platform.system() == "Darwin":

        # macOS -> bytes
        memoria_maxima_mb = (
            memoria_maxima /
            (1024 ** 2)
        )

    else:

        # Linux -> KB
        memoria_maxima_mb = (
            memoria_maxima /
            1024
        )

else:

    memoria_maxima_mb = (
        memoria_maxima /
        (1024 ** 2)
    )


# ============================================================
# RELATÓRIO DE EXECUÇÃO
# ============================================================

print("\n" + "=" * 60)
print("MÉTRICAS DE EXECUÇÃO E CONSUMO")
print("=" * 60)

print(
    f"Combinações de parâmetros : "
    f"{n_combinacoes_executadas}"
)

print(
    f"Folds por combinação      : "
    f"{n_folds}"
)

print(
    f"Fits de validação         : "
    f"{n_fits_cv}"
)

print(
    f"Fit final (refit)         : "
    f"1"
)

print(
    f"Total de fits             : "
    f"{n_fits_total}"
)

print(
    f"\nTempo total               : "
    f"{tempo_total:.2f} s"
)

print(
    f"Tempo total               : "
    f"{tempo_total / 60:.2f} min"
)

print(
    f"Tempo total               : "
    f"{tempo_total / 3600:.2f} h"
)

print(
    f"\nTempo médio de fit        : "
    f"{tempo_medio_fit:.4f} s"
)

print(
    f"Tempo mínimo de fit       : "
    f"{tempo_min_fit:.4f} s"
)

print(
    f"Tempo máximo de fit       : "
    f"{tempo_max_fit:.4f} s"
)

print(
    f"Tempo médio de scoring    : "
    f"{tempo_medio_score:.4f} s"
)

print(
    f"\nCPUs disponíveis          : "
    f"{n_cpus}"
)

print(
    f"Memória máxima residente  : "
    f"{memoria_maxima_mb:.2f} MB"
)

print("=" * 60)


# ============================================================
# TOP 10 CONFIGURAÇÕES
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)

colunas_top = [
    "rank_test_f1_macro",
    "mean_test_f1_macro",
    "std_test_f1_macro",
    "mean_test_accuracy",
    "mean_test_precision_macro",
    "mean_test_recall_macro",
    "mean_fit_time",
    "param_C",
    "param_l1_ratio",
    "param_solver",
    "param_class_weight",
]

top_10 = (
    df_resultados[
        colunas_top
    ]
    .sort_values(
        "rank_test_f1_macro"
    )
    .head(10)
)

print("\n" + "=" * 60)
print("TOP 10 CONFIGURAÇÕES POR F1 MACRO")
print("=" * 60)

print(
    top_10.to_string(
        index=False
    )
)


# ============================================================
# ANÁLISE DE C (força de regularização)
# ============================================================

comparacao_c = (
    df_resultados
    .groupby(
        "param_C"
    )[
        "mean_test_f1_macro"
    ]
    .agg(
        media="mean",
        maximo="max",
        desvio="std",
    )
)

print("\n" + "=" * 60)
print("ANÁLISE DE C (força de regularização)")
print("=" * 60)

print(
    comparacao_c.to_string()
)


# ============================================================
# ANÁLISE DE L1_RATIO / SOLVER
# ============================================================

comparacao_l1ratio_solver = (
    df_resultados
    .groupby(
        ["param_l1_ratio", "param_solver"]
    )[
        "mean_test_f1_macro"
    ]
    .agg(
        media="mean",
        maximo="max",
        desvio="std",
    )
)

print("\n" + "=" * 60)
print("ANÁLISE DE L1_RATIO / SOLVER")
print("=" * 60)

print(
    comparacao_l1ratio_solver.to_string()
)


# ============================================================
# MODELO FINAL
# ============================================================

print("\n" + "=" * 60)
print("MODELO FINAL")
print("=" * 60)

for parametro, valor in (
    grid_search.best_params_.items()
):

    print(
        f"{parametro:20s}: "
        f"{valor}"
    )

print(
    f"\nF1 macro CV: "
    f"{grid_search.best_score_:.4f}"
)

print("=" * 60)
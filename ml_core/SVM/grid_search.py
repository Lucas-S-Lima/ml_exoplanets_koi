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

from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    ParameterGrid,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

pkl_path = ROOT_DIR / "exoplanets_split.pkl"

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
# CONFIGURAÇÃO DO GRID (rodada 3 — linear/sigmoid de volta
# apenas para comparação + busca fina de C no rbf)
# ============================================================
# Contexto das rodadas anteriores:
#   - Rodada 1 (88 combinações): rbf venceu (máx 0,9045), linear
#     ficou ~7-8 pontos abaixo (máx 0,8288) e sigmoid foi descartado
#     por instabilidade (máx 0,688, desvio 0,093).
#   - Rodada 2 (40 combinações, só rbf+poly, C até 1000): melhor
#     resultado C=300/gamma=0.05 (0,9072); C=1000 empatou (0,90715),
#     sinalizando platô entre 300 e 1000.
#
# Nesta rodada:
#   - linear e sigmoid voltam SÓ para reconfirmar visualmente a
#     distância em relação ao rbf/poly (não esperamos que vençam).
#     Grid de C deles é mais modesto, já que não estamos buscando
#     otimizar esses dois — só documentar a comparação.
#   - rbf recebe um grid de C bem mais fino ao redor da região onde
#     o platô foi detectado (150-1000) e estende até 2000 para
#     confirmar que a curva realmente não volta a subir depois disso.
#   - gamma do rbf ganha mais pontos ao redor de 0.05 (o valor que
#     venceu na rodada 2), para mapear a vizinhança dele com mais
#     resolução.
#   - poly mantido com o mesmo grid da rodada 2 (não houve indício de
#     que valha a pena estender).
#
# ATENÇÃO: o solver do SVM linear fica sensivelmente mais lento em
# C alto (na rodada 1, C=100 já levava 3,5-5,4s por fit, contra
# 0,4-0,8s do rbf no mesmo C). max_iter foi limitado para evitar que
# um fit fique rodando por tempo desproporcional.

C_RANGE_LINEAR = [0.1, 1, 10, 100, 300, 1000]
C_RANGE_RBF = [50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]
GAMMA_RANGE_RBF = ["scale", 0.03, 0.05, 0.07, 0.1, 0.2]
C_RANGE_POLY = [1, 10, 100, 300]
GAMMA_RANGE_POLY = ["scale", 0.1]
C_RANGE_SIGMOID = [1, 10, 100]
GAMMA_RANGE_SIGMOID = ["scale", 0.05, 0.1]

MAX_ITER_SEGURANCA = 50_000

param_grid = [
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
    "Bloco linear  :",
    len(list(ParameterGrid(param_grid[0])))
)

print(
    "Bloco rbf     :",
    len(list(ParameterGrid(param_grid[1])))
)

print(
    "Bloco poly    :",
    len(list(ParameterGrid(param_grid[2])))
)

print(
    "Bloco sigmoid :",
    len(list(ParameterGrid(param_grid[3])))
)

print(
    f"\nCombinações esperadas: "
    f"{n_combinacoes_esperadas}"
)

print(
    f"✓ {n_combinacoes_esperadas} combinações calculadas dinamicamente "
    f"a partir do param_grid."
)

print("=" * 60)


# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
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

estimator = SVC(
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
    n_jobs=10,
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

assert len(grid_antes_do_fit) == n_combinacoes_esperadas, (
    "ERRO: o GridSearchCV não recebeu "
    "as combinações esperadas."
)

print(
    f"✓ GridSearchCV recebeu {n_combinacoes_esperadas} combinações."
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

print(
    "\nAtenção: o bloco 'linear' com C alto (300, 1000) tende a "
    "ser o mais lento desta rodada — acompanhe o verbose."
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

assert n_combinacoes_executadas == n_combinacoes_esperadas, (
    f"\nERRO CRÍTICO:\n"
    f"Esperadas: {n_combinacoes_esperadas}\n"
    f"Executadas: {n_combinacoes_executadas}\n"
    f"Faltantes: {len(faltantes)}\n"
    f"Extras: {len(extras)}"
)


# ============================================================
# RESULTADOS DO MODELO (VALIDAÇÃO CRUZADA)
# ============================================================

best_svm = (
    grid_search.best_estimator_
)

print("\n" + "=" * 60)
print("RESULTADOS DO GRID SEARCH (CV)")
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

print(
    f"Vetores de suporte    : "
    f"{best_svm.n_support_.sum()} "
    f"(de {x_koi_train.shape[0]} amostras de treino)"
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
# TOP 10 CONFIGURAÇÕES (GERAL)
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
    "param_kernel",
    "param_C",
    "param_gamma",
    "param_degree",
    "param_coef0",
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
print("TOP 10 CONFIGURAÇÕES POR F1 MACRO (GERAL)")
print("=" * 60)

print(
    top_10.to_string(
        index=False
    )
)


# ============================================================
# MELHOR CONFIGURAÇÃO POR KERNEL
# (a comparação que interessa: cada kernel no seu próprio melhor)
# ============================================================

melhor_por_kernel = (
    df_resultados
    .sort_values("rank_test_f1_macro")
    .groupby("param_kernel", as_index=False)
    .first()[colunas_top]
    .sort_values("mean_test_f1_macro", ascending=False)
)

print("\n" + "=" * 60)
print("MELHOR CONFIGURAÇÃO DE CADA KERNEL")
print("=" * 60)

print(
    melhor_por_kernel.to_string(
        index=False
    )
)


# ============================================================
# ANÁLISE POR KERNEL
# ============================================================

comparacao_kernel = (
    df_resultados
    .groupby(
        "param_kernel"
    )[
        "mean_test_f1_macro"
    ]
    .agg(
        media="mean",
        maximo="max",
        desvio="std",
    )
    .sort_values(
        "maximo",
        ascending=False,
    )
)

print("\n" + "=" * 60)
print("ANÁLISE POR KERNEL")
print("=" * 60)

print(
    comparacao_kernel.to_string()
)


# ============================================================
# ANÁLISE DE C (por kernel) — busca da borda fina
# ============================================================

comparacao_c = (
    df_resultados
    .groupby(
        ["param_kernel", "param_C"]
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
print("ANÁLISE DE C (por kernel)")
print("=" * 60)

print(
    comparacao_c.to_string()
)


# ============================================================
# ANÁLISE DE GAMMA (por kernel, quando aplicável)
# ============================================================

mask_com_gamma = (
    df_resultados["param_kernel"] != "linear"
)

comparacao_gamma = (
    df_resultados[mask_com_gamma]
    .groupby(
        ["param_kernel", "param_gamma"]
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
print("ANÁLISE DE GAMMA (kernels rbf/poly/sigmoid)")
print("=" * 60)

print(
    comparacao_gamma.to_string()
)


# ============================================================
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ============================================================
# Comparável com os números já registrados de Regressão Logística
# (acurácia teste 84,71%), Random Forest (acurácia teste 92,57%) e
# SVM rodada 2 (acurácia teste 93,04%, F1 macro teste 0,9254).

print("\n" + "=" * 60)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("=" * 60)

y_pred_test = best_svm.predict(x_koi_test)

acuracia_teste = accuracy_score(
    y_koi_test, y_pred_test
)

f1_macro_teste = f1_score(
    y_koi_test, y_pred_test, average="macro"
)

matriz_confusao = confusion_matrix(
    y_koi_test, y_pred_test
)

print(
    f"Acurácia (teste)  : {acuracia_teste:.4f}"
)

print(
    f"F1 macro (teste)  : {f1_macro_teste:.4f}"
)

print(
    f"\nMatriz de confusão:\n{matriz_confusao}"
)

print(
    f"\nRelatório de classificação:\n"
    f"{classification_report(y_koi_test, y_pred_test)}"
)

print("=" * 60)


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
    f"\nF1 macro (CV)   : "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"F1 macro (teste): "
    f"{f1_macro_teste:.4f}"
)

print(
    f"Acurácia (teste): "
    f"{acuracia_teste:.4f}"
)

print("=" * 60)
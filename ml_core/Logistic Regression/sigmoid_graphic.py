import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================
# Mesmo padrão usado no script de GridSearchCV

ROOT_DIR = Path(__file__).resolve().parents[2]
pkl_path = ROOT_DIR / "exoplanets.pkl"

print("\n" + "=" * 60)
print("CARREGAMENTO DOS DADOS")
print("=" * 60)

if not pkl_path.exists():
    raise FileNotFoundError(f"Arquivo não encontrado:\n{pkl_path}")

data = joblib.load(pkl_path)

x_koi_train = data["x_train"]
x_koi_test = data["x_test"]
y_koi_train = data["y_train"]
y_koi_test = data["y_test"]

print(f"x_train: {x_koi_train.shape} | x_test: {x_koi_test.shape}")
print("=" * 60)


# ============================================================
# TREINO DO MELHOR MODELO ENCONTRADO NO GRIDSEARCHCV
# ============================================================
# Melhores parâmetros obtidos: C=3000, l1_ratio=0.0 (L2),
# class_weight=None, solver='lbfgs', max_iter=2000

melhores_parametros = {
    "C": 3000,
    "l1_ratio": 0.0,
    "class_weight": None,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 42,
}

print("\n" + "=" * 60)
print("TREINANDO MODELO FINAL")
print("=" * 60)

print("Parâmetros:", melhores_parametros)

modelo = LogisticRegression(**melhores_parametros)
modelo.fit(x_koi_train, y_koi_train)

print("Classes:", modelo.classes_)
print("=" * 60)


# ============================================================
# CÁLCULO DE Z (SAÍDA LINEAR) E PROBABILIDADES REAIS
# ============================================================
# z = intercept_ + coef_ · x  -> é isso que decision_function retorna
# sigmoid(z) = predict_proba()[:, 1] -> probabilidade da classe positiva

z_test = modelo.decision_function(x_koi_test)
proba_test = modelo.predict_proba(x_koi_test)[:, 1]

print("\n" + "=" * 60)
print("ESTATÍSTICAS DE Z (SAÍDA LINEAR) NO CONJUNTO DE TESTE")
print("=" * 60)

print(f"z mínimo : {z_test.min():.3f}")
print(f"z máximo : {z_test.max():.3f}")
print(f"z médio  : {z_test.mean():.3f}")
print("=" * 60)


# ============================================================
# FUNÇÃO DE PLOTAGEM DA SIGMOIDE (REUTILIZÁVEL)
# ============================================================

def plotar_sigmoide(
    z_test,
    proba_test,
    y_test_array,
    classes,
    z_min_plot,
    z_max_plot,
    titulo_extra,
    output_path,
    anotar_outliers=False,
):
    classe_negativa, classe_positiva = classes

    mascara_positiva = y_test_array == classe_positiva
    mascara_negativa = y_test_array == classe_negativa

    z_curva = np.linspace(z_min_plot, z_max_plot, 500)
    sigmoide_curva = 1 / (1 + np.exp(-z_curva))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(
        z_curva,
        sigmoide_curva,
        color="#1f77b4",
        linewidth=2.5,
        label="Sigmoide: σ(z) = 1 / (1 + e⁻ᶻ)",
        zorder=2,
    )

    ax.scatter(
        z_test[mascara_negativa],
        proba_test[mascara_negativa],
        s=18,
        alpha=0.5,
        color="#d62728",
        label=f"Amostras reais: {classe_negativa}",
        zorder=3,
    )

    ax.scatter(
        z_test[mascara_positiva],
        proba_test[mascara_positiva],
        s=18,
        alpha=0.5,
        color="#2ca02c",
        label=f"Amostras reais: {classe_positiva}",
        zorder=3,
    )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, zorder=1)

    ax.text(
        0.02,
        0.53,
        "limiar de decisão (0.5)",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
    )

    ax.set_xlabel("z = β₀ + β₁x₁ + ... + βₙxₙ  (saída linear do modelo)")
    ax.set_ylabel("Probabilidade prevista  σ(z)")
    ax.set_title(
        "Função Sigmoide aplicada às amostras reais de teste"
        + titulo_extra
        + "\n"
        f"LogisticRegression (C={melhores_parametros['C']}, "
        f"l1_ratio={melhores_parametros['l1_ratio']}, "
        f"solver={melhores_parametros['solver']})"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(z_min_plot, z_max_plot)

    if anotar_outliers:
        n_outliers_acima = int(np.sum(z_test > z_max_plot))
        n_outliers_abaixo = int(np.sum(z_test < z_min_plot))

        if n_outliers_acima > 0:
            ax.annotate(
                f"+{n_outliers_acima} amostra(s) com\nz > {z_max_plot:.0f}\n"
                f"(z máx. real = {z_test.max():.0f})",
                xy=(0.98, 0.5),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
                color="dimgray",
                bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="lightgray"),
            )

        if n_outliers_abaixo > 0:
            ax.annotate(
                f"+{n_outliers_abaixo} amostra(s) com\nz < {z_min_plot:.0f}\n"
                f"(z mín. real = {z_test.min():.0f})",
                xy=(0.02, 0.5),
                xycoords="axes fraction",
                ha="left",
                fontsize=8,
                color="dimgray",
                bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="lightgray"),
            )

    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Gráfico salvo em: {output_path}")


y_test_array = np.asarray(y_koi_test)
classes = (modelo.classes_[0], modelo.classes_[1])

pasta_saida = Path(__file__).resolve().parent


# ------------------------------------------------------------
# 1) GRÁFICO COM RECORTE (zoom robusto via IQR)
# ------------------------------------------------------------

q1, q3 = np.percentile(z_test, [25, 75])
iqr = q3 - q1
margem = 2.0

# Limite tipo "boxplot" (1.5x IQR), mas nunca mais estreito que [-6, 6],
# que é a faixa onde a sigmoide já está praticamente saturada
z_min_recorte = min(q1 - 1.5 * iqr, -6) - margem
z_max_recorte = max(q3 + 1.5 * iqr, 6) + margem

n_outliers_abaixo = int(np.sum(z_test < z_min_recorte))
n_outliers_acima = int(np.sum(z_test > z_max_recorte))

print("\n" + "=" * 60)
print("RECORTE DO EIXO X (ZOOM NA REGIÃO DE TRANSIÇÃO)")
print("=" * 60)
print(f"Faixa exibida no gráfico : [{z_min_recorte:.2f}, {z_max_recorte:.2f}]")
print(f"Amostras fora à esquerda : {n_outliers_abaixo}")
print(f"Amostras fora à direita  : {n_outliers_acima}")
print("=" * 60)

plotar_sigmoide(
    z_test=z_test,
    proba_test=proba_test,
    y_test_array=y_test_array,
    classes=classes,
    z_min_plot=z_min_recorte,
    z_max_plot=z_max_recorte,
    titulo_extra=" (com recorte no eixo x)",
    output_path=pasta_saida / "sigmoide_com_recorte.png",
    anotar_outliers=True,
)


# ------------------------------------------------------------
# 2) GRÁFICO SEM RECORTE (range completo dos dados reais)
# ------------------------------------------------------------

margem_completa = max((z_test.max() - z_test.min()) * 0.02, 0.5)
z_min_completo = z_test.min() - margem_completa
z_max_completo = z_test.max() + margem_completa

plotar_sigmoide(
    z_test=z_test,
    proba_test=proba_test,
    y_test_array=y_test_array,
    classes=classes,
    z_min_plot=z_min_completo,
    z_max_plot=z_max_completo,
    titulo_extra=" (sem recorte — range completo)",
    output_path=pasta_saida / "sigmoide_sem_recorte.png",
    anotar_outliers=False,
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

y_pred = modelo.predict(x_koi_test)

print("\n" + "=" * 60)
print("RELATÓRIO DE CLASSIFICAÇÃO (CONJUNTO DE TESTE)")
print("=" * 60)
print(classification_report(y_koi_test, y_pred))
print("=" * 60)

matriz = confusion_matrix(y_koi_test, y_pred, labels=modelo.classes_)

fig_cm, ax_cm = plt.subplots(figsize=(6.5, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=modelo.classes_,
)

disp.plot(ax=ax_cm, cmap="Blues", colorbar=True, values_format="d")

ax_cm.set_title(
    "Matriz de Confusão — Regressão Logística\n"
    f"(C={melhores_parametros['C']}, l1_ratio={melhores_parametros['l1_ratio']}, "
    f"solver={melhores_parametros['solver']})"
)

fig_cm.tight_layout()

cm_output_path = pasta_saida / "matriz_confusao.png"
fig_cm.savefig(cm_output_path, dpi=150)
plt.close(fig_cm)

print(f"\nMatriz de confusão salva em: {cm_output_path}")
from pathlib import Path

import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# ============================================================
# GERAR E SALVAR FRONTEIRA DE DECISÃO (PCA 2D)
# ============================================================

print("\n" + "=" * 60)
print("GERANDO GRÁFICO DA FRONTEIRA DE DECISÃO (PCA 2D)")
print("=" * 60)

# 1. Padronização + Redução de Dimensionalidade (12D -> 2D)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_koi_train)

pca = PCA(n_components=2)
x_train_2d = pca.fit_transform(x_train_scaled)

# 2. Treinamento do SVM RBF no espaço reduzido (2D)
svm_2d = SVC(
    kernel="rbf",
    C=400,
    gamma="scale",
    class_weight=None,
    random_state=42
)
svm_2d.fit(x_train_2d, y_koi_train)

# 3. Criação da malha (mesh) para delimitar o hiperplano
x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predição em toda a extensão da malha
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. Construção da figura
plt.figure(figsize=(10, 7))

# Áreas delimitadas pelo hiperplano
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Dispersão dos pontos de treino
scatter = plt.scatter(
    x_train_2d[:, 0],
    x_train_2d[:, 1],
    c=y_koi_train,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    alpha=0.6,
    s=30
)

# Destaque dos Vetores de Suporte
sv = svm_2d.support_vectors_
plt.scatter(
    sv[:, 0],
    sv[:, 1],
    s=80,
    facecolors="none",
    edgecolors="k",
    linewidths=1.2,
    label="Vetores de Suporte"
)

plt.title("Fronteira de Decisão do SVM (RBF) via Redução PCA (2D)", fontsize=12)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(*scatter.legend_elements(), title="Classes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# 5. Salvamento da imagem e exibição
output_image_path = ROOT_DIR / "svm_decision_boundary.png"
plt.savefig(output_image_path, dpi=300)
print(f"Gráfico salvo com sucesso em: {output_image_path}")

plt.show()
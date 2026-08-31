import joblib
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

pkl_path = ROOT_DIR / "exoplanets_split.pkl"

data = joblib.load(pkl_path)

x_koi_train = data["x_train"]
x_koi_test = data["x_test"]

y_koi_train = data["y_train"]
y_koi_test = data["y_test"]


# ============================================================
# SUPPORT VECTOR MACHINE (AVALIAÇÃO DE DESEMPENHO EM 12D)
# ============================================================

svm_exoplanets = SVC(
    kernel="rbf",
    C=400,
    gamma=0.05,
    class_weight=None,
    random_state=42
)

# ============================================================
# TREINAMENTO
# ============================================================

svm_exoplanets.fit(x_koi_train, y_koi_train)

# ============================================================
# PREDIÇÃO
# ============================================================

predictions = svm_exoplanets.predict(x_koi_test)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("SVM CLASSIFIER REPORT")
print("=" * 60)

print(
    classification_report(
        y_koi_test,
        predictions
    )
)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_koi_test,
    predictions
)

print("Accuracy Score:")
print(f"{accuracy:.4f}")

# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

ConfusionMatrixDisplay.from_estimator(
    svm_exoplanets,
    x_koi_test,
    y_koi_test,
    cmap="Blues"
)

plt.title("Matriz de Confusão - SVM")
plt.tight_layout()
plt.show()


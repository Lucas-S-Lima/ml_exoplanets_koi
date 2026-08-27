import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

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
# RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    criterion="entropy",
    class_weight="balanced_subsample",
    max_depth=30,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# TREINAMENTO
# ============================================================

random_forest.fit(
    x_koi_train,
    y_koi_train
)

# ============================================================
# PREDIÇÃO
# ============================================================

predictions = random_forest.predict(
    x_koi_test
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFIER REPORT")
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
    random_forest,
    x_koi_test,
    y_koi_test,
    cmap="Blues"
)

plt.title("Matriz de Confusão - Random Forest")
plt.tight_layout()
plt.show()



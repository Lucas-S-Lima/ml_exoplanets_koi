import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from main import x_koi_train, y_koi_train


n_estimators_range = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
oob_errors = []

for n in n_estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(x_koi_train, y_koi_train)
    oob_errors.append(1 - rf.oob_score_)

plt.figure(figsize=(8, 5))
plt.plot(n_estimators_range, oob_errors, marker='o')
plt.xlabel('Número de árvores (n_estimators)')
plt.ylabel('Erro OOB')
plt.title('Erro OOB vs. Número de árvores - Random Forest')
plt.grid(True)
plt.show()
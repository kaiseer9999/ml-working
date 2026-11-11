"""Практическая работа №9: SVM, подбор C и gamma на make_moons."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUT = Path("svm_pr9_output")


def main():
    OUT.mkdir(exist_ok=True)
    X, y = make_moons(n_samples=600, noise=0.28, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Масштабирование находится внутри Pipeline и заново обучается в каждом fold.
    pipeline = Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": [0.01, 0.1, 1, 10],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline, param_grid, scoring="accuracy", cv=cv, n_jobs=-1,
        return_train_score=True,
    )
    start = perf_counter()
    search.fit(X_train, y_train)
    grid_seconds = perf_counter() - start
    best = search.best_estimator_
    pred = best.predict(X_test)

    # Линейный SVC для ориентира, с таким же scaler и тем же CV.
    linear = Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="linear", C=1))])
    start = perf_counter()
    linear.fit(X_train, y_train)
    linear_seconds = perf_counter() - start
    linear_pred = linear.predict(X_test)

    rows = []
    for p, mean, std, train in zip(
        search.cv_results_["params"], search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"], search.cv_results_["mean_train_score"],
    ):
        rows.append({"C": p["svc__C"], "gamma": p["svc__gamma"],
                     "cv_accuracy": mean, "cv_std": std, "train_accuracy": train})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "grid_results.csv", index=False)
    pivot = results.pivot(index="C", columns="gamma", values="cv_accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("Средняя accuracy на 5-fold CV (RBF SVC)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="CV accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "grid_heatmap.png", dpi=160)
    plt.close(fig)

    # Решение изображается в исходных координатах; весь prediction проходит через scaler.
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 350),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 350),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    zz = best.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=30, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.5)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="black", s=28)
    ax.set_title(f"RBF SVC: C={search.best_params_['svc__C']}, gamma={search.best_params_['svc__gamma']}")
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    fig.tight_layout()
    fig.savefig(OUT / "decision_boundary.png", dpi=160)
    plt.close(fig)

    print(f"Лучшие параметры RBF: {search.best_params_}")
    print(f"Лучшая CV accuracy: {search.best_score_:.4f}")
    print(f"RBF test accuracy/F1: {accuracy_score(y_test, pred):.4f} / {f1_score(y_test, pred):.4f}")
    print(f"Linear test accuracy/F1: {accuracy_score(y_test, linear_pred):.4f} / {f1_score(y_test, linear_pred):.4f}")
    print(f"Время GridSearchCV (включая CV): {grid_seconds:.3f} с")
    print(f"Время fit линейной модели: {linear_seconds:.3f} с")
    print(f"Артефакты сохранены в {OUT.resolve()}")


if __name__ == "__main__":
    main()

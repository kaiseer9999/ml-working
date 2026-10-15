import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc
)


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

df = pd.read_csv("data/data.csv")

print("=" * 60)
print("ПРАКТИЧЕСКАЯ РАБОТА — БИНАРНАЯ КЛАССИФИКАЦИЯ")
print("=" * 60)

print("\nРазмер датасета:")
print(df.shape)

print("\nПервые 5 строк:")
print(df.head())


# ============================================================
# 2. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ============================================================

target = "Heart Disease Status"

# Удаляем строки, где отсутствует целевая переменная
df = df.dropna(subset=[target])

# No = 0
# Yes = 1
df[target] = df[target].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=[target])
y = df[target]

print("\nРаспределение классов:")
print(y.value_counts())

print("\nДоля классов:")
print(y.value_counts(normalize=True))


# ============================================================
# 3. ОПРЕДЕЛЯЕМ ЧИСЛОВЫЕ И КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nЧисловые признаки:")
print(numeric_features)

print("\nКатегориальные признаки:")
print(categorical_features)


# ============================================================
# 4. ПРЕДОБРАБОТКА ЧИСЛОВЫХ ПРИЗНАКОВ
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 5. ПРЕДОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 6. ОБЩАЯ ПРЕДОБРАБОТКА
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 7. СОЗДАЁМ МОДЕЛЬ
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)


# ============================================================
# 8. РАЗДЕЛЕНИЕ TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nРазмер обучающей выборки:")
print(X_train.shape)

print("\nРазмер тестовой выборки:")
print(X_test.shape)


# ============================================================
# 9. ОБУЧЕНИЕ
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nМодель успешно обучена.")


# ============================================================
# 10. ПОЛУЧАЕМ ВЕРОЯТНОСТИ
# ============================================================

y_proba = model.predict_proba(
    X_test
)[:, 1]

print("\nПервые 10 вероятностей:")
print(y_proba[:10])


# ============================================================
# 11. ROC-AUC
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_proba
)

roc_auc = roc_auc_score(
    y_test,
    y_proba
)

print("\nROC-AUC:")
print(round(roc_auc, 4))


# ============================================================
# 12. ROC-КРИВАЯ
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC-кривая")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 13. PR-КРИВАЯ
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_proba
    )
)

pr_auc = auc(
    recall_curve,
    precision_curve
)

print("\nPR-AUC:")
print(round(pr_auc, 4))


plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("PR-кривая")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 14. ФУНКЦИЯ ОЦЕНКИ ПОРОГА
# ============================================================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return {
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 15. ПОРОГ ПО УМОЛЧАНИЮ = 0.5
# ============================================================

default_result = evaluate_threshold(
    y_test,
    y_proba,
    0.5
)

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ПРИ ПОРОГЕ 0.5")
print("=" * 60)

for key, value in default_result.items():

    if isinstance(value, float):
        print(
            f"{key}: {value:.4f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# 16. ПРОВЕРЯЕМ РАЗНЫЕ ПОРОГИ
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []

for threshold in thresholds:

    result = evaluate_threshold(
        y_test,
        y_proba,
        threshold
    )

    results.append(result)


results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 60)
print("СРАВНЕНИЕ ПОРОГОВ")
print("=" * 60)

print(
    results_df[
        [
            "threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "precision",
            "recall",
            "f1"
        ]
    ].round(3)
)


# ============================================================
# 17. ВЫБОР ПОРОГА
# ============================================================

# В этой задаче считаем,
# что пропустить заболевание (FN)
# хуже, чем ошибочно определить заболевание (FP).

FN_COST = 5
FP_COST = 1

results_df["cost"] = (
    FN_COST * results_df["FN"]
    +
    FP_COST * results_df["FP"]
)

best_row = results_df.loc[
    results_df["cost"].idxmin()
]

best_threshold = best_row["threshold"]


print("\n" + "=" * 60)
print("ВЫБРАННЫЙ ПОРОГ")
print("=" * 60)

print(
    "Порог:",
    round(best_threshold, 2)
)

print(
    "Стоимость ошибок:",
    int(best_row["cost"])
)


# ============================================================
# 18. РЕЗУЛЬТАТЫ ПРИ НОВОМ ПОРОГЕ
# ============================================================

best_result = evaluate_threshold(
    y_test,
    y_proba,
    best_threshold
)

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ПРИ НОВОМ ПОРОГЕ")
print("=" * 60)

for key, value in best_result.items():

    if isinstance(value, float):
        print(
            f"{key}: {value:.4f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# 19. ИТОГОВОЕ СРАВНЕНИЕ
# ============================================================

comparison = pd.DataFrame(
    [
        default_result,
        best_result
    ]
)

print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 60)

print(
    comparison[
        [
            "threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ].round(4)
)


# ============================================================
# 20. МАТРИЦЫ ОШИБОК
# ============================================================

print("\nМатрица ошибок при пороге 0.5:")

print(
    confusion_matrix(
        y_test,
        (
            y_proba >= 0.5
        ).astype(int)
    )
)


print("\nМатрица ошибок при новом пороге:")

print(
    confusion_matrix(
        y_test,
        (
            y_proba >= best_threshold
        ).astype(int)
    )
)


print("\n" + "=" * 60)
print("РАБОТА ЗАВЕРШЕНА")
print("=" * 60)
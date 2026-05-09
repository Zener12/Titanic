# Предсказание выживаемости на Титанике

Соревнование Kaggle — задача бинарной классификации: предсказать выживание пассажиров Титаника.

**Публичный результат: 0.801 (топ 6%)**

## Подход

Полный ML-пайплайн в Jupyter Notebook:

1. **EDA** — анализ выживаемости по полу, классу, порту посадки; корреляционный анализ; выбросы
2. **Препроцессинг** — заполнение пропусков возраста медианой по группам `[Pclass, Sex]` без утечки данных, флаг наличия каюты, размер семьи
3. **Feature Engineering**
   - `Title` — обращение из имени пассажира (Mr / Mrs / Miss / Master / Rare)
   - `FamSize` = SibSp + Parch + 1
   - `HasCabin` — бинарный флаг
   - `fam_survive_rate` — выживаемость семьи с leave-one-out по ключу `(Surname, Pclass, FamSize)`
4. **Baseline** — Logistic Regression с OHE, точность 83.8%
5. **Финальная модель** — CatBoost с нативной обработкой категориальных признаков, Stratified CV

## Результаты

| Модель | Accuracy (CV) | Публичный score |
|---|---|---|
| Logistic Regression | 83.8% | — |
| CatBoost | 86.0% | 0.801 |

## Стек

- Python, pandas, numpy
- scikit-learn, CatBoost
- matplotlib, seaborn
- Jupyter Notebook

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook titanic.ipynb
```

Данные скачиваются автоматически через `kagglehub` при первом запуске (требуется аккаунт Kaggle).

---

# Titanic Survival Prediction

Kaggle competition — binary classification task: predict passenger survival on the Titanic.

**Public leaderboard score: 0.801 (top 6%)**

## Approach

Full ML pipeline built in Jupyter Notebook:

1. **EDA** — survival rates by sex, class, port; correlation analysis; outlier detection
2. **Preprocessing** — age imputation by `[Pclass, Sex]` group median (no data leakage), cabin presence flag, family size feature
3. **Feature Engineering**
   - `Title` extracted from passenger name (Mr / Mrs / Miss / Master / Rare)
   - `FamSize` = SibSp + Parch + 1
   - `HasCabin` — binary flag
   - `fam_survive_rate` — leave-one-out family survival rate grouped by `(Surname, Pclass, FamSize)`
4. **Baseline** — Logistic Regression with OHE, accuracy 83.8%
5. **Final model** — CatBoost with native categorical feature handling, Stratified CV

## Results

| Model | Accuracy (CV) | Public Score |
|---|---|---|
| Logistic Regression | 83.8% | — |
| CatBoost | 86.0% | 0.801 |

## Stack

- Python, pandas, numpy
- scikit-learn, CatBoost
- matplotlib, seaborn
- Jupyter Notebook

## Run

```bash
pip install -r requirements.txt
jupyter notebook titanic.ipynb
```

Data is downloaded automatically via `kagglehub` on first run (requires Kaggle account).

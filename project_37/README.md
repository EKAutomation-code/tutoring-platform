
# Классификация цифр 0 и 1 из MNIST с использованием PCA и градиентного бустинга

Проект по бинарной классификации изображений рукописных цифр 0 и 1 из датасета MNIST. Используется метод главных компонент (PCA) для снижения размерности и градиентный бустинг (GradientBoostingClassifier) для классификации.

## Структура проекта

```
project_37/
├── project_37.py # Скрипт с моделью
├── MNIST_train.csv.gz # Обучающая выборка (загрузить отдельно)
├── MNIST_test.csv.gz # Тестовая выборка (загрузить отдельно)
└── README.md # Описание проекта
```

## Что делает скрипт

1. Загружает обучающую и тестовую выборки MNIST
2. Фильтрует только цифры **0** и **1** (бинарная классификация)
3. Применяет **PCA (Principal Component Analysis)** для снижения размерности с 784 признаков до M компонент
4. Обучает модель **градиентного бустинга (Gradient Boosting)**
5. Выводит матрицу ошибок и метрики качества

## Технологии

- Python 3.x
- `pandas` — работа с данными
- `numpy` — математические операции
- `scikit-learn`:
  - `PCA` — снижение размерности
  - `GradientBoostingClassifier` — градиентный бустинг
  - `accuracy_score`, `confusion_matrix`, `classification_report` — метрики качества

## Параметры модели

```
PCA                        | `svd_solver='full'`
Доля объяснённой дисперсии | 90% (порог)
Gradient Boosting          | `n_estimators=500`
Learning rate              | `0.8`
Max depth                  | `2`
Random state               | `23`
```

## Запуск проекта

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/EKAutomation-code/work.git
cd work/project_37
```

### 2. Установите зависимости

```bash
pip install pandas numpy scikit-learn
```

### 3. Запустите скрипт

```bash
python project_37.py
```

## Понимание ключевых метрик

```
M                          	   | Количество главных компонент, объясняющих 90% дисперсии данных
Координата 1-ой главной компоненты | Проекция первого изображения на первую главную компоненту
tp                                 | Количество правильно предсказанных цифр 1
Точность                           | Доля правильных ответов от всех предсказаний
```

## Настройка

Чтобы изменить порог PCA, отредактируйте строку в project_37.py:

### Было: 90%
```python
M = np.where(cumsum > 0.9)[0][0] + 1
```

### Стало: 95%
```python
M = np.where(cumsum > 0.95)[0][0] + 1
```

Чтобы изменить random_state, отредактируйте строку в project_37.py:

### Было: 23
```python
random_state=23
```

### Стало: 42
```python
random_state=42
```

Чтобы изменить параметры градиентного бустинга, отредактируйте строку в project_37.py:

### Было: 500 деревьев, learning_rate 0.8, max_depth 2
```python
GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.8,
    random_state=23,
    max_depth=2
)
```

### Стало: 300 деревьев, learning_rate 0.5, max_depth 3
```python
GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.5,
    random_state=23,
    max_depth=3
)
```

## Лицензия

MIT — код можно использовать и изменять для любых целей.
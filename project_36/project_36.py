import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_df = pd.read_csv('C://Users//egorb//Downloads//MNIST_train.csv')
test_df = pd.read_csv('C://Users//egorb//Downloads//MNIST_test.csv')
train_binary = train_df[train_df['label'].isin([0, 6])]
test_binary = test_df[test_df['label'].isin([0, 6])]
X_train = train_binary.drop('label', axis=1).values
y_train = train_binary['label'].values
X_test = test_binary.drop('label', axis=1).values
y_test = test_binary['label'].values
pca_full = PCA(svd_solver='full')
pca_full.fit(X_train)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
M = np.where(cumsum > 0.87)[0][0] + 1
pca = PCA(n_components=M, svd_solver='full')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
ninth_component_coord = X_train_pca[0, 8]
base_learners = [
    ('random_forest', RandomForestClassifier(random_state=19)),
    ('svc', CalibratedClassifierCV(SVC(random_state=19), ensemble=False))
]
final_estimator = LogisticRegression(random_state=19)
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=final_estimator,
    cv=5,
    stack_method='predict_proba'
)
stacking_model.fit(X_train_pca, y_train)
y_pred = stacking_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Цифра 0', 'Цифра 6']))
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
print(f"Количество компонент M: {M}")
print(f"Координата 9-й главной компоненты для 1-го изображения: {round(ninth_component_coord,3)}")
print(f"Точность модели: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"tp: {cm[1][1]}")




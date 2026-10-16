"""Week 2 Assignment - Task 3 (+ Task 5 bonus): compare classifiers on HOG features.

Trains Linear SVM, k-NN (k=5) and (bonus) Random Forest on the Setting A HOG
features produced by extract_features.py, and prints accuracy, a
classification report and a confusion matrix for each, followed by a summary
table.

The models dict and evaluation loop follow the starter snippet in the Week 2
Coding Assignment handout (Image and Scene Recognition, Kyungdong University);
classifiers and metrics are from scikit-learn: https://scikit-learn.org/stable/

Run from inside week2_assignment/ after extract_features.py:
    python compare_classifiers.py
"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Setting A (fine detail) features saved by extract_features.py
X = np.load('features_setting_A.npy')
y = np.load('labels_setting_A.npy')
class_names = np.load('class_names.npy', allow_pickle=True)

# Stratified split keeps class proportions equal in both sets; a fixed
# random_state makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    'Linear SVM': LinearSVC(max_iter=10000),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),  # Task 5 bonus
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f'--- {name} ---')
    print('Accuracy:', acc)
    print('Classification Report:')
    print(classification_report(y_test, preds, target_names=class_names))
    # rows = true class, columns = predicted class, in class_names order
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))
    print()

# Summary table comparing all classifiers
print('Summary:')
for name, acc in results.items():
    print(f'{name}: {acc:.3f}')

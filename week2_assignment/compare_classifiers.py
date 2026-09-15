import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = np.load('features_setting_A.npy')
y = np.load('labels_setting_A.npy')
class_names = np.load('class_names.npy', allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    'Linear SVM': LinearSVC(max_iter=10000),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
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
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))
    print()

print('Summary:')
for name, acc in results.items():
    print(f'{name}: {acc:.3f}')

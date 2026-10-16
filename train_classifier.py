import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_DIR = 'dataset'
IMG_SIZE = (128, 128)


def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    features = hog(
        img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm='L2-Hys',
    )
    return features


X, y = [], []
class_names = sorted(os.listdir(DATASET_DIR))
for label_idx, class_name in enumerate(class_names):
    class_folder = os.path.join(DATASET_DIR, class_name)
    for filename in sorted(os.listdir(class_folder)):
        path = os.path.join(class_folder, filename)
        try:
            X.append(extract_hog_features(path))
            y.append(label_idx)
        except Exception as e:
            print(f'Skipping {path}: {e}')

X = np.array(X)
y = np.array(y)
print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features each')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('\nAccuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=class_names))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

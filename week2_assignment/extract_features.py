"""Week 2 Assignment - Task 2: HOG feature extraction with two parameter settings.

Extracts HOG features for every image in ../dataset twice (Setting A: fine,
Setting B: coarse), saves features/labels for each setting as .npy files,
prints the feature vector lengths, and saves a side-by-side input-vs-HOG
visualization (one image per class) for Setting A.

Adapted from extract_hog_features() in train_classifier.py from the Week 2 Lab
Tutorial (Image and Scene Recognition, Kyungdong University), which uses
skimage.feature.hog: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog

Run from inside week2_assignment/:  python extract_features.py
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

DATASET_DIR = '../dataset'   # one subfolder per class
IMG_SIZE = (128, 128)        # every image is resized so all vectors have equal length

# The two HOG parameter settings required by the assignment
SETTINGS = {
    'A': dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)),    # fine detail
    'B': dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)),  # coarse detail
}


def extract_hog_features(image_path, params):
    """Load an image as grayscale, resize it, and return its HOG feature vector."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    return hog(img, block_norm='L2-Hys', **params)


def build_dataset(params):
    """Extract HOG features for every image; the label is the class folder's index."""
    X, y = [], []
    class_names = sorted(os.listdir(DATASET_DIR))
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(DATASET_DIR, class_name)
        # sorted() keeps the sample order identical on every OS, for reproducibility
        for filename in sorted(os.listdir(class_folder)):
            path = os.path.join(class_folder, filename)
            try:
                X.append(extract_hog_features(path, params))
                y.append(label_idx)
            except Exception as e:
                # e.g. an unreadable/corrupt file - skip it rather than crash
                print(f'Skipping {path}: {e}')
    return np.array(X), np.array(y), class_names


def save_visualization(params, class_names, setting_name):
    """Save one figure: top row = input image, bottom row = HOG image, one column per class."""
    fig, axes = plt.subplots(2, len(class_names), figsize=(4 * len(class_names), 8))
    for i, class_name in enumerate(class_names):
        # Use the first image (alphabetically) of each class
        class_folder = os.path.join(DATASET_DIR, class_name)
        filename = sorted(os.listdir(class_folder))[0]
        path = os.path.join(class_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        _, hog_image = hog(img, block_norm='L2-Hys', visualize=True, **params)
        # Stretch contrast so the gradient strokes are visible
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{class_name} - input')
        axes[0, i].axis('off')
        axes[1, i].imshow(hog_image_rescaled, cmap='gray')
        axes[1, i].set_title(f'{class_name} - HOG')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f'hog_visualization_setting_{setting_name}.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    class_names = None
    for setting_name, params in SETTINGS.items():
        X, y, class_names = build_dataset(params)
        print(f'Setting {setting_name} {params}: feature vector length = {X.shape[1]}')
        # Save features and labels separately for each setting
        np.save(f'features_setting_{setting_name}.npy', X)
        np.save(f'labels_setting_{setting_name}.npy', y)
        if setting_name == 'A':  # visualization is only required for Setting A
            save_visualization(params, class_names, setting_name)

    # Saved so compare_classifiers.py can print class names in its reports
    np.save('class_names.npy', np.array(class_names))
    print('Classes:', class_names)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

DATASET_DIR = '../dataset'
IMG_SIZE = (128, 128)

SETTINGS = {
    'A': dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)),
    'B': dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)),
}


def extract_hog_features(image_path, params):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    return hog(img, block_norm='L2-Hys', **params)


def build_dataset(params):
    X, y = [], []
    class_names = sorted(os.listdir(DATASET_DIR))
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(DATASET_DIR, class_name)
        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)
            try:
                X.append(extract_hog_features(path, params))
                y.append(label_idx)
            except Exception as e:
                print(f'Skipping {path}: {e}')
    return np.array(X), np.array(y), class_names


def save_visualization(params, class_names, setting_name):
    fig, axes = plt.subplots(2, len(class_names), figsize=(4 * len(class_names), 8))
    for i, class_name in enumerate(class_names):
        class_folder = os.path.join(DATASET_DIR, class_name)
        filename = sorted(os.listdir(class_folder))[0]
        path = os.path.join(class_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        _, hog_image = hog(img, block_norm='L2-Hys', visualize=True, **params)
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
        np.save(f'features_setting_{setting_name}.npy', X)
        np.save(f'labels_setting_{setting_name}.npy', y)
        if setting_name == 'A':
            save_visualization(params, class_names, setting_name)

    np.save('class_names.npy', np.array(class_names))
    print('Classes:', class_names)

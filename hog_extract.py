"""Week 2 Lab - Part 2: extract and visualize a HOG descriptor.

Code from Step 2.1 of the Week 2 Lab Tutorial (Image and Scene Recognition,
Kyungdong University).
"""
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Load as grayscale and resize to a fixed size
img = cv2.imread('images/sample1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))

# 9 orientation bins per 8x8-pixel cell, normalized over 2x2-cell blocks
features, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
)
print('HOG feature vector shape:', features.shape)

# Stretch contrast so the gradient strokes are visible
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Input Image')
axes[1].imshow(hog_image_rescaled, cmap='gray'); axes[1].set_title('HOG Visualization')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('hog_output.png', dpi=150)
plt.show()

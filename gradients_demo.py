"""Week 2 Lab - Part 1: compute and visualize image gradients.

Code from Step 1.2 of the Week 2 Lab Tutorial (Image and Scene Recognition,
Kyungdong University).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale (HOG is computed on intensity, not color)
img = cv2.imread('images/sample1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Horizontal (gx) and vertical (gy) intensity change via the Sobel operator
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Convert to gradient magnitude (edge strength) and angle (edge direction, degrees)
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original')
axes[1].imshow(magnitude, cmap='gray'); axes[1].set_title('Gradient Magnitude')
axes[2].imshow(angle, cmap='hsv'); axes[2].set_title('Gradient Direction')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('gradients_output.png', dpi=150)
plt.show()

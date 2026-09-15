# Week 2 — HOG Features + Classical Classifiers

Image and Scene Recognition, Department of Smart Computing, Kyungdong University.

## Contents

- `gradients_demo.py`, `hog_extract.py`, `train_classifier.py`, `lab_report.md` — Week 2 Lab (gradients, HOG extraction/visualization, Linear SVM baseline on a 3-class dataset).
- `week2_assignment/` — Week 2 Coding Assignment (two HOG parameter settings, Linear SVM vs k-NN comparison, written analysis).
- `images/` — sample photos used for gradient/HOG visualization.
- `dataset/` — 3-class image dataset (`cats`, `dogs`, `birds`) used for training.

## Results Summary

The best accuracy achieved was **0.60 (60%)**, produced by a **Linear SVM** trained on HOG features extracted with **Setting A** (`orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)`, 8100-dimensional feature vectors), outperforming a k-NN (k=5) classifier on the same features, which reached 0.40 (40%) accuracy. Full breakdown, classification reports, confusion matrices, and written analysis are in [`lab_report.md`](lab_report.md) and [`week2_assignment/analysis.md`](week2_assignment/analysis.md).

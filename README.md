# Week 2 — HOG Features + Classical Classifiers

Image and Scene Recognition, Department of Smart Computing, Kyungdong University.

## Contents

- `gradients_demo.py`, `hog_extract.py`, `train_classifier.py`, `lab_report.md` — Week 2 Lab (gradients, HOG extraction/visualization, Linear SVM baseline on a 3-class dataset).
- `week2_assignment/` — Week 2 Coding Assignment (two HOG parameter settings, Linear SVM vs k-NN comparison, written analysis).
- `images/` — sample photos used for gradient/HOG visualization.
- `dataset/` — 3-class image dataset (`cats`, `dogs`, `birds`) used for training.

## Dataset Sources

- **Cats:** images collected via Google Image search.
- **Dogs:** images collected via Google Image search.
- **Birds:** images collected via Google Image search.

Images are named `<class>_NNN.jpg` (e.g. `bird_001.jpg`). The dataset is small (~4 MB), so it is committed in full.

## How to Run

```bash
pip install -r requirements.txt

# Lab (run from the repository root)
python gradients_demo.py
python hog_extract.py
python train_classifier.py

# Assignment (run from inside week2_assignment/, because the scripts use ../dataset)
cd week2_assignment
python extract_features.py
python compare_classifiers.py
```

## Results Summary

The best accuracy achieved was **0.60 (60%)**, produced by a **Linear SVM** trained on HOG features extracted with **Setting A** (`orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)`, 8100-dimensional feature vectors). For comparison, a bonus Random Forest classifier reached 0.50 (50%) and a k-NN (k=5) classifier reached 0.40 (40%) on the same features — Linear SVM was the strongest of the three. Full breakdown, classification reports, confusion matrices, and written analysis are in [`lab_report.md`](lab_report.md) and [`week2_assignment/analysis.md`](week2_assignment/analysis.md).

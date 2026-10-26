# Week 2 — HOG Features + Classical Classifiers

Image and Scene Recognition, Department of Smart Computing, Kyungdong University.

## Contents

- `gradients_demo.py`, `hog_extract.py`, `train_classifier.py`, `lab_report.md` — Week 2 Lab (gradients, HOG extraction/visualization, Linear SVM baseline on a 3-class dataset).
- `week2_assignment/` — Week 2 Coding Assignment (two HOG parameter settings, Linear SVM vs k-NN comparison, written analysis).
- `images/` — sample photos used for gradient/HOG visualization.
- `dataset/` — 3-class image dataset (`cats`, `dogs`, `birds`) used for training.
- `cnn_labs/` — Week 3 Lab (`Week3_CNN_Lab_YourID.ipynb`, CNN from scratch on CIFAR-10, 73.7% test accuracy) and Week 4 Lab (`Week4_Transfer_Lab_YourID.ipynb`, transfer learning with ResNet-18 on 5,000 images: scratch 50.4% < feature extraction 71.8% < fine-tuned 77.8% test accuracy). CIFAR-10 is downloaded automatically into `cnn_labs/data/` and is not committed.

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

On the assignment's 80/20 train/test split (64 training / 16 test images), the best accuracy achieved was **0.5625 (56.25%)**, produced by a **Linear SVM** trained on HOG features extracted with **Setting A** (`orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)`, 8100-dimensional feature vectors). For comparison, a bonus Random Forest classifier reached 0.4375 and a k-NN (k=5) classifier reached 0.375 on the same features — Linear SVM was the strongest of the three. (The lab's `train_classifier.py` uses the tutorial's 75/25 split and reaches 0.60.) Full breakdown, classification reports, confusion matrices, and written analysis are in [`lab_report.md`](lab_report.md) and [`week2_assignment/analysis.md`](week2_assignment/analysis.md).

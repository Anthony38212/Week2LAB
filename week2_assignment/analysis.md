# Week 2 Assignment — Written Analysis

## 1. Feature vector length: Setting A vs Setting B

Setting A (`pixels_per_cell=(8, 8)`) produces a feature vector of length **8100**, while Setting B (`pixels_per_cell=(16, 16)`) produces a feature vector of length **1764** — over four times shorter. This difference comes directly from how `pixels_per_cell` controls the number of cells the image is divided into. With an 8×8 pixel cell on a 128×128 image, the image is split into 16×16 = 256 cells; with a 16×16 pixel cell, the same image only splits into 8×8 = 64 cells — a quarter as many. Fewer cells means fewer local histograms, and since each block groups a fixed number of cells (2×2) and each cell contributes 9 orientation bins, the total feature count (number of block positions × cells per block × orientations) shrinks proportionally to the number of cells. In short: larger cells cover more pixels each, so you need fewer of them to tile the image, which directly shortens the descriptor.

## 2. Linear SVM vs k-NN accuracy

On the Setting A features (80 samples, 20 held out for testing), **Linear SVM reached 0.60 accuracy, while k-NN (k=5) reached only 0.40** — SVM performed noticeably better. This gap makes sense given how each algorithm makes decisions in an 8100-dimensional feature space with a small dataset. k-NN classifies purely by raw distance to the nearest training examples, but in very high-dimensional spaces like this, distances between points become less meaningful ("curse of dimensionality") — with roughly 20 training images per class, there simply aren't enough neighbors densely packed around each point for distance-based voting to be reliable. Linear SVM instead learns a weighted decision boundary across all 8100 dimensions during training, which lets it emphasize the dimensions that are actually informative for separating the classes and ignore noisy ones, making it considerably more robust on a small, high-dimensional dataset like this one.

## 3. Most confused pair of classes

Looking across both confusion matrices, the most confused pair is **birds vs. cats**, most visible in the k-NN results: 4 of 6 true birds were misclassified as cats, and 2 of 7 true cats were misclassified as birds (6 total cross-confusions, the largest off-diagonal total in either matrix). This is somewhat surprising since birds and cats don't look alike overall, but HOG only encodes edge/gradient shape, not color or texture — a bird photographed close-up or a small/curled-up bird can produce a rounded silhouette with texture-like gradient patterns (feathers) that resembles the fur-edge patterns HOG picks up on a cat. A concrete fix would be to **increase the training set size for birds and cats specifically** (more than the current ~24-27 images each) with more consistent framing (e.g., always a clear side profile of the whole animal) — this would give both HOG extraction and the classifier more consistent, representative examples to separate the two classes on, rather than relying on a handful of images where pose or crop happens to make the silhouettes converge.

## 4. Real-world industry scenario for HOG + classical classifier

A strong real-world fit for a lightweight HOG + classical classifier pipeline (over a deep CNN) is **basic security camera motion/shape filtering** — for example, an embedded camera at a warehouse entrance that needs to distinguish "a person walked into frame" from "a leaf blew across frame" or "a car passed by," running entirely on-device with no GPU and limited power. HOG + Linear SVM can run in milliseconds on a low-power embedded chip, doesn't need a large labeled dataset or cloud connectivity to retrain, and is fully interpretable — an engineer can visually inspect exactly what gradient pattern triggered a detection, which matters for debugging false alarms in a security context. A deep CNN would likely be more accurate in complex, cluttered scenes, but its computational cost and power draw make it a poor fit for a cheap, battery- or solar-powered camera that just needs a reliable, cheap, real-time shape filter before deciding whether to bother uploading footage or waking a more expensive analysis pipeline.

## Task 5 (Bonus) — Random Forest Comparison

Adding a Random Forest classifier (`n_estimators=200`) to the comparison on Setting A features gives:

| Classifier | Accuracy |
|---|---|
| Linear SVM | **0.60** |
| Random Forest | 0.50 |
| k-NN (k=5) | 0.40 |

Random Forest lands between the other two — better than k-NN but still behind Linear SVM. This **does not change the overall conclusion**: Linear SVM remains the best classifier for this dataset and feature setting. Random Forest's own confusion matrix shows an interesting failure mode: it gets **perfect precision on birds (1.00)** but very low recall (0.33) — meaning when it does predict "birds" it's always right, but it's overly conservative and misses most actual birds, instead defaulting many of them to "dogs." This tracks with how Random Forest works: it builds many decision trees, each splitting on individual HOG feature thresholds, which tends to work well when a few features are strongly discriminative but struggles to combine the more spread-out, gradual patterns across all 8100 dimensions the way SVM's single weighted hyperplane can. With more training data, Random Forest would likely close this gap somewhat, but on this small dataset Linear SVM's ability to use all dimensions jointly gives it the edge.

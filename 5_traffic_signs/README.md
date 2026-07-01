# Traffic Sign Recognition

Multi-class image classification of German traffic signs (43 classes), completed as an additional task during the Elevvo Pathways Machine Learning Internship.

## Problem

Classify traffic sign images into one of 43 categories using the GTSRB (German Traffic Sign Recognition Benchmark) dataset, handling real-world challenges like varying image sizes, lighting conditions, and class imbalance.

## Dataset

- **Source:** GTSRB (German Traffic Sign Recognition Benchmark) via Kaggle
- **Classes:** 43 traffic sign categories (speed limits, warnings, mandatory signs)
- **Format:** Images with bounding box ROI annotations (Train.csv, Test.csv)

## Approach

Three model configurations were built and compared:

**1. Custom CNN (from scratch)**
- 3 convolutional layers (32 → 64 → 64 filters) with MaxPooling
- Dense classifier with softmax output for 43 classes
- Input: 32×32 RGB images, cropped to ROI bounding box and normalized

**2. Custom CNN + Data Augmentation**
- Same architecture as above
- Added random horizontal flip, brightness, and contrast augmentation to training data
- Evaluated impact of augmentation on generalization

**3. MobileNetV2 (Transfer Learning)**
- Pre-trained MobileNetV2 backbone (frozen weights, ImageNet)
- Input resized to 224×224 to match MobileNet requirements
- Added GlobalAveragePooling + Dropout (0.2) + Dense (43 classes) head

## Key Implementation Details

- ROI cropping applied before resizing (preserving sign region, not background)
- Stratified train/validation split to preserve class distribution
- `tf.data` pipeline with `AUTOTUNE` prefetching for training efficiency
- Training/validation accuracy curves compared across all three models

## Tools & Libraries

Python, TensorFlow/Keras, OpenCV, Pandas, NumPy, Matplotlib, Scikit-learn

## Notes

This project was completed as a bonus/secondary task within the internship, under tighter time constraints than the four core projects. It demonstrates familiarity with CNN architectures, data augmentation, and transfer learning concepts using TensorFlow/Keras.

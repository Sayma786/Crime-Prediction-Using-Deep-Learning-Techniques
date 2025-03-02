# Crime Prediction Using Deep Learning Techniques

## Introduction
This project focuses on object detection for crime prevention, specifically detecting weapons (knives and guns) using deep learning algorithms. The system leverages YOLOv5, SSD, and R-CNN for real-time weapon detection in images. The goal is to enhance public security and assist law enforcement by automatically identifying potential threats.

## Features
- Weapon detection from images
- Implementation of deep learning models: YOLOv5, R-CNN, and SSD
- Comparison of different object detection models
- Performance evaluation using standard metrics

## Problem Statement
With increasing crimes involving weapons in public places, existing surveillance systems lack efficient real-time detection. This project develops a deep-learning-based object detection system to recognize and track weapons from images, helping law enforcement act swiftly.

## Objectives
- Train deep learning models on a dataset containing weapons (knives, guns, people)
- Compare accuracy and speed trade-offs between YOLOv5, SSD, and R-CNN
- Evaluate system performance using standard metrics

## Technologies Used
### Deep Learning Frameworks
- YOLOv5 for real-time object detection
- R-CNN (Region-based Convolutional Neural Network) for accurate detection
- SSD (Single Shot Detector) for balancing speed and accuracy

### Tools and Libraries
- Python (3.10.6)
- TensorFlow / PyTorch
- OpenCV for image processing
- NumPy, Pandas for data handling
- Matplotlib, Seaborn for visualization

## Dataset
- Images include guns, knives, and persons
- Manually labeled bounding boxes for object detection
- Dataset sourced from Kaggle and open repositories

## System Architecture
1. Data preprocessing: Cleaning and annotating images
2. Model training: Training YOLOv5, SSD, and R-CNN
3. Model evaluation: Comparing performance using mAP (Mean Average Precision)
4. Weapon detection: Detecting and classifying weapons in images

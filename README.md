# DeepFashionNet: CNN for Fashion MNIST Classification

This repository contains the complete implementation and supporting materials for a deep learning project using a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. This was developed for the CS5330 Computer Vision course at Northeastern University (Fall 2024).

## Project Summary

The Fashion MNIST dataset contains 70,000 grayscale images across 10 fashion categories (e.g., shirts, sneakers, dresses). Our goal was to accurately classify these items using a custom-designed deep CNN in PyTorch, supported by data augmentation and regularization.

Key features:
- CNN with 3 convolutional layers, batch normalization, ReLU, max pooling, and dropout
- Final fully connected layers with softmax activation
- Data augmentation including random rotations, zoom, and shifts
- Early stopping and evaluation via accuracy/loss curves and confusion matrix
- Achieved over 99.4% test accuracy

## Tech Stack

- Python 3.10
- PyTorch
- NumPy, Matplotlib
- Fashion MNIST Dataset

## System Requirements

- MacOS (Apple M1 Pro chip)
- Visual Studio Code (1.95.1 Universal)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib

## Running the Model

- To train and evaluate the CNN model: python fashion_CNN.py
- To visualize data augmentation techniques: python data_augmentation.py
- To generate result visualizations and confusion matrix: python poster_augmentation.py

## Results
- Validation Accuracy: Reached 99.4% with minimal gap from training accuracy
- Loss Trends: Training and validation loss both converged quickly and remained below 1.0 after early epochs
- Test Accuracy: 99.43% on unseen test set
- Confusion Matrix: High per-class accuracy, with minor misclassifications (e.g., Pullover vs. Shirt)

## References
- Fashion MNIST Dataset on Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist
- PyTorch Beginner Tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
- Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms – Xiao et al., 2017

## Acknowledgments
- Special thanks to Professor Bruce Maxwell for his guidance and instruction throughout the CS5330 Computer Vision course at Northeastern University.



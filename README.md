# DeepFashionNet: CNN for Fashion MNIST Classification

A PyTorch-based deep learning model built to classify fashion items using the Fashion MNIST dataset. This project was developed for the CS5330 Computer Vision course at Northeastern University (Fall 2024). The model achieves high accuracy with a custom CNN architecture, data augmentation, and early stopping strategies.

---

## Features

### Core Functionalities

- **Image Classification**:  
  Trained on 70,000 grayscale images across 10 fashion categories (shirts, sneakers, dresses, etc.). Uses a custom-designed CNN with 3 convolutional layers, batch normalization, ReLU, dropout, and max pooling.

- **Data Augmentation**:  
  Includes random rotation, zoom, and shift to improve generalization.

- **Model Evaluation**:  
  Supports training/validation loss tracking, accuracy plots, and confusion matrix analysis.

- **High Accuracy**:  
  Achieved **99.43%** test accuracy and **99.4%** validation accuracy.

---

## Tech Stack

- Python 3.10  
- PyTorch  
- NumPy, Matplotlib  
- Fashion MNIST Dataset  

---

## System Requirements

- macOS (tested on Apple M1 Pro chip)  
- Visual Studio Code (v1.95.1 Universal)

---

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision matplotlib

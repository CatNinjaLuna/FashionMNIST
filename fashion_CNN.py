'''
Name: Carolina Li
Date: Dec/4/2024
File: fashion_CNN.py
Purpose:
This file implements a complete pipeline for training and evaluating a Convolutional Neural Network (CNN)
for image classification using the Fashion MNIST dataset. The main objectives include:
1. Data preparation: Splitting the dataset into training, validation, and test sets, normalizing pixel values, 
and applying data augmentation to enhance the training data.
2. Model design and training: Building a multi-layer CNN model to classify images into ten fashion categories, 
compiling with appropriate loss and optimization functions, and training with early stopping to prevent overfitting.
3. Evaluation and visualization: Analyzing the model's performance using classification reports, confusion matrices, 
and accuracy/loss visualizations to interpret results and assess model behavior.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Install dataset
mnist = tf.keras.datasets.mnist
data = mnist.load_data()

# prepare training and testing labels
(train_val_data, train_val_label), (test_data, test_label) = data

# split the trainning dataset into train and validation
train_data, valid_data, train_label, valid_label = train_test_split(train_val_data, 
                                                                    train_val_label, 
                                                                    random_state=42,
                                                                    test_size=0.2, 
                                                                    stratify=train_val_label)
# print debugging statement
print(train_data.shape, train_label.shape)
print(valid_data.shape, valid_label.shape)
print(test_data.shape, test_label.shape)

# Reshape the data in the form of height, weight, channels
train_data = train_data[..., np.newaxis]
valid_data = valid_data[..., np.newaxis]
test_data = test_data[..., np.newaxis]
print('---------------------------')
print(train_data.shape, train_label.shape)
print(valid_data.shape, valid_label.shape)
print(test_data.shape, test_label.shape)

'''
(48000, 28, 28) (48000,)
(12000, 28, 28) (12000,)
(10000, 28, 28) (10000,)
---------------------------
(48000, 28, 28, 1) (48000,)
(12000, 28, 28, 1) (12000,)
(10000, 28, 28, 1) (10000,)

'''

# Normalize the data
print(train_data.min(), train_data.max())
print(valid_data.min(), valid_data.max())
print(test_data.min(), test_data.max())

train_data = train_data.astype('float32') / 255.0
valid_data = valid_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0
print('---------------------------')

print(train_data.min(), train_data.max())
print(valid_data.min(), valid_data.max())
print(test_data.min(), test_data.max())

# Data augmentation for training
# Training generator(augmentation, shuffle data for randomness)
train_datagen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1)

val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Augmenting traning data, test data and validation data without adding new data
# Feeding the data to the model in batches of 32
train = train_datagen.flow(train_data, train_label, batch_size=32, seed=42, shuffle=True)
val = val_datagen.flow(valid_data, valid_label, batch_size=32, shuffle=False)
test = test_datagen.flow(test_data, test_label, batch_size=32, shuffle=False)

# Building and training the CNN model
model = tf.keras.Sequential(
   [
      tf.keras.layers.InputLayer(shape = (28, 28, 1)), # input layers
      tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same'),   # 1st CNN layer
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.2),

      tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), padding = 'same'),   # 2nd CNN layer
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.3),

      tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), padding = 'same'),   # 3rd CNN layer
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.4),

      # flattening layer
      tf.keras.layers.Flatten(),
      # fully-connected layer
      tf.keras.layers.Dense(units = 128, activation = tf.keras.activations.relu),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.5),
      # output layers
      tf.keras.layers.Dense(units = 10, activation = tf.keras.activations.softmax)

   ]
)

# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = [tf.keras.metrics.SparseCategoricalAccuracy])

# Setting parameters for loss validation
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 5,
                                                  verbose = 1,
                                                  mode = 'min',
                                                  restore_best_weights=True
                                                    )
# Training the model
history = model.fit(train, 
                    epochs = 20,
                    validation_data = val,
                    callbacks=[early_stopping]
                    )

# Visualization of training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Visualization of training loss & validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Model evaluation on the test data
model.evaluate(test)
predictions = model.predict(test)
predictions_y = predictions.argmax(axis=1)
print(classification_report(test.y, predictions_y))

# Draw the confusion matrix
# Draw the confusion matrix
label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
fig, ax = plt.subplots(figsize=(15, 7))
disp = ConfusionMatrixDisplay(confusion_matrix(test.y, predictions_y), display_labels=label)
disp.plot(ax=ax)
ax.set_title("Confusion Matrix for Fashion MNIST Classification", pad=20)  # Adjusted padding for space
ax.set_xlabel("Predicted Category")  # Horizontal axis label
ax.set_ylabel("True Category")       # Vertical axis label
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:29:33 2024

@author: Hacker
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
#import io
#import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255
    return np.expand_dims(image_array, axis=(0, -1))

# Function to predict and sum digits
def predict_and_sum(image_paths):
    total_sum = 0
    for image_path in image_paths:
        image_array = preprocess_image(image_path)
        predictions = model.predict(image_array)
        digit = np.argmax(predictions)
        total_sum =+ digit
    return total_sum

# Example usage with a list of image paths
image_paths = ['C:/Users/Hacker/Desktop/all projects/python/img/t.png', 'C:/Users/Hacker/Desktop/all projects/python/img/s.png']  # Replace with actual image paths
print(image_paths)
total_sum = predict_and_sum(image_paths)
print("Sum of digits:", total_sum)
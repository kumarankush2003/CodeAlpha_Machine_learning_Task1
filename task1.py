# -*- coding: utf-8 -*-
"""Task1.ipynb"""

# Install necessary packages
!pip install tensorflow numpy matplotlib

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import io

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to include the channel dimension
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Function to predict if an uploaded image is handwritten
def predict_handwritten(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = image / 255.0
    image = image.reshape((1, 28, 28, 1))
    prediction = model.predict(image)
    confidence = np.max(prediction)
    return confidence

# Upload and predict an image
uploaded = files.upload()

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}"')
    img = Image.open(io.BytesIO(uploaded[fn]))
    plt.imshow(img, cmap='gray')
    plt.show()
    confidence = predict_handwritten(img)
    print(f'Confidence: {confidence:.4f}')
    if confidence > 0.5:  # Threshold for determining if it is handwritten
        print("The image is likely handwritten.")
    else:
        print("The image is likely not handwritten.")
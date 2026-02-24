# Simple Deep Learning Model using Keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (0 to 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build simple model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Convert 2D image to 1D
    layers.Dense(128, activation='relu'),   # Hidden layer
    layers.Dense(10, activation='softmax')  # Output layer (10 digits)
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
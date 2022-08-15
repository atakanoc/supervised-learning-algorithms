# ---- Exercise 5 - Fashion MNIST Neural Network ---- #

# --- Imports --- #
import tensorflow
from tensorflow import keras
from keras.datasets import fashion_mnist as mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# --- Part 1 - Prepare dataset --- #
# -- Load dataset.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# -- Flatten X.  TODO: Remove if it works without flattening.
#X_train = X_train.reshape((-1, 784))
#X_test = X_test.reshape((-1, 784))

# --- Part 2 - Prepare Neural Network --- #
# -- Build dimensions.
clf = Sequential([
    Dense(64, activation='sigmoid', input_shape=(28, 28)),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='sigmoid')
])

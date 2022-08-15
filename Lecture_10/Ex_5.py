# ---- Exercise 5 - Fashion MNIST Neural Network ---- #

# --- Imports --- #
from tensorflow import keras
from keras.datasets import fashion_mnist as mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

# --- Part 1 - Load & prepare dataset --- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# -- Flatten X.
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

# --- Part 2 - Prepare Neural Network --- #
# -- Build dimensions.
clf = Sequential([
    Dense(64, activation='sigmoid', input_shape=(784,)),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='sigmoid')
])

# -- Compile Neural Network.
clf.compile(
    optimizer='SGD',
    loss='mse',
    metrics=['accuracy']
)

# -- Train Neural Network.
load_weights = False

if load_weights:
    clf.load_weights('clf/fashion_mnist')
else:
    # -- Train.
    clf.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
    )

    # -- Save to disk.
    clf.save_weights('clf/fashion_mnist')
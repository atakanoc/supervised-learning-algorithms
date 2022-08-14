from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# --- Load dataset.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# --- Normalize X.
X_train = (X_train / 255) - 0.5
X_test = (X_test / 255) - 0.5

# --- Flatten X.
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

# ---
pass
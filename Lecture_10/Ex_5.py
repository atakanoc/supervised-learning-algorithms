# ---- Exercise 5 - Fashion MNIST Neural Network ---- #

# --- Imports --- #
from sklearn.model_selection import GridSearchCV
from keras.datasets import fashion_mnist as mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
import numpy as np

# --- Functions --- #
def create_nn(
    activation='sigmoid', optimizer='SGD', hidden_nodes=64
):
    # -- Build dimensions.
    clf = Sequential([
        Dense(hidden_nodes, activation=activation, input_shape=(784,)),
        Dense(hidden_nodes, activation=activation),
        Dense(10, activation=activation)
    ])

    # -- Compile Neural Network.
    clf.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['accuracy']
    )

# --- Part 1 - Load & prepare dataset --- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# -- Flatten X.
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

# --- Part 2 - Grid search for best hyper-parameters --- #
params = {
    'activation': ['sigmoid', 'relu', 'softmax'],
    'optimizer': ['SGD', 'Adam'],
    'hidden_nodes': [16, 32, 64]
}

clf = KerasClassifier(model=create_nn, epochs=10, batch_size=64)  # Wrap Keras model for sci-kitlearn
grid = GridSearchCV(clf, params, n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

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
import a3


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

    return clf

# --- Part 1 - Load & prepare dataset --- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
X_valid, y_valid = X_train[50000:], y_train[50000:]
X_train, y_train = X_train[:50000], y_train[:50000]

# -- Flatten X.
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))
X_valid = X_valid.reshape((-1, 784))

# --- Part 2 - Grid search for best hyper-parameters --- #
# TODO: Make grid-search optional with loading instead. Model should not be trained.
# TODO: Training should also be optional allowing you to load weights instead.

search = True
if search:  # TODO: Save pickle

    # -- Grid of hyper-parameters.
    params = {
        'activation': ['sigmoid', 'relu'],
        'optimizer': ['SGD', 'Adam'],
        'hidden_nodes': [128, 256, 512]
    }

    # -- Grid search.
    best_params = []
    best_accuracy = 0
    for a in params['activation']:
        for o in params['optimizer']:
            for n in params['hidden_nodes']:
                print(f"Model = {a}, {o}, {n}")
                temp_clf = create_nn(a, o, n)
                temp_clf.fit(X_train, y_train, batch_size=64, epochs=1)  # Low epoch for quick testing.
                accuracy = temp_clf.evaluate(X_valid, y_valid)[1]

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = [a, o, n]

                print("\n\n")

    a3.save_pickle("pickles/best_params.pickle", best_params)
else:  # TODO: Load pickle
    pass
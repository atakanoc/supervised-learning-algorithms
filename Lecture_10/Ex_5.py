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

    return clf

# --- Part 1 - Load & prepare dataset --- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# -- Flatten X.
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

# --- Part 2 - Grid search for best hyper-parameters --- #
params = {
    'activation': ['sigmoid', 'relu'],
    'optimizer': ['SGD', 'Adam'],
    'hidden_nodes': [128, 256, 512]
}

# TODO: Make grid-search optional with loading instead.
for a in params['activation']:
    for o in params['optimizer']:
        for n in params['hidden_nodes']:
            clf = create_nn(a, o, n)
            clf.fit(X_train, y_train, batch_size=64, epochs=10)
            print(f"Model = {a}, {o}, {n}")
            clf.evaluate(X_test, y_test)
            print("\n\n")


# clf = KerasClassifier(model=create_nn, epochs=10, batch_size=64,
#                       activation=None, hidden_nodes=None)  # Wrap Keras model for sci-kitlearn
# grid = GridSearchCV(clf, params, n_jobs=-1, verbose=3)
# grid.fit(X_train, y_train)
# clf = grid.best_estimator_
# print(clf.get_params())

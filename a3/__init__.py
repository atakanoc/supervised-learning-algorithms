import numpy as np
import os
import gzip
from urllib.request import urlretrieve
import pickle


# Credits = https://mattpetersen.github.io/load-mnist-with-numpy
def mnist(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if nonexistant. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'mnist')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels

train_X, train_y, test_X, test_y = mnist()


def one_vs_all_parser(y: np.ndarray):
    if y.ndim != 1:
        raise ValueError(f"y must be 1 dimensional, instead is {y.ndim}.")

    # Find all unique responses
    nums = np.unique(y)

    # Create a y for each unique response
    y_e = {}
    for num in nums:
        col = (y == num).astype(int)
        y_e[num] = col

    return y_e


class OneVsAllClassifier:
    def __init__(self, clfs):
        self.clfs = clfs

    def predict(self, X):
        if len(X.shape) != 2:
            raise ValueError("X is not 2 dimensional")

        y = np.zeros((X.shape[0], len(self.clfs)))

        # I DON'T SEE A WAY TO VECTORIZE THIS :(
        # Actually, there can be 1 for loop instead of 2. Yay!
        #for row_idx in range(X.shape[0]):
        #   for clf_idx in range(len(self.clfs)):
        #      y[row_idx, clf_idx] = self.clfs[clf_idx].predict(X[row_idx].reshape(1, -1))

        # Attempt 2
        for i in range(len(self.clfs)):
            y[:, i] = self.clfs[i].predict(X)

        return y


def ensemble_predict_binary(clfs, X):
    y = np.zeros(X.shape[0])
    for clf in clfs:
        y += clf.predict_proba(X)[:, 1]
    y /= len(clfs)
    return np.round(y)


def save_pickle(file_path, data):
    file = open(file_path, 'ab')
    pickle.dump(data, file)
    file.close()


def load_pickle(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data
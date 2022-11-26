import numpy as np


def make_rowwise(foo):
    """
    Decorator, makes a rowwise metric from a metric comparing two vectors.
    """
    def rowwise_foo(X, Y):
        result = []
        for i, (x, y) in enumerate(zip(X, Y)):
            result.append(foo(x, y))
        return np.array(result)
    return rowwise_foo


def rowwise_cosine(y_true, y_pred):
    """
    https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
    """
    return 1 - np.einsum('ij,ij->i', y_true, y_pred) / (
                np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
        )
  
def rowwise_mse(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean(1)


def rowwise_rmse(y_true, y_pred):
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean(1))
  

def rowwise_se(y_true, y_pred):
    return np.sum(np.square(np.subtract(y_true, y_pred)), axis=1)
  

def rowwise_euclid(y_true, y_pred):
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).sum(1))


def make_dist(x):
    return x / np.sum(x)


@make_rowwise
def rowwise_kl_divergence(x, y):
    x, y = make_dist(x), make_dist(y)   # normalize to a distribution
    return np.sum(np.where(y != 0, x * np.log(x / y), 0))
import numpy as np

def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

def random_normal(rows, cols):
    return np.random.randn(rows, cols)

def ones(rows, cols):
    return np.ones((rows, cols))

def zeros(rows, cols):
    return np.zeros((rows, cols))
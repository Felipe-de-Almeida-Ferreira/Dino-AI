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

'''def mutate(weights, biases):
    new_weights = weights + np.random.uniform(-1, 1)
    new_biases = biases + np.random.uniform(-1, 1)
    return new_weights, new_biases
    '''

def mutate_weights(weights):
    new_weights = weights + np.random.uniform(-100, 100)
    print(new_weights)
    return new_weights

def mutate_biases(biases):
    new_biases = biases + np.random.uniform(-10, 10)
    return new_biases
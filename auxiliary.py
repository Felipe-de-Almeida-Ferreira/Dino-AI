import numpy as np
import sys

def relu(x, derivative=False):
    if derivative:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    x = np.clip(x, -500, 500)
    if derivative:
        y = sigmoid(x)
        return y*(1 - y)
    return 1.0/(1.0 + np.exp(-x))

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

def mutate_weights(weights, mutation_rate=0.1):
    mutation = np.random.uniform(-mutation_rate, mutation_rate, weights.shape)
    return weights + mutation

def mutate_biases(biases, mutation_rate=0.1):
    mutation = np.random.uniform(-mutation_rate, mutation_rate, biases.shape)
    return biases + mutation
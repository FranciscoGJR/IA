import numpy as np


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def leaky_relu(x):
    return max(0.01 * x, x)


def leaky_relu_derivative(x):
    return 1 if x > 0 else 0.01


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


activation_functions = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

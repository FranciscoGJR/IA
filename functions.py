import numpy as np
class Functions:
    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def relu_derivative(x):
        return 1 if x > 0 else 0

    @staticmethod
    def leaky_relu(x):
        return max(0.01 * x, x)

    @staticmethod
    def leaky_relu_derivative(x):
        return 1 if x > 0 else 0.01

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# GUILHERME DIAS JIMENES - 11911021
# IGOR AUGUSTO DOS SANTOS - 11796851
# LAURA PAIVA DE SIQUEIRA – 1120751

from math import exp
import numpy as np

#Reduz o problema de desaparecimento do gradiente o que viabiliza modelos mais profundos. Tem calculo rapido, o que acelera o treinamento, mas sofre com neuronios mortos.
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

#Resolve o problema de neuronios mortos da RELU, mas o parametro de vazamento pode precisar de ajustes
def leaky_relu(x):
    return max(0.01 * x, x)

def leaky_relu_derivative(x):
    return 1 if x > 0 else 0.01

#Tem convergência mais rápida durante o treinamento em comparação com a sigmoid, mas sofre com problemas do desaparecimento do gradiente.
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

#Boa opção para problemas de saida binaria, mas também é afetada pelo desaparecimento do gradiente
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
#Tem melhor otimização, mas tem computabilidade custosa, precisa de uma constante beta, para este teste, definimos como 1
def swish(x):
    return x * sigmoid(1*x)

def swish_derivative(x):
    sigmoid_beta_x = sigmoid(1*x)
    return sigmoid_beta_x + 1 * x * sigmoid_beta_x * (1 - sigmoid_beta_x)

# Normaliza as saídas de cada camada, o que facilita manter uma média e variância próximas de zero e um. Porem seu alpha e lambda são sensiveis a valores que podem prejudicar a normalização.
def selu(x, lambda_constant=1.0507, alpha_constant=1.67326):
    return lambda_constant * np.where(x > 0, x, alpha_constant * (np.exp(x) - 1))

def selu_derivative(x, lambda_constant=1.0507, alpha_constant=1.67326):
    return lambda_constant * np.where(x > 0, 1, alpha_constant * np.exp(x))

# Melhor computacionalmente, porem pode ter desempenho menor que a swish tradicional
def hard_swish(x):
    return x * np.minimum(np.maximum(x + 3, 0), 6) / 6

def hard_swish_derivative(x):
    return np.where(x < -3, 0, np.where(x <= 3, (2 * x + 3) / 6, 1))

#Tem uma melhor generalização e redução do desaparecimento do gradiente, porém é mais custosa computacionalmente devido às operações de log e exponencial.
# def mish(x):
#     return x * np.tanh(np.log1p(np.exp(x)))
# def mish_derivative(x):
#     omega = np.exp(x) + 2 + np.exp(-x)
#     delta = 4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)
#     return np.exp(x) * delta / (omega ** 2)

# Validação rápida e com saídas limitadas que mitiga a explosão do gradiente; Pode perder informações por ser uma função saturada
def hard_tanh(x):
    return np.clip(x, -1, 1)

def hard_tanh_derivative(x):
    return np.where((x > -1) & (x < 1), 1, 0)


activation_functions = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'swish': (swish, swish_derivative),
    'selu': (selu, selu_derivative),
    'hard_swish': (hard_swish, hard_swish_derivative),
    'hard_tanh': (hard_tanh, hard_swish_derivative)
}

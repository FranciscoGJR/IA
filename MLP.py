# EP ACH2016 - "IA" | Turma 04
# BRUNO LEITE DE ANDRADE - 11369642
# IGOR AUGUSTO DOS SANTOS - 11796851
# + ...

import numpy as np
import numpy.typing as npt
from math import inf
from typing import List
import pandas as pd


NO_NODES_INPUT = 120
NO_NODES_HIDDEN = 40
NO_NODES_OUTPUT = 26
MAX_EPOCH = 25
TOLERANCE = 0.01
LEARNING_RATE = 0.5

# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
def ACTIVATE(x):
	return 1 / (1 + np.exp(-x))

def DERIVATIVE(x):
	act = ACTIVATE(x)
	return act * (1 - act)

RELU = lambda x: np.maximum(0, x)
DERIVATIVE_RELU = lambda x: np.where(x > 0, 1, 0)


class Model:

	def __init__(self, wl=None, activation_function=ACTIVATE, derivative_function=DERIVATIVE):
		self.activation_function = activation_function
		self.derivative_function = derivative_function

		self.nodes = [
			np.zeros(NO_NODES_INPUT, np.double),
			np.zeros(NO_NODES_HIDDEN, np.double),
			np.zeros(NO_NODES_OUTPUT, np.double)
		]

		if wl is None:
			self.weights = [
				np.random.randn(NO_NODES_HIDDEN, NO_NODES_INPUT + 1) * 0.01,
				np.random.randn(NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1) * 0.01
			]
		else:
			self.weights = [
				np.full((NO_NODES_HIDDEN, NO_NODES_INPUT + 1), wl, np.double),
				np.full((NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1), wl, np.double)
			]

		self.delta = [
			np.full((NO_NODES_HIDDEN, NO_NODES_INPUT + 1), 0, np.double),
			np.full((NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1), 0, np.double)
		]

	def classification_accuracy(self, test_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]]):
		correct = 0
		for i, entry in enumerate(test_set):
			output = self.feedforward(entry)
			if np.argmax(output) == np.argmax(target[i]):
				correct += 1
		return correct / len(test_set)

	def feedforward(self, input_data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		vetorized_activate = np.vectorize(ACTIVATE)
		self.nodes[0] = input_data
		for i in range(1, len(self.nodes)):
			hidden_out = np.dot(self.weights[i-1], np.append(self.nodes[i-1], 1))
			self.nodes[i] = vetorized_activate(hidden_out)

		return self.nodes[-1]

	def apply_changes(self):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + self.delta[i]

	def backpropagation(self, error: npt.NDArray[np.double]):
		error_info = []
		for i, neuron in enumerate(self.nodes[-1]):
			neuron_input = np.dot(self.weights[-1][i], np.append(self.nodes[-2], 1))
			error_correction = error[i] * self.derivative_function(neuron_input)
			error_info.append(error_correction)
			self.delta[-1][i] = LEARNING_RATE * error_correction * np.append(self.nodes[-2], 1)

		for i, neuron in enumerate(self.nodes[-2]):
			soma = 0
			for ie, er in enumerate(error_info):
				soma += er * self.weights[-1][ie][i]
			neuron_input = np.dot(self.weights[0][i], np.append(self.nodes[-3], 1))
			error_correction = soma * self.derivative_function(neuron_input)
			self.delta[-2][i] = LEARNING_RATE * error_correction * np.append(self.nodes[-3], 1)

	def train(self, training_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]]):
		epoch = 0
		avg_error = inf
		while epoch < MAX_EPOCH or avg_error < TOLERANCE:
			epoch += 1
			for i, entry in enumerate(training_set):
				output = self.feedforward(entry)
				error = target[i] - output
				self.backpropagation(error)
				self.apply_changes()
			print(f"Época: {epoch}/{MAX_EPOCH}")

# main
if __name__ == '__main__':
	model = Model()

	input_data = np.load('./caracteres/X.npy')

	# remove dados de teste
	input_data = input_data[:-130]

	target_data = np.load('./caracteres/Y_classe.npy')

	print("taxa de acerto no conjundo de treinamento antes do treino")
	taxa = model.classification_accuracy(input_data, target_data) * 100
	print(f"{taxa:.3f}%")

	model.train(input_data, target_data)

	print("Taxa de acerto no conjunto de treinamento depois do treino")
	taxa = model.classification_accuracy(input_data, target_data) * 100
	print(f"{taxa:.3f}%")

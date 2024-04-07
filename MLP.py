# EP ACH2016 - "IA" | Turma 04
# BRUNO LEITE DE ANDRADE - 11369642
# IGOR AUGUSTO DOS SANTOS - 11796851
# + ...

import numpy as np
import numpy.typing as npt
from math import inf
from typing import List


NO_NODES_INPUT = 2
NO_NODES_HIDDEN = 2
NO_NODES_OUTPUT = 2
MAX_EPOCH = 10E3
TOLERANCE = 0.01

# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))


class Model:

	def __init__(self, wl1, wl2):

		self.nodes = [
			np.zeros(NO_NODES_INPUT, np.double),
			np.zeros(NO_NODES_HIDDEN, np.double),
			np.zeros(NO_NODES_OUTPUT, np.double)
		]
		
		self.weights = [
			np.full((NO_NODES_HIDDEN, NO_NODES_INPUT + 1), wl1, np.double),
			np.full((NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1), wl2, np.double)
		]

	def feedforward(self, input_data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		vetorized_activate = np.vectorize(ACTIVATE)
		self.nodes[0] = input_data
		for i in range(1, len(self.nodes)):
			hidden_out = np.dot(self.weights[i-1], np.append(self.nodes[i-1], 1))
			self.nodes[i] = vetorized_activate(hidden_out)
			
		return self.nodes[-1]

	def train(self, training_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]]):
		epoch = 0
		avg_error = inf
		while epoch < MAX_EPOCH or avg_error < TOLERANCE:
			epoch += 1
			for i, entry in enumerate(training_set):
				output = self.feedforward(entry)
				error = target[i] - output
				print(error)

			# TODO: Backpropagation
			break

# main
if __name__ == '__main__':
	model = Model(0.5, 0.5)
	input_test = [1, 2]

	print(f"saida modelo: {model.feedforward(input_test)}")
	print("erro modelo:")
	model.train([input_test], [2])

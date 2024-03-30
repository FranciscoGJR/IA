# EP ACH2016 - "IA" | Turma 04
# BRUNO LEITE DE ANDRADE - 11369642
# + ...

import numpy as np
import numpy.typing as np_type
from math import inf
from typing import List


NO_NODES_INPUT = 64
NO_NODES_HIDDEN = 42
NO_NODES_OUTPUT = 27
MAX_EPOCH = 10E3
TOLERANCE = 0.01

# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))


class Model():

	def __init__(wl0, wl1, wl2):

		nodes = 
		[
			np.full((NO_NODES_INPUT, 1), 0, np.double),
			np.full((NO_NODES_HIDDEN, 1), 0, np.double),
			np.full((NO_NODES_OUTPUT, 1), 0, np.double)
		]

		weights = 
		[
			np.full((NO_NODES_INPUT + 1, 1), wl0, np.double),
			np.full((NO_NODES_HIDDEN, NO_NODES_INPUT + 1), wl1, np.double),
			np.full((NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1), wl2, np.double)
		]
		
	def aggregate(input_layer: npt.NDArray[np.double], layer: int, node: int) -> np.double:
		return np.dot(self.weights[layer][node], input_layer) 

	def activate(previous_layer: npt.NDArray[np.double], layer: int, node: int) -> None:
		nodes[layer][node] = ACTIVATE(self.aggregate(previous_layer, layer, node))
	
	def train(training_set : List[npt.NDArray[np.double]) -> None:
		
		epoch = 0
		avg_error = inf
		
		while epoch < MAX_EPOCH or avg_error < TOLERANCE:
		
			for entry in training_set:
				
				# Talvez seja possível otimizar esse loop com alguma função do Numpy
				layers = list(entry).extend(self.nodes[:-1])
				for layer_index, layer in enumerate(layers):		
					self.nodes[layer_index] = np.asarray([self.aggregate(layer, layer_index, n) for n in self.nodes[layer_index])
					
				# TODO: Testa se `self.nodes[2] -> tem saídas esperadas. Depois calcula o erro e faz o backfeed


# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# IGOR AUGUSTO DOS SANTOS - 11796851
# 
# + ...

import numpy as np
import numpy.typing as npt
from math import inf
from typing import List
import pandas as pd
from random import shuffle

# ARQUITETURA:

# TODO: Avaliar possibilidade de mover definição de arquitetura para dentro da classe 'Model', a fim de possibilitar melhor automação de testes parâmetricos

NO_NODES_INPUT = 120
NO_NODES_HIDDEN = 42
NO_NODES_OUTPUT = 26
MAX_EPOCH = 150
VALIDATION_INTERVAL = 5
INERTIA = 6
ERROR_TOLERANCE = 0.15

# TODO: implementar função de alfa
LEARNING_RATE = lambda x: 0.1

# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))
ACTIVATE_DERIVATIVE = lambda x: ACTIVATE(x) * (1 - ACTIVATE(x))

# Alternativamente, pode se usar:
#ACTIVATE = lambda x: np.maximum(0, x)
#ACTIVATE_DERIVATIVE = lambda x: np.where(x > 0, 1, 0)


class Model:

	def __init__(self, w: List[npt.NDArray[np.double]] = None):
		
		# Espaço de memória dos neurônios
		self.nodes = [
			np.zeros(NO_NODES_INPUT, np.double),
			np.zeros(NO_NODES_HIDDEN, np.double),
			np.zeros(NO_NODES_OUTPUT, np.double)
		]
		
		# Inicialização dos pesos nas camadas entre os neurônios
		# Caso a função de inicialização não receba o argumento "w", inicializa o modelo com pesos aleatórios
		if w is None:
			self.weights = \
			[
				np.random.randn(NO_NODES_HIDDEN, NO_NODES_INPUT + 1) * 0.01,
				np.random.randn(NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1) * 0.01
			]
		else:
			self.weights = \
			[
				np.full((NO_NODES_HIDDEN, NO_NODES_INPUT + 1), w[0], np.double),
				np.full((NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1), w[1], np.double)
			]

	def feed_forward(self, data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		
		self.nodes[0] = data
		
		for i in range(1, len(self.nodes)):
		
			# Calcula o produto escalar entre todos neurônios de chegada e seus respectivos pesos (função de agregação)
			layer_output = np.dot(self.weights[i-1], np.append(self.nodes[i-1], 1))
			
			# aplica a função de ativação na camada de neurônios
			self.nodes[i] = np.vectorize(ACTIVATE)(layer_output)
		
		return self.nodes[-1]

	def __backpropagation(self, error: npt.NDArray[np.double], delta, epoch) -> None:
		
		error_info = []
		
		for i, neuron in enumerate(self.nodes[-1]):
			neuron_input = np.dot(self.weights[-1][i], np.append(self.nodes[-2], 1))
			error_correction = error[i] * ACTIVATE_DERIVATIVE(neuron_input)
			error_info.append(error_correction)
			delta[-1][i] = LEARNING_RATE(1) * error_correction * np.append(self.nodes[-2], 1)

		for i, neuron in enumerate(self.nodes[-2]):
			soma = 0
			
			for ie, er in enumerate(error_info):
				soma += er * self.weights[-1][ie][i]
			
			neuron_input = np.dot(self.weights[0][i], np.append(self.nodes[-3], 1))
			error_correction = soma * ACTIVATE_DERIVATIVE(neuron_input)
			delta[-2][i] = LEARNING_RATE(epoch) * error_correction * np.append(self.nodes[-3], 1)
	
	def classification_accuracy(self, test_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]]):
		correct = 0
		for i, entry in enumerate(test_set):
			output = self.feed_forward(entry)
			if np.argmax(output) == np.argmax(target[i]):
				correct += 1
		return correct / len(test_set)
		
	def average_error(self, data, data_target) -> float:
		 return np.average(np.absolute(data_target - self.feed_forward(data)))
	
	def train(self, input_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]], training_validation_proportion: float = 2/3) -> List[npt.NDArray[np.double]]:
		
		# ------------------------------------------------- #
		# --- Definição de funções ajudantes de `train` --- #
		
		# Funções definidas dentro da função "train" a fim de garantir o escopo de acesso apenas à função train
		
		def apply_changes(delta) -> None:
			for i in range(len(self.weights)):
				self.weights[i] = self.weights[i] + delta[i]
		
		def check_to_calculate_accuracy(momentum: int, epoch: int, training_validation_proportion: float):
			if training_validation_proportion == 1:
				return False
			if momentum == INERTIA:
				return (epoch + 1) % VALIDATION_INTERVAL == 0
			else:
				return True
		
		# ------------------------------------------------- #
		# -------- Definição de parâmetros gerais --------- #
		
		# O argumento training_validation_proportion tem por objetivo dividir o conjunto inicial de input em dois subconjuntos:
		#	* Conjunto de treinamento 'training_set' (que efetivamente alimenta o cálculo dos erros)
		#	* Conjunto de validação 'validation_set' (que é usado para calcular a acurácia do modelo com dados que não alimentem o treinamento)
		
		# levanta exceção em caso de 'training_validation_proportion' que impossibilite uma divisão do conjunto inicial 'input_set' 
		if training_validation_proportion > 1 or training_validation_proportion < 0.5 :
			raise ValueError(f"Parameter 'training_validation_proportion' should be betwen 1 and 0.5, got {training_validation_proportion}")
		
		training_slice_index = int(len(input_set)*training_validation_proportion)
		shuffle_index_range = list(range(len(input_set))) 
		shuffle(shuffle_index_range)
		input_set = input_set[shuffle_index_range]
		target = target[shuffle_index_range]
		training_set = input_set[:training_slice_index]
		validation_set = input_set[training_slice_index:]
		
		# Variável "momentum" é usada para realizar a validação 'INERTIA' número de vezes, caso a validação tenha superado a validação anterior
		momentum = INERTIA
		
		# Salva snapshots do modelo a cada nova validação utilizando o 'validation_set' 
		accuracy_timeline = [((self.classification_accuracy(validation_set, target[training_slice_index:]), -1, self.weights))]
		
		# Salva os valores de correção dos pesos
		delta = [
			np.full((NO_NODES_HIDDEN, NO_NODES_INPUT + 1), 0, np.double),
			np.full((NO_NODES_OUTPUT, NO_NODES_HIDDEN + 1), 0, np.double)
		]
		
		# ------------------------------------------------- #
		# -------------- Loop de Treinamento -------------- #
		
		# TODO: implementar funcionalidade de 'verbose_printing', para possibilidade de impressão de parâmetros do modelo a cada época
		
		for epoch in range(MAX_EPOCH):
			if momentum == 0:
				break
			
			# Para cada índice, e dado do conjunto de treinamento: 
			for i, entry in enumerate(training_set):
				error = target[i] - self.feed_forward(entry)
				self.__backpropagation(error, delta, epoch) # TODO avaliar possibilidade de transformar 'delta' em valor de retorno de '__backpropagation'
				apply_changes(delta)
			
			# Checa se irá calcular a acurácia do modelo para a época atual, utilizando o 'validation_set'
			if check_to_calculate_accuracy(momentum, epoch, training_validation_proportion): 
				current_accuracy = self.classification_accuracy(validation_set, target[training_slice_index:]), epoch, self.weights 
				accuracy_timeline.append(current_accuracy)
				if accuracy_timeline[-1][0] > 1 - ERROR_TOLERANCE or momentum < INERTIA:
					if accuracy_timeline[-1][0] > accuracy_timeline[-2][0]:
						momentum = INERTIA - 1
					else:
						momentum -= 1
		
		return accuracy_timeline

	def __repr__(self):
		return (f'Model({self.weights})')
		
	def __str__(self):
		return self.weights


# main
if __name__ == '__main__':
		
	model = Model()

	input_data = np.load('./test/X.npy')
	target_data = np.load('./test/Y_classe.npy')
	# remove dados de teste
	input_data = input_data[:-130]
	target_data = target_data[:-130]
	
	taxa = model.classification_accuracy(input_data, target_data) * 100
	print("taxa de acerto no conjunto de treinamento antes do treino: " + f"{taxa:.3f}%")
	
	acc = []		
	
	try:	
		acc = model.train(input_data, target_data)

	finally:
		print(*[f'{(100*m[0])}% -> epoch: {m[1]}' for m in acc], sep='\n')



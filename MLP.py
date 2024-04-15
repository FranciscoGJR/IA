# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# IGOR AUGUSTO DOS SANTOS - 11796851
# + ...

import numpy as np
import numpy.typing as npt
import pandas as pd
from constants import *
from math import inf
from random import shuffle
from typing import List


class Model:

	NO_NODES_INPUT = 120
	NO_NODES_HIDDEN = 42
	NO_NODES_OUTPUT = 26
	MAX_EPOCH = 100
	VALIDATION_INTERVAL = 5
	INERTIA = 6
	ERROR_TOLERANCE = 0.15

	# TODO: implementar função de alfa
	LEARNING_RATE = lambda x: 0.1

	# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
	ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))
	ACTIVATE_DERIVATIVE = lambda x: Model.ACTIVATE(x) * (1 - Model.ACTIVATE(x))

	# Alternativamente, pode se usar:
	#ACTIVATE = lambda x: np.maximum(0, x)
	#ACTIVATE_DERIVATIVE = lambda x: np.where(x > 0, 1, 0)

	def __init__(self, w: List[npt.NDArray[np.double]] = None):
		
		# Espaço de memória dos neurônios
		self.nodes = [
			np.zeros(Model.NO_NODES_INPUT, np.double),
			np.zeros(Model.NO_NODES_HIDDEN, np.double),
			np.zeros(Model.NO_NODES_OUTPUT, np.double)
		]
		
		# Inicialização dos pesos nas camadas entre os neurônios
		# Caso a função de inicialização não receba o argumento "w", inicializa o modelo com pesos aleatórios
		if w is None:
			self.weights = \
			[
				np.random.randn(Model.NO_NODES_HIDDEN, Model.NO_NODES_INPUT + 1) * 0.01,
				np.random.randn(Model.NO_NODES_OUTPUT, Model.NO_NODES_HIDDEN + 1) * 0.01
			]
		else:
			self.weights = \
			[
				np.full((Model.NO_NODES_HIDDEN, Model.NO_NODES_INPUT + 1), w[0], np.double),
				np.full((Model.NO_NODES_OUTPUT, Model.NO_NODES_HIDDEN + 1), w[1], np.double)
			]

	def feed_forward(self, data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:

		self.nodes[INPUT_LAYER] = data

		for current_layer in range(HIDDEN_LAYER, len(self.nodes)):
			previous_layer = current_layer - ONE

			# Calcula o produto escalar entre todos neurônios de chegada e seus respectivos pesos (função de agregação)
			layer_output = np.dot(self.weights[previous_layer], np.append(self.nodes[previous_layer], BIAS))

			# aplica a função de ativação na camada de neurônios
			self.nodes[current_layer] = np.vectorize(Model.ACTIVATE)(layer_output)

		return self.nodes[OUTPUT_LAYER]

	# TODO: Considerar retornar "delta", ao invés de receber como parâmetro
	def __backpropagation(self, error: npt.NDArray[np.double], delta, epoch) -> None:

		error_info = []

		for current_neuron, neuron in enumerate(self.nodes[OUTPUT_LAYER]):
			neuron_input = np.dot(self.weights[LAST][current_neuron], np.append(self.nodes[HIDDEN_LAYER], BIAS))
			error_correction = error[current_neuron] * Model.ACTIVATE_DERIVATIVE(neuron_input)
			error_info.append(error_correction)
			delta[LAST][current_neuron] = Model.LEARNING_RATE(ONE) * error_correction * np.append(self.nodes[HIDDEN_LAYER], BIAS)

		for current_neuron, neuron in enumerate(self.nodes[HIDDEN_LAYER]):
			sum = ZERO

			for ie, er in enumerate(error_info):
				sum += er * self.weights[LAST][ie][current_neuron]

			neuron_input = np.dot(self.weights[FIRST][current_neuron], np.append(self.nodes[INPUT_LAYER], BIAS))
			error_correction = sum * Model.ACTIVATE_DERIVATIVE(neuron_input)
			delta[FIRST][current_neuron] = Model.LEARNING_RATE(epoch) * error_correction * np.append(self.nodes[INPUT_LAYER], BIAS)


	def classification_accuracy(self, test_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]]):
		correct = ZERO
		for index, entry in enumerate(test_set):
			output = self.feed_forward(entry)
			if np.argmax(output) == np.argmax(target[index]):
				correct += ONE
		return correct / len(test_set)


	def average_error(self, data, data_target) -> float:
		 return np.average(np.absolute(data_target - self.feed_forward(data)))
	
	# TODO: implementar funcionalidade de 'verbose_printing', para possibilidade de impressão de parâmetros do modelo a cada época
	def train(
		self,
		training_set: List[npt.NDArray[np.double]],
		target_set: List[npt.NDArray[np.double]],
		validation_set: List[npt.NDArray[np.double]] = [],
		validation_target_set: List[npt.NDArray[np.double]] = []
		) -> List[npt.NDArray[np.double]]:
		
		# levanta exceção em caso de inconsistência nos dados de entrada 
		if len(training_set) != len(target_set) or len(validation_set) != len(validation_target_set):
			raise ValueError(f"Arguments of train function don't match length requirements\n" + \
				"'training_set' and 'target_set' lenghts should be equal, got{len(training_set), len(target_set)}.\n" +\
				"'validation_set' and 'validation_target_set' lenghts should be equal, got{len(validation_set), len(validation_target_set)}.\n")
		
		# ------------------------------------------------- #
		# --- Definição de funções ajudantes de `train` --- #
		
		# Funções definidas dentro da função "train" a fim de garantir o escopo de acesso apenas à função train
		
		def apply_changes(delta) -> None:
			for index in range(len(self.weights)):
				self.weights[index] = self.weights[index] + delta[index]
		
		def check_to_calculate_accuracy(momentum: int, epoch: int, validation_set_len: int) -> bool:
			if validation_set_len == ZERO:
				return False
			if momentum == Model.INERTIA:
				return (epoch + ONE) % Model.VALIDATION_INTERVAL == ZERO
			else:
				return True
		
		# ------------------------------------------------- #
		# -------- Definição de parâmetros gerais --------- #		
				
		# Variável "momentum" é usada para realizar a validação 'INERTIA' número de vezes, caso a validação tenha superado a validação anterior
		momentum = Model.INERTIA
		
		# Salva snapshots do modelo a cada nova validação utilizando o 'validation_set' 
		accuracy_timeline = [((self.classification_accuracy(validation_set, validation_target_set), -1, self.weights))]
		
		# Salva os valores de correção dos pesos
		delta = [
			np.full((Model.NO_NODES_HIDDEN, Model.NO_NODES_INPUT + ONE), ZERO, np.double),
			np.full((Model.NO_NODES_OUTPUT, Model.NO_NODES_HIDDEN + ONE), ZERO, np.double)
		]
		
		# ------------------------------------------------- #
		# -------------- Loop de Treinamento -------------- #
		
		for epoch in range(Model.MAX_EPOCH):
			if momentum == ZERO:
				break
			
			error_count_in_epoch = 0
			
			# Para cada índice, e dado do conjunto de treinamento: 
			for index, entry in enumerate(training_set):
				error = target_set[index] - self.feed_forward(entry)
				error_count_in_epoch += sum([error_instance != 0 for error_instance in error]) 
				self.__backpropagation(error, delta, epoch) # TODO avaliar possibilidade de transformar 'delta' em valor de retorno de '__backpropagation'
				apply_changes(delta)
			
			# Checa se irá calcular a acurácia do modelo para a época atual, utilizando o 'validation_set'
			if check_to_calculate_accuracy(momentum, epoch, len(validation_set)): 
				current_accuracy = self.classification_accuracy(validation_set, validation_target_set), epoch, self.weights 
				accuracy_timeline.append(current_accuracy)
				if accuracy_timeline[-1][0] > 1 - Model.ERROR_TOLERANCE or momentum < Model.INERTIA:
					if accuracy_timeline[-1][0] > accuracy_timeline[-2][0]:
						momentum = Model.INERTIA - 1
					else:
						momentum -= 1

			# Se não houver critério de para antecipada, checa se houve alteração nos pesos na última época
			#elif error_count_in_epoch == 0:
			#	break
		
		return accuracy_timeline

	def __repr__(self):
		pass
		#return '[np.asarray(' + repr(self.weights[0].tolist()) + '), np.asarray(' + repr(self.weights[1].tolist()) + ')]'
		
	def __str__(self):
		return 


# TODO: Função retorna uma nova classe extendida de Model, com os parâmetros de configuração kwargs
def from_architecture(**kwargs) -> type:
	pass


# main
if __name__ == '__main__':
		
	model = Model()
	shuffled_indexes = list(range(1326))
	shuffle(shuffled_indexes)
	input_data = np.load('./test/X.npy')[shuffled_indexes]
	target_data = np.load('./test/Y_classe.npy')[shuffled_indexes]
	
	TRAINING_SET_SIZE = 882
	VALIDATION_SET_SIZE = 294
	TEST_SET_SIZE = 150
	
	training_set = input_data[:TRAINING_SET_SIZE]
	training_target_set = target_data[:TRAINING_SET_SIZE]
	validation_set = input_data[TRAINING_SET_SIZE:TRAINING_SET_SIZE+VALIDATION_SET_SIZE]
	validation_target_set = target_data[TRAINING_SET_SIZE:TRAINING_SET_SIZE+VALIDATION_SET_SIZE]
	test_set = input_data[-TEST_SET_SIZE:]
	test_target_set = target_data[-TEST_SET_SIZE:]
	print(len(input_data), len(target_data), len(training_set), len(training_target_set), len(validation_set), len(validation_target_set))
	# remove dados de teste
	#input_data = input_data[:-130]
	#target_data = target_data[:-130]
		
	taxa = model.classification_accuracy(training_set, training_target_set) * 100
	print("taxa de acerto no conjunto de treinamento antes do treino: " + f"{taxa:.3f}%")
	
	acc = []		
	acc = model.train(training_set, training_target_set, validation_set, validation_target_set)
	try:	
		#acc = model.train(training_set, training_target_set, validation_set, validation_target_set)
		pass
	finally:
		print(*[f'{(100*m[0]):.6f}% -> epoch: {m[1]}' for m in acc], sep='\n')

		exit()		
		
		# O argumento training_validation_proportion tem por objetivo dividir o conjunto inicial de input em dois subconjuntos:
		# TODO: Delegar essa funcionalidade para outra função
		#	* Conjunto de treinamento 'training_set' (que efetivamente alimenta o cálculo dos erros)
		#	* Conjunto de validação 'validation_set' (que é usado para calcular a acurácia do modelo com dados que não alimentem o treinamento)
		
		#training_slice_index = int(len(input_set)*training_validation_proportion)
		#shuffle_index_range = list(range(len(input_set))) 
		#shuffle(shuffle_index_range)
		#input_set = input_set[shuffle_index_range]
		#target = target[shuffle_index_range]
		#training_set = input_set[:training_slice_index]
		#validation_set = input_set[training_slice_index:]
	


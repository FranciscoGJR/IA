# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# IGOR AUGUSTO DOS SANTOS - 11796851
# + ...
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm

from constants import *
from math import inf
from random import shuffle
from typing import List


class Model:

	NO_NODES_INPUT = 120
	NO_NODES_HIDDEN = 42
	NO_NODES_OUTPUT = 26
	DEFAULT_MAX_EPOCH = 100
	VALIDATION_INTERVAL = 1
	INERTIA = 6
	ERROR_TOLERANCE = 0.15

	# TODO: implementar função de alfa
	LEARNING_RATE = lambda x: 0.5

	# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
	ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))
	ACTIVATE_DERIVATIVE = lambda x: Model.ACTIVATE(x) * (1 - Model.ACTIVATE(x))

	# Alternativamente, pode se usar:
	#ACTIVATE = lambda x: np.maximum(0, x)
	#ACTIVATE_DERIVATIVE = lambda x: np.where(x > 0, 1, 0)

	def __init__(self, w: List[npt.NDArray[np.double]] = None):
		# armazena no modelo as informações de cada erro
		self.epoch_errors = []
		self.validation_error = []


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
		total_error = ZERO
		for index, entry in enumerate(test_set):
			output = self.feed_forward(entry)
			error = target[index] - output
			total_error += self.layer_error(error)
			if np.argmax(output) == np.argmax(target[index]):
				correct += ONE

		avg_error = total_error / len(test_set)
		return {'accuracy': correct / len(test_set), 'error': avg_error}

	# função auxiliar para calcular o erro quadrático médio
	def layer_error(self, error: npt.NDArray[np.double]) -> np.double:
		avg_error = error ** 2
		avg_error = np.sum(avg_error) ** 0.5
		return avg_error

	def plot_error(self) -> None:
		# Exibe o gráfico com os erros de cada época, no treino e na validação
		plt.plot(self.epoch_errors, label='Treinamento')
		validation_errors = [x['error'] for x in self.validation_error]
		validation_epochs = [x['epoch'] for x in self.validation_error]
		plt.plot(validation_epochs, validation_errors, label='Validação')
		plt.xlabel('Época')
		plt.ylabel('Erro')
		plt.legend()
		plt.show()

	def average_error(self, data, data_target) -> float:
		 return np.average(np.absolute(data_target - self.feed_forward(data)))

	def train(self, input_set: List[npt.NDArray[np.double]], target: List[npt.NDArray[np.double]], training_validation_proportion: float = 2/3, max_epoch: int = DEFAULT_MAX_EPOCH) -> List[npt.NDArray[np.double]]:
		
		# ------------------------------------------------- #
		# --- Definição de funções ajudantes de `train` --- #
		
		# Funções definidas dentro da função "train" a fim de garantir o escopo de acesso apenas à função train

		def apply_changes(delta) -> None:
			for index in range(len(self.weights)):
				self.weights[index] = self.weights[index] + delta[index]

		def check_to_calculate_accuracy(momentum: int, epoch: int, training_validation_proportion: float):
			if training_validation_proportion == ONE:
				return False
			if momentum == Model.INERTIA:
				return (epoch + ONE) % Model.VALIDATION_INTERVAL == ZERO
			else:
				return True
		
		# ------------------------------------------------- #
		# -------- Definição de parâmetros gerais --------- #
		
		# O argumento training_validation_proportion tem por objetivo dividir o conjunto inicial de input em dois subconjuntos:
		# TODO: Delegar essa funcionalidade para outra função
		#	* Conjunto de treinamento 'training_set' (que efetivamente alimenta o cálculo dos erros)
		#	* Conjunto de validação 'validation_set' (que é usado para calcular a acurácia do modelo com dados que não alimentem o treinamento)
		
		# levanta exceção em caso de 'training_validation_proportion' que impossibilite uma divisão do conjunto inicial 'input_set' 
		if training_validation_proportion > ONE or training_validation_proportion < HALF :
			raise ValueError(f"Parameter 'training_validation_proportion' should be betwen 1 and 0.5, got {training_validation_proportion}")		
				
		training_slice_index = int(len(input_set)*training_validation_proportion)
		shuffle_index_range = list(range(len(input_set))) 
		shuffle(shuffle_index_range)
		input_set = input_set[shuffle_index_range]
		target = target[shuffle_index_range]
		training_set = input_set[:training_slice_index]
		validation_set = input_set[training_slice_index:]
		
		# Variável "momentum" é usada para realizar a validação 'INERTIA' número de vezes, caso a validação tenha superado a validação anterior
		momentum = Model.INERTIA
		
		# Salva snapshots do modelo a cada nova validação utilizando o 'validation_set' 
		accuracy_timeline = [((self.classification_accuracy(validation_set, target[training_slice_index:])['accuracy'], -1, self.weights))]
		
		# Salva os valores de correção dos pesos
		delta = [
			np.full((Model.NO_NODES_HIDDEN, Model.NO_NODES_INPUT + ONE), ZERO, np.double),
			np.full((Model.NO_NODES_OUTPUT, Model.NO_NODES_HIDDEN + ONE), ZERO, np.double)
		]
		
		# ------------------------------------------------- #
		# -------------- Loop de Treinamento -------------- #
		
		# TODO: implementar funcionalidade de 'verbose_printing', para possibilidade de impressão de parâmetros do modelo a cada época
		progress_bar = tqdm.trange(max_epoch, ncols=150)
		for epoch in progress_bar:
			if momentum == ZERO:
				break
			total_error = ZERO
			# Para cada índice, e dado do conjunto de treinamento: 
			for index, entry in enumerate(training_set):
				error = target[index] - self.feed_forward(entry)
				total_error += self.layer_error(error)
				self.__backpropagation(error, delta, epoch) # TODO avaliar possibilidade de transformar 'delta' em valor de retorno de '__backpropagation'
				apply_changes(delta)

			# Checa se irá calcular a acurácia do modelo para a época atual, utilizando o 'validation_set'
			if check_to_calculate_accuracy(momentum, epoch, training_validation_proportion):
				classification_accuracy_result = self.classification_accuracy(validation_set, target[training_slice_index:])
				current_accuracy = classification_accuracy_result['accuracy'], epoch, self.weights
				accuracy_timeline.append(current_accuracy)
				self.validation_error.append({'epoch': epoch, 'error': classification_accuracy_result['error']})
				if accuracy_timeline[-1][0] > 1 - Model.ERROR_TOLERANCE or momentum < Model.INERTIA:
					if accuracy_timeline[-1][0] > accuracy_timeline[-2][0]:
						momentum = Model.INERTIA - 1
					else:
						momentum -= 1

			mean_error = total_error / len(training_set)
			self.epoch_errors.append(mean_error)
			progress_bar.set_description(f"Epoch: {epoch} - Erro quadrático médio: {mean_error:.3f} - Acurácia: {accuracy_timeline[-1][0]:.3f}")
		return accuracy_timeline

	def __repr__(self):
		return (f'Model({self.weights})')
		
	def __str__(self):
		return self.weights


# TODO: Função retorna uma nova classe extendida de Model, com os parâmetros de configuração kwargs
def from_architecture(**kwargs):
	pass


# main
if __name__ == '__main__':
		
	model = Model()

	input_data = np.load('./test/X.npy')
	target_data = np.load('./test/Y_classe.npy')
	# remove dados de teste
	input_data = input_data[:-130]
	target_data = target_data[:-130]
	
	taxa = model.classification_accuracy(input_data, target_data)['accuracy'] * 100
	print("taxa de acerto no conjunto de treinamento antes do treino: " + f"{taxa:.3f}%")
	
	acc = []		
	
	try:	
		acc = model.train(input_data, target_data)
		model.plot_error()
	except KeyboardInterrupt:
		model.plot_error()

	except Exception as e:
		print(e)
	finally:
		print(*[f'{(100*m[0])}% -> epoch: {m[1]}' for m in acc], sep='\n')

		exit()

		training_slice_index = int(len(input_set)*training_validation_proportion)
		shuffle_index_range = list(range(len(input_set))) 
		shuffle(shuffle_index_range)
		input_set = input_set[shuffle_index_range]
		target = target[shuffle_index_range]
		training_set = input_set[:training_slice_index]
		validation_set = input_set[training_slice_index:]



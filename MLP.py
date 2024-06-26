# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# GUILHERME DIAS JIMENES - 11911021
# IGOR AUGUSTO DOS SANTOS - 11796851
# LAURA PAIVA DE SIQUEIRA – 1120751


import ast
import math
import string
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
import json
import os
import pickle
import datetime
import types
import seaborn as sn

from constants import *
from inspect import getsource
from math import inf
from random import shuffle
from typing import List

class MetaModel(type):
	
	def __architecture__(cls):
		architecture_implementation = {}
		
		for arg, arg_type in {k: type(cls.__dict__[k]) for k in filter(lambda arg: arg[:2] != '__', cls.__dict__.keys())}.items():
			
			if arg_type in [types.FunctionType, types.LambdaType]:
				
				architecture_implementation[arg] = ast.dump(ast.parse(getsource(cls.__dict__[arg]).strip()))
			
			else:
				
				architecture_implementation[arg] = cls.__dict__[arg]
			
		for top_cls in cls.__bases__:
			if type(top_cls) == MetaModel:

				architecture_implementation = top_cls.__architecture__() | architecture_implementation

		return architecture_implementation


class Model(metaclass=MetaModel):
	
	# ------------------------------------------------------------------------------------- #
	# -------------------- Definição de arquitetura estática do modelo -------------------- #

	NO_NODES_INPUT = 120
	NO_NODES_HIDDEN = 60
	NO_NODES_OUTPUT = 26
	CLASSIFICATION_THRESHOLD = 0.5
	CLASSIFICATION_CRITERIA = 'threshold'
	# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
	ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))
	ACTIVATE_DERIVATIVE = lambda x: Model.ACTIVATE(x) * (1 - Model.ACTIVATE(x))
	# ----------------------------------------------------------------------- #
	# -------------- Definição de arquitetura de treinamento ---------------- #

	DEFAULT_MAX_EPOCH = 200
	VALIDATION_INTERVAL = 10
	INERTIA = 6 # Contador para parada antecipada
	ERR_RATE_THRESHOLD = 0.2
	AVG_ERROR_THRESHOLD = 0.01
	MODEL_EARLY_STOP_CRITERIA = 'avg_error'
	LEARNING_RATE_START = 1.0
	LEARNING_RATE_DECAY = 50.0# quanto menor, mais rápido o decaimento
     # Coeficiente de regularização L2
	USE_PENALIZATION = False
	L2_COEFF = 0.0001

	# Função que calcula a taxa de aprendizado para uma época
	def LEARNING_RATE(self, epoch: int) -> float:
		return type(self).LEARNING_RATE_START * np.e ** (-epoch / Model.LEARNING_RATE_DECAY)

	# Inicialização do modelo
	def __init__(self, w: List[npt.NDArray[np.double]] = None):
		# armazena no modelo as informações de cada erro
		self.epoch_errors = []
		self.validation_error = []

		# Espaço de memória dos neurônios
		self.nodes = [
			np.zeros(type(self).NO_NODES_INPUT, np.double),
			np.zeros(type(self).NO_NODES_HIDDEN, np.double),
			np.zeros(type(self).NO_NODES_OUTPUT, np.double)
		]

		# Inicialização dos pesos nas camadas entre os neurônios
		# Caso a função de inicialização não receba o argumento "w", inicializa o modelo com pesos aleatórios
		if w is None:
			self.weights = \
				[
					np.random.randn(type(self).NO_NODES_HIDDEN, type(self).NO_NODES_INPUT + 1) * 0.01,
					np.random.randn(type(self).NO_NODES_OUTPUT, type(self).NO_NODES_HIDDEN + 1) * 0.01
				]
		else:
			self.weights = \
				[
					np.full((type(self).NO_NODES_HIDDEN, type(self).NO_NODES_INPUT + 1), w[0], np.double),
					np.full((type(self).NO_NODES_OUTPUT, type(self).NO_NODES_HIDDEN + 1), w[1], np.double)
				]

	def set_weights_from_file(self, file_path: str) -> None:
		with open(file_path, 'rb') as f:
			self.weights = pickle.load(f)

	def feed_forward(self, data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:

		# aloca os dados de entrada na rede
		self.nodes[INPUT_LAYER] = data

		for current_layer in range(HIDDEN_LAYER, len(self.nodes)):
			previous_layer = current_layer - ONE

			# Calcula o produto escalar entre todos neurônios de chegada e seus respectivos pesos (função de agregação)
			layer_output = np.dot(self.weights[previous_layer], np.append(self.nodes[previous_layer], BIAS))

			# aplica a função de ativação na camada de neurônios
			self.nodes[current_layer] = np.vectorize(type(self).ACTIVATE)(layer_output)
			# a função vectorize é utilizada para converter uma função escalar em uma função vetorial, aplicando para todos os valores de um array

		return self.nodes[OUTPUT_LAYER]

	def classify(self, data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
		output = self.feed_forward(data)

		# Classificação utilizando o valor máximo
		if type(self).CLASSIFICATION_CRITERIA == 'max_value':
			return np.argmax(output)

		# Classificação utilizando o threshold
		if type(self).CLASSIFICATION_CRITERIA == 'threshold':
			threshold_array = np.vectorize(lambda x: x >= type(self).CLASSIFICATION_THRESHOLD)(output)
			if sum(threshold_array) == 1:
				return np.where(threshold_array == 1)[0][0]

		return None

	# calcula métricas de avaliação do modelo
	def evaluate_model(self, test_set: List[npt.NDArray[np.double]], test_target_set: List[npt.NDArray[np.double]]):
		correct = ZERO
		total_error = ZERO
		matriz = np.zeros((26, 26), dtype=int)

		for index, entry in enumerate(test_set):
			output = self.feed_forward(entry)
			error = test_target_set[index] - output
			total_error += self.average_layer_error(error)

			# utiliza o maior valor na camada de saída para classificação
			if type(self).CLASSIFICATION_CRITERIA == 'max_value':
				if np.argmax(output) == np.argmax(test_target_set[index]):
					correct += ONE

				matriz[np.argmax(output), np.argmax(test_target_set[index])] += 1

			# utiliza o threshold para classificação
			if type(self).CLASSIFICATION_CRITERIA == 'threshold':
				threshold_array = np.vectorize(lambda x: x >= type(self).CLASSIFICATION_THRESHOLD)(output)

				matriz[np.where(threshold_array == 1), np.where(test_target_set[index] == 1)] += 1

				if sum(threshold_array) == 1 and np.where(threshold_array == 1) == np.where(test_target_set[index] == 1):
					correct += ONE

		avg_error = total_error / len(test_set)
		return {'error_rate': 1 - (correct / len(test_set)), 'avg_error': avg_error, 'confusion_matrix': matriz}

	# função auxiliar para calcular o erro quadrático médio
	def average_layer_error(self, error: npt.NDArray[np.double]) -> np.double:
		avg_error = error ** 2
		avg_error = np.sum(avg_error) ** 0.5  
		return avg_error

	def plot_error(self, show: bool = True, save_path=None) -> None:
		# Exibe o gráfico com os erros de cada época, no treino e na validação
		plt.plot(self.epoch_errors, label='Treinamento')
		validation_errors = [x['error'] for x in self.validation_error]
		validation_epochs = [x['epoch'] for x in self.validation_error]
		plt.plot(validation_epochs, validation_errors, label='Validação')
		plt.xlabel('Época')
		plt.ylabel('Erro')
		plt.legend()
		if save_path is not None:
			plt.savefig(save_path)
		if show:
			plt.show()
		plt.close()
		plt.cla()
		plt.clf()


	def plot_confusion_matrix(self, confusion_matrix: npt.NDArray[np.double], show: bool=True, save_path=None) -> None:
		# Exibe o gráfico com a matriz de confusão
		alphabet = string.ascii_uppercase  # Get uppercase alphabet letters
		df_cm = pd.DataFrame(confusion_matrix.T, index=[alphabet[i] for i in range(26)],
							 columns=[alphabet[i] for i in range(26)])

		plt.figure(figsize=(10, 7))
		sn.heatmap(df_cm, annot=True, fmt='d')
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		if save_path is not None:
			plt.savefig(save_path)
		if show:
			plt.show()
		plt.close()
		plt.cla()
		plt.clf()

	def average_error(self, data, data_target) -> float:
		return np.average(np.absolute(data_target - self.feed_forward(data)))

	def save_model(self, model_name=None, confusion_matrix=None, custom_data=None) -> None:

		# checa se o diretório existe, caso contrário, cria
		os.makedirs('./modelos', exist_ok=True)

		# checa se o arquivo de modelos já existe, caso contrário, cria
		if not os.path.exists('./modelos/models.json'):
			with open('./modelos/models.json', 'w') as f:
				json.dump([], f)

		# lendo o json com os dados dos modelos já salvos
		with open('./modelos/models.json', 'r') as f:
			models = json.load(f)

		model_name = type(self).__name__ + '_' + str(int(datetime.datetime.now(datetime.UTC).timestamp())) if model_name is None else model_name

		# Cria diretório com o nome do modelo
		os.makedirs(f'./modelos/{model_name}', exist_ok=True)

		with open(f'./modelos/{model_name}/weights.pkl', 'wb') as f:
			pickle.dump(self.weights, f)

		self.plot_error(save_path=f'./modelos/{model_name}/error_plot.png', show=False)
		self.plot_confusion_matrix(confusion_matrix, save_path=f'./modelos/{model_name}/confusion_matrix.png', show=False)

		model_info = {
			"model_name": model_name,
			"timestamp": datetime.datetime.now(datetime.UTC).timestamp(),
			"epoch_errors": self.epoch_errors,
			"validation_errors": self.validation_error,
			"error_plot_path": f"./modelos/{model_name}_error_plot.png",
			"weights_path": f"./modelos/{model_name}_weights.npy",
			"confusion_matrix_path": f"./modelos/{model_name}_confusion_matrix.png" if confusion_matrix is not None else "",
			"custom_data": custom_data,
			"static": type(self).__architecture__()
		}
		models.append(model_info)

		# Salva as informações do modelo em um json
		with open('./modelos/models.json', 'w') as f:
			json.dump(models, f, indent=4)


	def train(
			self,
			training_set: List[npt.NDArray[np.double]],
			target_set: List[npt.NDArray[np.double]],
			validation_set: List[npt.NDArray[np.double]] = [],
			validation_target_set: List[npt.NDArray[np.double]] = [],
			verbose: bool = True
	) -> List[npt.NDArray[np.double]]:

		# levanta exceção em caso de inconsistência nos dados de entrada
		if len(training_set) != len(target_set) or len(validation_set) != len(validation_target_set):
			raise ValueError(f"Arguments of train function don't match length requirements\n" + \
							 "'training_set' and 'target_set' lenghts should be equal, got{len(training_set), len(target_set)}.\n" + \
							 "'validation_set' and 'validation_target_set' lenghts should be equal, got{len(validation_set), len(validation_target_set)}.\n")

		# ------------------------------------------------- #
		# --- Definição de funções ajudantes de `train` --- #

		# Funções definidas dentro da função "train" a fim de garantir o escopo de acesso apenas à função train

		def apply_changes(delta) -> None:
			for index in range(len(self.weights)):
				# Aplicando a atualização dos pesos incluindo o termo de regularização L1
				self.weights[index] = self.weights[index] + delta[index]

		def backpropagation(error: npt.NDArray[np.double], epoch) -> List[npt.NDArray[np.double]]:
      		# Inicializa a matriz com os deltas de cada camad
			delta = [
				np.full((type(self).NO_NODES_HIDDEN, type(self).NO_NODES_INPUT + 1), ZERO, np.double),
				np.full((type(self).NO_NODES_OUTPUT, type(self).NO_NODES_HIDDEN + 1), ZERO, np.double)
			]
			# Array que armazena as informações de erro da camada de saída
			error_info = []
			# Processamento da camada de saída realizado para cada neuronio
			for current_neuron, neuron in enumerate(self.nodes[OUTPUT_LAYER]):
				neuron_input = np.dot(self.weights[LAST][current_neuron], np.append(self.nodes[HIDDEN_LAYER], BIAS))
				error_correction = error[current_neuron] * type(self).ACTIVATE_DERIVATIVE(neuron_input)
				error_info.append(error_correction)
    			#Calcula o delta junto do coeficiente lambda do L2 para penalização de pesos
				if self.USE_PENALIZATION:
					delta[LAST][current_neuron] = (
						self.LEARNING_RATE(epoch) * error_correction * np.append(self.nodes[HIDDEN_LAYER], BIAS)
						+ type(self).L2_COEFF * self.weights[LAST][current_neuron]
					)
				else:
					delta[LAST][current_neuron] = self.LEARNING_RATE(ONE) * error_correction * np.append(
						self.nodes[HIDDEN_LAYER], BIAS)	 # calcula o delta do erro

			# Repetindo para a o processamento da camada oculta
			for current_neuron, neuron in enumerate(self.nodes[HIDDEN_LAYER]):
				err_sum = ZERO
    			#Calcula a contribuição de cada neurônio da camada de saída para o erro do neurônio atual
				for ie, er in enumerate(error_info):
					err_sum += er * self.weights[LAST][ie][current_neuron]
				neuron_input = np.dot(self.weights[FIRST][current_neuron], np.append(self.nodes[INPUT_LAYER], BIAS)) #Calcula o valor de entrada do neuronio
				error_correction = err_sum * type(self).ACTIVATE_DERIVATIVE(neuron_input) #Calcula a correção de erro
				#Calcula o delta junto do coeficiente lambda do L2 para penalização de pesos
				if self.USE_PENALIZATION:
					delta[FIRST][current_neuron] = (
						self.LEARNING_RATE(epoch) * error_correction * np.append(self.nodes[INPUT_LAYER], BIAS)
						+ type(self).L2_COEFF * self.weights[FIRST][current_neuron]
					)
				else:
					delta[FIRST][current_neuron] = self.LEARNING_RATE(epoch) * error_correction * np.append(
						self.nodes[INPUT_LAYER], BIAS)

			return delta

		def check_to_evaluate(momentum: int, epoch: int, validation_set_len: int) -> bool:
			if validation_set_len == ZERO:
				return False
			if momentum == type(self).INERTIA:
				return (epoch + ONE) % type(self).VALIDATION_INTERVAL == ZERO
			else:
				return True

		# ------------------------------------------------- #
		# -------- Definição de parâmetros gerais --------- #

		# Variável "momentum" é usada para realizar a validação 'INERTIA' número de vezes, caso a validação tenha superado a validação anterior
		momentum = type(self).INERTIA

		# Salva snapshots do modelo a cada nova validação utilizando o 'validation_set'
		training_timeline = [((self.evaluate_model(validation_set, validation_target_set), -1, self.weights))]

		# Helper dict
		early_stop_map = {'error_rate': type(self).ERR_RATE_THRESHOLD, 'avg_error': type(self).AVG_ERROR_THRESHOLD}

		# ------------------------------------------------- #
		# -------------- Loop de Treinamento -------------- #

		progress_bar = tqdm.trange(type(self).DEFAULT_MAX_EPOCH, ncols=100)
		for epoch in progress_bar:  #range(type(self).DEFAULT_MAX_EPOCH):
			if momentum == ZERO:
				break
			total_error = ZERO
			# Para cada índice, e dado do conjunto de treinamento:
			for index, entry in enumerate(training_set):
				error = target_set[index] - self.feed_forward(entry)
				total_error += self.average_layer_error(error)
				apply_changes(backpropagation(error, epoch))

			# Checa se irá calcular a acurácia do modelo para a época atual, utilizando o 'validation_set'
			if check_to_evaluate(momentum, epoch, len(validation_set)):
				evaluate_model_result = self.evaluate_model(validation_set, validation_target_set)
				# Adiciona informações dos erros e pesos em um snapshot do modelo
				training_timeline.append((evaluate_model_result, epoch, self.weights))
				self.validation_error.append({'epoch': epoch, 'error': evaluate_model_result['avg_error']})

				# Verifica se o erro médio do modelo é menor que o critério de parada
				if training_timeline[-1][0][type(self).MODEL_EARLY_STOP_CRITERIA] <= early_stop_map[
					type(self).MODEL_EARLY_STOP_CRITERIA] or momentum < type(self).INERTIA:
					# Calcula a diferença entre o erro atual e o anterior
					diff = training_timeline[-1][0][type(self).MODEL_EARLY_STOP_CRITERIA] - \
						   training_timeline[-2][0][type(self).MODEL_EARLY_STOP_CRITERIA]
					diff = abs(diff)
					# Se a diferença entre os erros for alta, e o erro atual for menor que o anterior, o treinamento continua
					if training_timeline[-1][0][type(self).MODEL_EARLY_STOP_CRITERIA] < training_timeline[-2][0][
						type(self).MODEL_EARLY_STOP_CRITERIA] and diff > 0.005:
						momentum = type(self).INERTIA - 1
					else:
						# Caso contrário, o contador para interromper o treinamento é decrementado
						momentum -= 1

			# exibição da interface de linha de comando
			if verbose:
				mean_error = total_error / len(training_set)
				self.epoch_errors.append(mean_error)
				progress_bar.set_description(
					f"Epoch: {epoch} - Erro: {mean_error:.3f} - Acerto {(1 - training_timeline[-1][0]['error_rate'])*100:.2f}% - α: {self.LEARNING_RATE(epoch):.3f}")

		return training_timeline

	# Acho que tem bug aqui
	def __repr__(self):
   		return type(self).__name__ + '([np.asarray(' + repr(self.weights[0].tolist()) + '), np.asarray(' + repr(self.weights[1].tolist()) + ')])'


def from_architecture(class_name: str, **kwargs) -> type:
	
	intersect = list(filter(lambda attribute: attribute[:2] != '__', kwargs.keys() & Model.__dict__.keys()))

	if len(intersect) < len(kwargs):
		raise ValueError(f'Dunder attributes and new atributes not allowed on creating a new Model class, got {kwargs}')

	for parameter in kwargs.keys():

		if type(kwargs[parameter]) != type(Model.__dict__[parameter]):
			raise TypeError(
				f"All parameters given to 'from_architecture' must be of same type on Model class, received {parameter}: {type(kwargs[parameter])}, expected: {type(Model.__dict__[parameter])}")

	if ('ACTIVATE' in kwargs.keys() and 'ACTIVATE_DERIVATIVE' not in kwargs.keys()) or (
			'ACTIVATE_DERIVATIVE' in kwargs.keys() and 'ACTIVATE' not in kwargs.keys()):
		raise ValueError("If 'ACTIVATE' lambda function is given, then 'ACTIVATE_DERIVATIVE' must also be given")

	return type(class_name, (Model,), kwargs)


# main
if __name__ == '__main__':

	model = Model()
	input_data = np.load('./test/X.npy')
	target_data = np.load('./test/Y_classe.npy')

	TRAINING_SET_SIZE = 902
	VALIDATION_SET_SIZE = 294
	TEST_SET_SIZE = 130

	test_set = input_data[-TEST_SET_SIZE:]
	test_target_set = target_data[-TEST_SET_SIZE:]

	print(len(test_set))

	# removendo ultimos 130 elementos
	input_data = input_data[:-TEST_SET_SIZE]
	target_data = target_data[:-TEST_SET_SIZE]

	# embaralhando os dados
	shuffled_indexes = list(range(TRAINING_SET_SIZE + VALIDATION_SET_SIZE))
	shuffle(shuffled_indexes)
	input_data = input_data[shuffled_indexes]
	target_data = target_data[shuffled_indexes]

	training_set = input_data[:TRAINING_SET_SIZE]
	training_target_set = target_data[:TRAINING_SET_SIZE]
	validation_set = input_data[TRAINING_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE]
	validation_target_set = target_data[TRAINING_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE]
	test_result = (model.evaluate_model(test_set, test_target_set))
	print(
		"taxa de acerto no conjunto de teste depois do treinamento: " + f"{(1 - test_result['error_rate']) * 100:.3f}%")
	print("erro médio do conjunto de teste depois do treinamento: " + f"{test_result['avg_error']:.3f}%")

	acc = []

	try:
		acc = model.train(training_set, training_target_set, validation_set, validation_target_set, verbose=True)
	except KeyboardInterrupt:
		print('Salvando o modelo')
	finally:
		test_result = (model.evaluate_model(test_set, test_target_set))
		model.plot_confusion_matrix(test_result['confusion_matrix'])
		model.plot_error()
		model.save_model(confusion_matrix=test_result['confusion_matrix'])
		print(
			"taxa de acerto no conjunto de teste depois do treinamento: " + f"{(1 - test_result['error_rate']) * 100:.3f}%")
		print("erro médio do conjunto de teste depois do treinamento: " + f"{test_result['avg_error']:.3f}%")
		print(*[f'{(100 * (1 - m[0]["error_rate"])):.6f}% -> epoch: {m[1]}' for m in acc], sep='\n')



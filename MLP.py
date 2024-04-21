# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# IGOR AUGUSTO DOS SANTOS - 11796851
# + ...


import ast
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

# TODO: Implementar método __new__ para evitar implantação de classes Model sem parâmetros (estáticos) necessários
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
	NO_NODES_HIDDEN = 42
	NO_NODES_OUTPUT = 26
	CLASSIFICATION_THRESHOLD = 0.85
	# Créditos: <a href="https://stackoverflow.com/a/29863846">Neil G, Stack Overflow</a>
	ACTIVATE = lambda x: np.exp(-np.logaddexp(0, -x))

	# ----------------------------------------------------------------------- #
	# -------------- Definição de arquitetura de treinamento ---------------- #

	DEFAULT_MAX_EPOCH = 200
	VALIDATION_INTERVAL = 1
	INERTIA = 6
	ERR_RATE_THRESHOLD = 0.2
	AVG_ERROR_THRESHOLD = 0.3
	MODEL_EARLY_STOP_CRITERIA = 'error_rate'
	# TODO: implementar função de alfa
	LEARNING_RATE = lambda x: 1 * np.e ** (-x / 50)
	ACTIVATE_DERIVATIVE = lambda x: Model.ACTIVATE(x) * (1 - Model.ACTIVATE(x))

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

	def feed_forward(self, data: npt.NDArray[np.double]) -> npt.NDArray[np.double]:

		self.nodes[INPUT_LAYER] = data

		for current_layer in range(HIDDEN_LAYER, len(self.nodes)):
			previous_layer = current_layer - ONE

			# Calcula o produto escalar entre todos neurônios de chegada e seus respectivos pesos (função de agregação)
			layer_output = np.dot(self.weights[previous_layer], np.append(self.nodes[previous_layer], BIAS))

			# aplica a função de ativação na camada de neurônios
			self.nodes[current_layer] = np.vectorize(type(self).ACTIVATE)(layer_output)

		return self.nodes[OUTPUT_LAYER]

	def evaluate_model(self, test_set: List[npt.NDArray[np.double]], test_target_set: List[npt.NDArray[np.double]]):
		correct = ZERO
		total_error = ZERO
		matriz = np.zeros((26, 26), dtype=int)

		for index, entry in enumerate(test_set):
			output = self.feed_forward(entry)
			error = test_target_set[index] - output
			total_error += self.average_layer_error(error)

			# Checagem anterior:
			#if np.argmax(output) == np.argmax(test_target_set[index]):
			#	correct += ONE

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

	def plot_error(self, save_path=None) -> None:
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
		plt.show()

	def plot_confusion_matrix(self, confusion_matrix: npt.NDArray[np.double], save_path=None) -> None:
		# Exibe o gráfico com a matriz de confusão
		df_cm = pd.DataFrame(confusion_matrix, index=[i for i in range(26)], columns=[i for i in range(26)])
		plt.figure(figsize=(10, 7))
		sn.heatmap(df_cm, annot=True, fmt='d')
		if save_path is not None:
			plt.savefig(save_path)
		plt.show()

	def average_error(self, data, data_target) -> float:
		return np.average(np.absolute(data_target - self.feed_forward(data)))

	def save_model(self, model_name = None, confusion_matrix = None) -> None:
		
		# lendo o json com os dados dos modelos já salvos
		with open('./modelos/models.json', 'r') as f:
			models = json.load(f)

		model_name = type(self).__name__ + '_' + str(int(datetime.datetime.now(datetime.UTC).timestamp())) if model_name is None else model_name

		# Cria diretório com o nome do modelo
		os.makedirs(f'./modelos/{model_name}', exist_ok=True)

		with open(f'./modelos/{model_name}/weights.pkl', 'wb') as f:
			pickle.dump(self.weights, f)

		self.plot_error(f'./modelos/{model_name}/error_plot.png')
		self.plot_confusion_matrix(confusion_matrix, f'./modelos/{model_name}/confusion_matrix.png')

		model_info = {
			"model_name": model_name,
			"timestamp": datetime.datetime.now(datetime.UTC).timestamp(),
			"epoch_errors": self.epoch_errors,
			"validation_errors": self.validation_error,
			"error_plot_path": f"./modelos/{model_name}_error_plot.png",
			"weights_path": f"./modelos/{model_name}_weights.npy",
			"confusion_matrix_path": f"./modelos/{model_name}_confusion_matrix.png" if confusion_matrix is not None else "",
			"static": type(self).__architecture__()
		}
		models.append(model_info)

		# Salva as informações do modelo em um json
		with open('./modelos/models.json', 'w') as f:
			json.dump(models, f, indent=4)


	# TODO: implementar funcionalidade de 'verbose_printing', para possibilidade de impressão de parâmetros do modelo a cada época
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
				self.weights[index] = self.weights[index] + delta[index]

		# TODO: é possível aplicar as mudanças assim que o delta for calculado? eliminando a função `apply_changes`
		def backpropagation(error: npt.NDArray[np.double], epoch) -> List[npt.NDArray[np.double]]:

			delta = [
				np.full((type(self).NO_NODES_HIDDEN, type(self).NO_NODES_INPUT + ONE), ZERO, np.double),
				np.full((type(self).NO_NODES_OUTPUT, type(self).NO_NODES_HIDDEN + ONE), ZERO, np.double)
			]
			error_info = []

			for current_neuron, neuron in enumerate(self.nodes[OUTPUT_LAYER]):
				neuron_input = np.dot(self.weights[LAST][current_neuron], np.append(self.nodes[HIDDEN_LAYER], BIAS))
				error_correction = error[current_neuron] * type(self).ACTIVATE_DERIVATIVE(neuron_input)
				error_info.append(error_correction)
				delta[LAST][current_neuron] = type(self).LEARNING_RATE(ONE) * error_correction * np.append(
					self.nodes[HIDDEN_LAYER], BIAS)

			for current_neuron, neuron in enumerate(self.nodes[HIDDEN_LAYER]):
				err_sum = ZERO

				for ie, er in enumerate(error_info):
					err_sum += er * self.weights[LAST][ie][current_neuron]

				neuron_input = np.dot(self.weights[FIRST][current_neuron], np.append(self.nodes[INPUT_LAYER], BIAS))
				error_correction = err_sum * type(self).ACTIVATE_DERIVATIVE(neuron_input)
				delta[FIRST][current_neuron] = type(self).LEARNING_RATE(epoch) * error_correction * np.append(
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

		# TODO: implementar funcionalidade de 'verbose_printing', para possibilidade de supressão da impressão de parâmetros do modelo a cada época (a fim de melhorar velocidade)
		progress_bar = tqdm.trange(type(self).DEFAULT_MAX_EPOCH, ncols=100)
		for epoch in progress_bar:  #range(type(self).DEFAULT_MAX_EPOCH):
			if momentum == ZERO:
				break
			total_error = ZERO
			# Para cada índice, e dado do conjunto de treinamento:
			for index, entry in enumerate(training_set):
				error = training_target_set[index] - self.feed_forward(entry)
				total_error += self.average_layer_error(error)
				apply_changes(backpropagation(error, epoch))

			# Checa se irá calcular a acurácia do modelo para a época atual, utilizando o 'validation_set'
			if check_to_evaluate(momentum, epoch, len(validation_set)):
				evaluate_model_result = self.evaluate_model(validation_set, validation_target_set)
				training_timeline.append((evaluate_model_result, epoch, self.weights))
				self.validation_error.append({'epoch': epoch, 'error': evaluate_model_result['error_rate']})
				# TODO: fazer a checagem pelo erro médio, ao invés da acurácia
				if training_timeline[-1][0][type(self).MODEL_EARLY_STOP_CRITERIA] <= early_stop_map[
					type(self).MODEL_EARLY_STOP_CRITERIA] or momentum < type(self).INERTIA:
					if training_timeline[-1][0][type(self).MODEL_EARLY_STOP_CRITERIA] < training_timeline[-2][0][
						type(self).MODEL_EARLY_STOP_CRITERIA]:
						momentum = type(self).INERTIA - 1
					else:
						momentum -= 1

			if verbose:
				mean_error = total_error / len(training_set)
				self.epoch_errors.append(mean_error)
				progress_bar.set_description(
					f"Epoch: {epoch} - Erro: {mean_error:.3f} - Taxa Apren.: {type(self).LEARNING_RATE(epoch):.3f}")

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
	shuffled_indexes = list(range(1326))
	shuffle(shuffled_indexes)
	input_data = np.load('./test/X.npy')[shuffled_indexes]
	target_data = np.load('./test/Y_classe.npy')[shuffled_indexes]

	TRAINING_SET_SIZE = 882
	VALIDATION_SET_SIZE = 294
	TEST_SET_SIZE = 150

	training_set = input_data[:TRAINING_SET_SIZE]
	training_target_set = target_data[:TRAINING_SET_SIZE]
	validation_set = input_data[TRAINING_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE]
	validation_target_set = target_data[TRAINING_SET_SIZE:TRAINING_SET_SIZE + VALIDATION_SET_SIZE]
	test_set = input_data[-TEST_SET_SIZE:]
	test_target_set = target_data[-TEST_SET_SIZE:]
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



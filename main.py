from MLP import Model, from_architecture
from random import shuffle
from functions import activation_functions
import numpy as np
import json

input_data = np.load('./test/X.npy')
target_data = np.load('./test/Y_classe.npy')

TRAINING_SET_SIZE = 902
VALIDATION_SET_SIZE = 294
TEST_SET_SIZE = 130

test_set = input_data[-TEST_SET_SIZE:]
test_target_set = target_data[-TEST_SET_SIZE:]

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

# lendo o json com as definições de modelos
with open('model_definitions.json', 'r') as file:
    model_definitions = json.load(file)

# ordena os modelos pelo número de neuronios na camada oculta
model_definitions.sort(key=lambda x: x.get('NO_NODES_HIDDEN', 42), reverse=True)

print('Quantidade de modelos a serem treinados:', len(model_definitions))
for definition in model_definitions:
    print(f'Treinando o modelo {definition.get('class_name', "Modelo sem nome")}')

    activation_name = definition.get('ACTIVATE')
    if activation_name in activation_functions:
        definition['ACTIVATE'], definition['ACTIVATE_DERIVATIVE'] = activation_functions[activation_name]

    model = from_architecture(**definition)()
    model.train(training_set, training_target_set, validation_set, validation_target_set)
    result = model.evaluate_model(test_set, test_target_set)
    model.save_model(model_name=definition.get('class_name', 'Sem Nome'), confusion_matrix=result['confusion_matrix'])

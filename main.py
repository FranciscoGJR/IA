from MLP import Model, from_architecture
from random import shuffle
from functions import activation_functions
from data_loader import cross_validation_split
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

# executa primeiro os modelos com prioridade alta e baixo número de neuronios ocultos
model_definitions.sort(key=lambda x: (-x.get('PRIORITY', 0), x.get('NO_NODES_HIDDEN', 42)))

print('Quantidade de modelos a serem treinados:', len(model_definitions))
for i, definition in enumerate(model_definitions):
    print(f'{i+1} - {definition.get("class_name", "Modelo sem nome")}')

for definition in model_definitions:
    if definition.get('PRIORITY'):
        del definition['PRIORITY']

    print(f'Treinando o modelo {definition.get('class_name', "Modelo sem nome")}')

    activation_name = definition.get('ACTIVATE')
    if activation_name in activation_functions:
        definition['ACTIVATE'], definition['ACTIVATE_DERIVATIVE'] = activation_functions[activation_name]

    CROSS_VALIDATION_SIZE = definition.get('CROSS_VALIDATION', 1)
    if definition.get('CROSS_VALIDATION'):
        del definition['CROSS_VALIDATION']

    if CROSS_VALIDATION_SIZE == 1:
        model = from_architecture(**definition)()
        model.train(training_set, training_target_set, validation_set, validation_target_set)
        result = model.evaluate_model(test_set, test_target_set)
        custom_data = {
            "test_accuracy": (1 - result['error_rate']) * 100,
            "test_avg_error": result['avg_error'],
            "crossvalidation": False,
        }
        model.save_model(model_name=definition.get('class_name', 'Sem Nome'), confusion_matrix=result['confusion_matrix'], custom_data=custom_data)

    else:
        print('Modelo com validação cruzada')
        folds, target_folds = cross_validation_split(input_data, target_data, CROSS_VALIDATION_SIZE)

        for i in range(CROSS_VALIDATION_SIZE):
            print(f'fold {i+1} de {CROSS_VALIDATION_SIZE}')
            fold_training_set = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
            fold_training_target_set = np.concatenate([fold for j, fold in enumerate(target_folds) if j != i])

            fold_validation_set = folds[i]
            fold_validation_target_set = target_folds[i]

            model = from_architecture(**definition)()
            model.train(fold_training_set, fold_training_target_set, fold_validation_set, fold_validation_target_set)
            result = model.evaluate_model(test_set, test_target_set)
            custom_data = {
                "test_accuracy": (1 - result['error_rate']) * 100,
                "test_avg_error": result['avg_error'],
                "crossvalidation": True,
                "crossvalidation_fold": i+1,
                "crossvalidation_size": CROSS_VALIDATION_SIZE,
                "crossvalidation_group": definition.get('class_name')
            }
            model.save_model(model_name=definition.get('class_name', 'Sem Nome') + f'_{i}', confusion_matrix=result['confusion_matrix'], custom_data=custom_data)


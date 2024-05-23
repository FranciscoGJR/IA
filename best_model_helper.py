# EP ACH2016 - "IA" | TURMA 04
# BRUNO LEITE DE ANDRADE - 11369642
# FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
# GUILHERME DIAS JIMENES - 11911021
# IGOR AUGUSTO DOS SANTOS - 11796851
# LAURA


import json

with open('./modelos_apresentacao/models.json') as f:
    data = json.load(f)

cross_validation_groups = {}

print('#### Modelos com cross validation ####')

# filtra os modelos cross validation
for item in data:
    custom_data = item.get('custom_data')
    if custom_data and custom_data.get('crossvalidation'):
        if custom_data['crossvalidation_group'] not in cross_validation_groups:
            cross_validation_groups[custom_data['crossvalidation_group']] = []
        cross_validation_groups[custom_data['crossvalidation_group']].append(item)

# calcula médias de acurácia para o cross validation
for group in cross_validation_groups:
    sum_accuracy = 0
    for item in cross_validation_groups[group]:
        sum_accuracy += item['custom_data']['test_accuracy']
    avg_accuracy = sum_accuracy / len(cross_validation_groups[group])
    print(f'Grupo {group}: {avg_accuracy:.2f}%')

print('\n\n\n')

# ordena modelos por melhor acurácia
data = sorted(data, key=lambda x: x['custom_data']['test_accuracy'], reverse=True)

for item in data:
    acuracia = (item['custom_data']['test_accuracy'])
    print(item['model_name'], f'{acuracia:.2f}%')
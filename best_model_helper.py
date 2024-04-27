import json

with open('./modelos_interessantes/models.json') as f:
    data = json.load(f)

# ordena modelos por melhor acurácia

data = sorted(data, key=lambda x: x['validation_errors'][-1]["error"])

for item in data:
    acuracia = (1 - item['validation_errors'][-1]["error"]) * 100
    print(item['model_name'], f'{acuracia:.2f}%')
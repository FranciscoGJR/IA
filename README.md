# MLP - Reconhecimento de caracteres

## Integrantes
* BRUNO LEITE DE ANDRADE - 11369642
* FRANCISCO OLIVEIRA GOMES JUNIOR - 12683190
* GUILHERME DIAS JIMENES - 11911021
* IGOR AUGUSTO DOS SANTOS - 11796851
* LAURA

## Observações
* O arquivo `main.py` é responsável por treinar os modelos e armazenar os resultados na pasta `./modelos`.
* O arquivo `mlp.py` contém a implementação da rede neural
* O arquivo `best_model_helper.py` é responsável por carregar os modelos treinados e exibir os resultados.
* O arquivo `data_loader.py` é responsável por carregar os dados de treino e teste.
* O arquivo `functions.py` contém funções de ativação e derivadas.
* O código foi desenvolvido em Python 3.12

## Como executar
1. Clone o repositório
2. Crie o ambiente virtual com o comando:
```bash
python3 -m venv .venv
```
3. Ative o ambiente virtual com o comando:
```bash
source .venv/bin/activate
```
4. Instale as dependências com o comando:
```bash
pip install -r requirements.txt
```
5. Execute o arquivo `main.py` com o comando:
```bash
python main.py
```

## Utilização
```
[?] Selecione as arquiteturas que deseja treinar (use espaço para selecionar e setas para mover)::
 > [ ] Todos
   [ ] 60n_max_150epoch_recomendado
   [ ] 60n_threshold_early_stop
   [ ] 60n_threshold_cross_validation
   [ ] 10n_max
   [ ] 26n_threshold
   [ ] 26n_max
   [ ] 60n_threshold
   [ ] 60n_threshold_leaky_relu
   [ ] 60n_threshold_tanh
   [ ] 60n_max_300epoch
   [ ] 60n_threshold_relu
   [ ] 60n_maxvalue_sigmoid


```
1. Utilize as setas para navegar, espaço para selecionar e enter para confirmar. O primeiro modelo é o recomendado com base nos testes
2. Os modelos serão treinados conforme a seleção do usuário e os resultados serão armazenados na pasta `./modelos`.
3. Caso queira visualizar os resultados, execute o arquivo `best_model_helper.py` com o comando:
```bash
python best_model_helper.py
```

4. Exemplo de saida:
```
#### Modelos com cross validation ####
Grupo 60n_threshold_cross_validation: 68.23%




60n_max 81.54%
26n_max 76.92%
300_threshold 71.54%
60n_threshold_cross_validation_5 70.77%
60n_threshold_cross_validation_8 70.00%
60n_threshold_cross_validation_3 69.23%
60n_threshold 69.23%
60n_threshold_cross_validation_0 68.46%
60n_threshold_cross_validation_1 68.46%
60n_threshold_cross_validation_7 68.46%
60n_threshold_cross_validation_9 68.46%
60n_threshold_cross_validation_2 67.69%
60n_threshold_cross_validation_4 65.38%
60n_threshold_cross_validation_6 65.38%
60n_threshold_leaky_relu 63.08%
10n_max 62.31%
60n_threshold_tanh 62.31%
60n_threshold_early_stop 62.31%
60n_threshold_tanh 60.00%
```
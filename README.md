# Perceptron Neural Network Implementation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pytest](https://img.shields.io/badge/Tests-pytest-green)
![Status](https://img.shields.io/badge/Project-Academic-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="https://i.imgur.com/jFXlHBs.png" width="450">
</p>

## Sobre o projeto

Este projeto apresenta a implementação de uma **Rede Neural Artificial do tipo Perceptron**, proposta por [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) em 1958.

O perceptron é um dos primeiros modelos de neurônio artificial e constitui a base histórica das áreas de:

- [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning)
- [Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)

O objetivo deste projeto é demonstrar, de forma didática e modular, o funcionamento de um perceptron através de:

- geração de datasets
- treinamento supervisionado
- normalização de dados
- avaliação de desempenho
- visualização da fronteira de decisão

## Modelo matemático

O perceptron calcula uma combinação linear das entradas:

$y = \mathrm{sign}(w^T x + b)$

onde:

- $x$ representa o vetor de entradas  
- $w$ representa o vetor de pesos sinápticos  
- $b$ é o bias  
- $\mathrm{sign}(\cdot)$ é a função de ativação que define a classe da saída

Durante o treinamento, os pesos são atualizados pela seguinte regra de aprendizado:

$w_{t+1} = w_t + \eta\, y\, x$

onde:

- $\eta$ é a taxa de aprendizado  
- $y$ é o rótulo verdadeiro

## Funcionalidades

### 1. Implementação do Perceptron

O projeto implementa um perceptron de camada simples capaz de aprender problemas **linearmente separáveis** como:

- AND
- OR
- datasets sintéticos bidimensionais

Também demonstra a limitação clássica do perceptron ao tentar aprender o problema **XOR**, que não é linearmente separável.

### Treinamento do Perceptron – classificação de regiões

O GIF mostra o treinamento do perceptron para classificar regiões como Norte (1) ou Sul (-1).  
Como os dados não são linearmente separáveis, pontos verdes são corretos e vermelhos estão incorretos; a linha azul é a fronteira de decisão.

<p align="center">
  <img src="https://imgur.com/tu7jCk1.gif" width="450">
</p>

### 2. Pipeline de treinamento

O pipeline inclui:

- geração automática de datasets
- divisão treino/teste
- normalização com StandardScaler
- treinamento do perceptron
- cálculo de acurácia

### 3. Visualização da fronteira de decisão

Para datasets bidimensionais, o sistema gera gráficos mostrando a **fronteira de decisão aprendida pelo modelo**.

Isso permite visualizar como o perceptron separa as classes no espaço de características.

## Estrutura do projeto

```
project/
│
├── data/
│ ├── raw/ # datasets gerados
│ └── test/ # datasets usados nos testes
│
├── scripts/ # scripts para execução do projeto
│
├── src/ # código-fonte principal
│ ├── config/ # configurações do perceptron
│ ├── datasets/ # geração e manipulação de datasets
│ ├── helpers/ # funções auxiliares
│ ├── models/ # implementação do perceptron
│ ├── prediction/ # lógica de predição
│ ├── preprocessing/# normalização e preparação dos dados
│ ├── training/ # pipeline de treinamento
│ └── visualization/# visualização da fronteira de decisão
│
├── tests/ # testes automatizados (pytest)
│
├── requirements.txt # dependências do projeto
└── README.md
```

## Instalação

Clone o repositório:

```
git clone git@github.com:natamleao/RNA-Perceptron.git
cd RNA-Perceptron

```
Crie um ambiente virtual:

```
python -m venv venv

```
Ative o ambiente:

Linux / MacOS

```
source venv/bin/activate
```

Instale as dependências:

```
pip install -r requirements.txt
```

## Executar experimentos

Execute o pipeline completo:

```
python src/main.py
```

ou
```
chmod +x /scripts/run_perceptron.sh
```

```
./scripts/run_perceptron.sh
```

## Testes automatizados

O projeto inclui testes automatizados utilizando [pytest](https://pytest.org).

Execute:

```
pytest -v
```

Os testes verificam:

- geração de datasets
- carregamento de dados
- normalização
- divisão treino/teste
- treinamento do perceptron
- comportamento esperado em AND e XOR

## Resultados esperados

| Dataset | Resultado/Acurácia |
|-------|-------|
| AND | 100% |
| OR | 100% |
| XOR | Falha esperada |

Isso demonstra a limitação fundamental do perceptron para problemas **não linearmente separáveis**.

# Autor

**Natam Leão Ferreira**

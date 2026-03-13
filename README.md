# Perceptron Neural Network Implementation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pytest](https://img.shields.io/badge/Tests-pytest-green)
![Status](https://img.shields.io/badge/Project-Academic-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="https://i.imgur.com/jFXlHBs.png" width="450">
</p>

## Sobre o projeto

Este projeto apresenta a implementação de uma **Rede Neural Artificial do tipo Perceptron**, proposta por :contentReference[oaicite:0]{index=0} em 1958.

O perceptron é um dos primeiros modelos de neurônio artificial e constitui a base histórica das áreas de:

- :contentReference[oaicite:1]{index=1}
- :contentReference[oaicite:2]{index=2}

O objetivo deste projeto é demonstrar, de forma didática e modular, o funcionamento de um perceptron através de:

- geração de datasets
- treinamento supervisionado
- normalização de dados
- avaliação de desempenho
- visualização da fronteira de decisão

# Modelo Matemático

O perceptron calcula uma combinação linear das entradas:

y = \text{sign}(w \cdot x + b)

onde:

- \(x\) representa o vetor de entradas
- \(w\) representa os pesos sinápticos
- \(b\) é o bias
- a função `sign()` define a classe da saída

Durante o treinamento, os pesos são atualizados pela regra de aprendizado:

\[
w_{t+1} = w_t + \eta \cdot y \cdot x
\]

onde:

- \( \eta \) é a taxa de aprendizado
- \( y \) é o rótulo verdadeiro

# Funcionalidades

## 1. Implementação do Perceptron

O projeto implementa um perceptron de camada simples capaz de aprender problemas **linearmente separáveis** como:

- AND
- OR
- datasets sintéticos bidimensionais

Também demonstra a limitação clássica do perceptron ao tentar aprender o problema **XOR**, que não é linearmente separável.

## 2. Pipeline de Treinamento

O pipeline inclui:

- geração automática de datasets
- divisão treino/teste
- normalização com StandardScaler
- treinamento do perceptron
- cálculo de acurácia

## 3. Visualização da Fronteira de Decisão

Para datasets bidimensionais, o sistema gera gráficos mostrando a **fronteira de decisão aprendida pelo modelo**.

Isso permite visualizar como o perceptron separa as classes no espaço de características.

# Estrutura do Projeto

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

# Instalação

Clone o repositório:

```
git clone <repo-url>
cd project

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

# Executar Experimentos

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

# Testes Automatizados

O projeto inclui testes utilizando :contentReference[oaicite:4]{index=4}.

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

# Resultados Esperados

| Dataset | Resultado |
|-------|-------|
| AND | 100% |
| OR | 100% |
| XOR | Falha esperada |

Isso demonstra a limitação fundamental do perceptron para problemas **não linearmente separáveis**.

# Autor

**Natam Leão Ferreira**  

> Projeto desenvolvido para fins de aprendizado.
```

* colocar um **GIF do perceptron aprendendo a fronteira de decisão** (fica absurdamente bonito no GitHub).

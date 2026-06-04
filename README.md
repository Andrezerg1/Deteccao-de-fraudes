# Detecção de Fraudes com Machine Learning e Redes Neurais

## Sobre o Projeto

Este projeto implementa um sistema de **detecção de fraudes em transações financeiras** utilizando **Machine Learning** e **Redes Neurais Artificiais** com Python.

O objetivo é analisar características de uma transação, como valor, horário e local de compra, para prever automaticamente se ela possui comportamento suspeito ou não.

O modelo foi desenvolvido utilizando uma rede neural do tipo **MLP (Multi-Layer Perceptron)**, capaz de aprender padrões complexos presentes nos dados históricos e identificar possíveis fraudes em novas transações.

---

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-Learn
* Redes Neurais (MLPClassifier)

---

## Estrutura dos Dados

O conjunto de dados contém as seguintes informações:

| Coluna         | Descrição                                     |
| -------------- | --------------------------------------------- |
| transaction_id | Identificador da transação                    |
| customer_id    | Identificador do cliente                      |
| amount         | Valor da transação                            |
| time           | Hora da transação                             |
| location       | Local da compra (Online ou Loja Física)       |
| is_fraud       | Indica se a transação é fraude (1) ou não (0) |

### Exemplo

| transaction_id | customer_id | amount  | time | location    | is_fraud |
| -------------- | ----------- | ------- | ---- | ----------- | -------- |
| TRANS_1668     | CUST_257    | 3871.09 | 10   | Loja Física | 0        |
| TRANS_731      | CUST_391    | 4222.56 | 23   | Online      | 1        |

---

# Etapas do Projeto

## 1. Pré-processamento dos Dados

Antes do treinamento, os dados passam por algumas transformações importantes:

### Remoção de Colunas Irrelevantes

As colunas `transaction_id` e `customer_id` são removidas por não possuírem valor preditivo para o modelo.

```python
df = df.drop(columns=['transaction_id', 'customer_id'])
```

---

### Codificação de Variáveis Categóricas

A variável `location` é convertida para formato numérico utilizando One-Hot Encoding.

Exemplo:

| location    |
| ----------- |
| Online      |
| Loja Física |

Transforma-se em:

| location_Online |
| --------------- |
| 1               |
| 0               |

---

### Separação entre Entradas e Saídas

As características utilizadas pelo modelo ficam em `X`, enquanto a variável alvo fica em `y`.

```python
X = df_encoded.drop(columns=['is_fraud'])
y = df_encoded['is_fraud']
```

---

### Normalização dos Dados

Os atributos numéricos são normalizados utilizando StandardScaler.

```python
scaler = StandardScaler()
```

Isso garante que variáveis como valor da compra e horário estejam na mesma escala, facilitando o aprendizado da rede neural.

---

### Divisão entre Treino e Teste

O conjunto de dados é dividido em:

* 80% para treinamento
* 20% para testes

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

---

# Arquitetura da Rede Neural

Foi utilizada uma rede neural do tipo Multi-Layer Perceptron (MLP).

```python
MLPClassifier(
    hidden_layer_sizes=(10, 8, 10),
    max_iter=500,
    learning_rate_init=0.01,
    activation='relu'
)
```

### Estrutura

```text
Entrada
   │
   ▼

[10 neurônios]
   │
   ▼

[8 neurônios]
   │
   ▼

[10 neurônios]
   │
   ▼

Saída
(Fraude ou Não Fraude)
```

### Parâmetros

| Parâmetro          | Função                           |
| ------------------ | -------------------------------- |
| hidden_layer_sizes | Define as camadas ocultas        |
| activation='relu'  | Função de ativação               |
| learning_rate_init | Taxa de aprendizado              |
| max_iter           | Máximo de épocas de treinamento  |
| random_state       | Reprodutibilidade dos resultados |

---

# Avaliação do Modelo

Após o treinamento, o modelo foi avaliado utilizando o conjunto de teste.

### Resultados

| Métrica         | Valor  |
| --------------- | ------ |
| Accuracy        | 74.50% |
| Precision Média | 74.46% |
| Recall Médio    | 74.49% |
| F1-Score Médio  | 74.47% |

### Classe Não Fraude (0)

* Precision: 73.10%
* Recall: 74.10%
* F1-Score: 73.60%

### Classe Fraude (1)

* Precision: 75.84%
* Recall: 74.87%
* F1-Score: 75.35%

Esses resultados indicam que a rede neural conseguiu identificar padrões relevantes nos dados e apresentou desempenho equilibrado entre as classes.

---

# Previsão de Novas Transações

Após o treinamento, o modelo pode analisar novas transações.

### Exemplo de Entrada

```python
new_transactions = [
    {"amount": 3871.09, "time": 10, "location": "Loja Física"},
    {"amount": 12.56, "time": 12, "location": "Online"}
]
```

---

### Saída

| amount  | time | location    | fraud_probability | is_fraud_predicted |
| ------- | ---- | ----------- | ----------------- | ------------------ |
| 3871.09 | 10   | Loja Física | 0.478             | 0                  |
| 12.56   | 12   | Online      | 0.429             | 0                  |

---

## Como Interpretar

A coluna `fraud_probability` representa a probabilidade estimada de fraude.

Exemplo:

```text
0.02 → 2% de chance de fraude
0.50 → 50% de chance de fraude
0.95 → 95% de chance de fraude
```

Já a coluna `is_fraud_predicted` representa a classificação final:

```text
0 = Não Fraude
1 = Fraude
```

---

# Objetivos de Aprendizagem

Este projeto foi desenvolvido para estudar conceitos de:

* Machine Learning
* Redes Neurais Artificiais
* Classificação Binária
* Pré-processamento de Dados
* One-Hot Encoding
* Normalização de Dados
* Avaliação de Modelos
* Probabilidade de Classificação
* Detecção de Fraudes

---

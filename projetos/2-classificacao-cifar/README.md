# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Letícia Giulia Ribeiro Gomes**

### 1️⃣ Resumo da Arquitetura do Modelos

Para resolver o desafio do CIFAR-10 sem travar a CPU durante o treino, montei uma CNN bem simples e direta usando o Sequential do Keras, dividida em três partes principais:

. Data Augmentation: Logo no começo do modelo, coloquei camadas de RandomFlip("horizontal"), RandomRotation(0.1) e RandomZoom(0.1). A ideia foi fazer com que cada época de treino visse as imagens com pequenas variações, o que ajuda muito a evitar overfitting sem precisar gerar arquivos extras no disco.

. Blocos Convolucionais: Criei 3 blocos sequenciais para extração de características. Cada um tem:

Uma camada Conv2D (32, 64 e 128 filtros, respectivamente) com ativação ReLU.
BatchNormalization para ajudar o treino a convergir mais rápido.
MaxPooling2D para diminuir a dimensão da imagem e focar só no que importa.

. Camadas Finais: No final, usei Flatten para transformar tudo em um vetor, adicionei Dropout (0.4 e 0.3) pra dar mais uma segurada na memorização, uma camada Dense de 128 neurônios e a camada de saída com Dense(10) e Softmax para dar as probabilidades de cada classe.

### 2️⃣ Bibliotecas Utilizadas

. TensorFlow / Keras: 2.15.0 (ou superior) — Usei para carregar o dataset, construir e treinar a rede, aplicar os callbacks e converter o modelo final.
. NumPy: 1.26.4 — Usei na parte de inferência para formatar as matrizes das imagens e processar os resultados.

### 3️⃣ Técnica de Otimização do Modelo

No script optimize_model.py, apliquei a técnica de Dynamic Range Quantization na hora de converter o modelo com o TFLiteConverter, o que ela faz é pegar os pesos da rede que estavam em precisão alta (float32) e transformar em inteiros de 8 bits (int8). O ganho é enorme: o arquivo fica quase 4 vezes menor e passa a rodar muito mais rápido em dispositivos de borda (Edge AI), como celulares ou robôs, mantendo praticamente a mesma acurácia.

### 4️⃣ Resultados Obtidos

Acurácia Final de Validação: aproximadamente 72.50 % (pode variar um pouco a cada treino por conta do ambiente).
Tamanho do model.h5: aproximadamente 2.1 MB 
Tamanho do model.tflite: aproximadamente 580 KB (ficou cerca de 72% menor)
### 5️⃣ Comentários Adicionais (Opcional)

A principal dificuldade foi o CIFAR-10, que é bem mais chato que o MNIST porque as imagens são coloridas e têm fundos muito ruidosos, então foi mais demorado tentar ajustar o tamanho da rede para ter um resultado bacana de acurácia sem fazer a CPU sofrer para treinar.
Decisões de Código: Optar por colocar a Data Augmentation diretamente dentro da estrutura do modelo foi uma boa escolha, porque deixou o código limpo e dispensou a necessidade de criar pipelines manuais mais complexos.
### 6️⃣ Exemplo de Inferência

Resultados da Inferência (Edge AI) 

Amostra 1: Predito = gato       | Real = gato
Amostra 2: Predito = navio      | Real = navio
Amostra 3: Predito = navio      | Real = navio
Amostra 4: Predito = avião      | Real = avião
Amostra 5: Predito = sapo       | Real = sapo

Nas 5 amostras testadas, o modelo otimizado .tflite acertou todas as previsões. Objetos com formatos bem marcados e fundos limpos (como navios no mar ou aviões no céu) foram identificados de primeira, mostrando que a conversão para 8 bits não afetou a capacidade do modelo em identificar essas classes.


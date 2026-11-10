[English](MODELS_GUIDE.md) · [Português (Brasil)](MODELS_GUIDE.pt-BR.md) · [Español](MODELS_GUIDE.es.md)

[← README](README.pt-BR.md)

# Modelos e otimização

Os regressores BATEIA-1 e seus scalers estão no [registro dos modelos](https://doi.org/10.5281/zenodo.24094057); o [dataset de pesquisa](https://doi.org/10.5281/zenodo.24087976) constitui outro registro. Para escolher artefatos, preparar entradas e realizar inferência, siga o [guia de dados próprios](CUSTOM_DATA.pt-BR.md). Pesos ImageNet das extratoras são distintos desses regressores treinados.

## Representações de características

As extratoras VGG, ResNet e DenseNet oferecem dois modos:

| Modo | Características salvas | Entrada sequencial |
| --- | --- | --- |
| GAP | `.pt` | Vetor global, sequência de comprimento 1 (`config.RESHAPE_CONFIG`) |
| Sem GAP | `_semgap.pt` | Mapa espacial 7 × 7, sequência de comprimento 49 (`config.RESHAPE_CONFIG_SEQ`) |

As 49 posições representam localizações espaciais, não medições físicas de tempo. Combine sufixo e extratora com a configuração do treinamento.

## Regressores

| Modelo | Implementação | Operação |
| --- | --- | --- |
| MLP | `models/architectures.py` | Camadas densas com não linearidades e dropout |
| LSTM / GRU | `models/architectures.py` | Predição pela saída do último passo da sequência |
| 1D-CNN | `models/architectures.py` | Convoluções 1D seguidas de `AdaptiveAvgPool1d(1)` |
| SVR / Random Forest | `models/core_trainer_sklearn.py` e módulos dos modelos | Regressão scikit-learn |

## Fluxo Optuna

1. Carregar características e alvos para a divisão por aglomerados escolhida.
2. Avaliar hiperparâmetros de `bateia/config_data/hyperparams.py`, minimizando o objetivo de validação (RMSE nas funções objetivo dos modelos).
3. Usar a melhor configuração no treinamento final e avaliar no teste selecionado. Preserve os scalers de entrada/saída junto ao modelo.

O menu atualmente define `config_geral_treino['n_trials']` como **1**. Aumente o orçamento configurado para uma busca mais ampla; iniciar o menu não garante múltiplos trials. As constantes de épocas e treinamento são configuradas separadamente.

## Escopo de avaliação

- **AM:** separação explícita de treino/validação/teste por aglomerado. A opção manual usa 15 para treino, 4 para validação e os restantes para teste; a estratificada permite escolher os de teste e divide o desenvolvimento em 70/30, com restrições de tamanho dos grupos.
- **BM:** usa todos os dados; não demonstra generalização para aglomerados inéditos.
- Um único aglomerado de teste não é requisito universal do menu atual.

O artigo avalia MLP e SVR; a dissertação/código incluem outros regressores. Consulte [uso](USAGE.pt-BR.md) para dados e [paralelismo](PARALELISMO-GPUS.md) para execução.

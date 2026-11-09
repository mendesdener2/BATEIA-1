# Guia de Modelos e Treinamento BATEIA 🧠

Este guia detalha as escolhas técnicas de arquitetura e a lógica de treinamento utilizada no projeto.

---

## 1. Global Average Pooling (GAP)
A maior distinção no treinamento dos modelos sequenciais (LSTM, GRU, 1D-CNN) é o tratamento do GAP:

### COM GAP (`config.RESHAPE_CONFIG`)
*   O extrator de características (backbone) colapsa as dimensões espaciais ($7 \times 7$) em um vetor global.
*   O modelo sequencial recebe uma **sequência de tamanho 1**.
*   Foco: Representação global e sintética da imagem.

### SEM GAP (`config.RESHAPE_CONFIG_SEQ`)
*   Mantemos o mapa de características espacial $7 \times 7$.
*   Achatamos as dimensões espaciais para criar uma **sequência de 49 passos**.
*   Foco: Relações espaciais tratadas como temporais pela rede recorrente.

---

## 2. Arquiteturas Suportadas

| Modelo | Implementação | Característica Principal |
| :--- | :--- | :--- |
| **LSTM / GRU** | `architectures.py` | Utiliza o estado oculto do último passo temporal para a predição final. |
| **1D-CNN** | `architectures.py` | Convoluções temporais seguidas de `AdaptiveAvgPool1d(1)` para extração de invariâncias. |
| **MLP** | `architectures.py` | Rede densa padrão para benchmarks de baixa complexidade. |
| **SVR / RF** | `core_trainer_sklearn.py` | Modelos clássicos usados como baseline estatístico. |

---

## 3. Fluxo de Otimização (Optuna)
O projeto utiliza o **Optuna** para busca de hiperparâmetros:
1.  **Trials:** Cada modelo é treinado múltiplas vezes com diferentes combinações de parâmetros (definidos em `bateia/config_data/hyperparams.py`).
2.  **Objetivo:** Minimizar o RMSE de validação no conjunto estratificado.
3.  **Finalização:** Os melhores parâmetros de cada rodada são salvos e usados para o treinamento do "Modelo Final" que será avaliado contra o conjunto de Teste Único (`n_test=1`).

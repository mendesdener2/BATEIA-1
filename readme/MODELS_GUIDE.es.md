[English](MODELS_GUIDE.md) · [Português (Brasil)](MODELS_GUIDE.pt-BR.md) · [Español](MODELS_GUIDE.es.md)

[← README](README.es.md)

# Modelos y optimización

Los regresores BATEIA-1 y sus scalers están en el [registro de modelos](https://doi.org/10.5281/zenodo.24094057); el [dataset de investigación](https://doi.org/10.5281/zenodo.24087976) es un registro separado. Para seleccionar artefactos, preparar entradas e inferir, siga la [guía de datos propios](CUSTOM_DATA.es.md). Los pesos ImageNet de extractores son distintos de estos regresores entrenados.

## Representaciones de características

Los extractores VGG, ResNet y DenseNet ofrecen dos modos:

| Modo | Características guardadas | Entrada secuencial |
| --- | --- | --- |
| GAP | `.pt` | Vector global, longitud de secuencia 1 (`config.RESHAPE_CONFIG`) |
| Sin GAP | `_semgap.pt` | Mapa espacial 7 × 7, longitud de secuencia 49 (`config.RESHAPE_CONFIG_SEQ`) |

Las 49 posiciones representan ubicaciones espaciales, no mediciones físicas de tiempo. Haga coincidir sufijo y extractor con la configuración de entrenamiento.

## Regresores

| Modelo | Implementación | Operación |
| --- | --- | --- |
| MLP | `models/architectures.py` | Capas densas con no linealidades y dropout |
| LSTM / GRU | `models/architectures.py` | Predicción desde la salida del último paso |
| 1D-CNN | `models/architectures.py` | Convoluciones 1D seguidas de `AdaptiveAvgPool1d(1)` |
| SVR / Random Forest | `models/core_trainer_sklearn.py` y módulos de modelos | Regresión scikit-learn |

## Flujo Optuna

1. Cargar características y etiquetas para la división por cúmulos elegida.
2. Evaluar hiperparámetros de `bateia/config_data/hyperparams.py`, minimizando el objetivo de validación (RMSE en las funciones objetivo).
3. Usar la mejor configuración para entrenamiento final y evaluar en la prueba seleccionada. Conserve los scalers de entrada/salida junto al modelo.

El menú define actualmente `config_geral_treino['n_trials']` como **1**. Aumente el presupuesto configurado para una búsqueda más amplia; iniciar el menú no garantiza múltiples trials. Las constantes de épocas y entrenamiento se configuran por separado.

## Alcance de evaluación

- **AM:** separación explícita por cúmulo entre entrenamiento, validación y prueba. La opción manual usa 15 de entrenamiento, 4 de validación y los restantes de prueba; la estratificada permite elegir los de prueba y divide el desarrollo en 70/30, con restricciones de tamaño de grupos.
- **BM:** usa todos los datos; no demuestra generalización a cúmulos no vistos.
- Un solo cúmulo de prueba no es un requisito universal del menú actual.

El artículo evalúa MLP y SVR; la tesis/código incluyen otros regresores. Consulte [uso](USAGE.es.md) para datos y [paralelismo](GPU_PARALLELISM.es.md) para ejecución.

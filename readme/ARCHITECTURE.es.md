[English](ARCHITECTURE.md) · [Português (Brasil)](ARQUITETURA.md) · [Español](ARCHITECTURE.es.md)

[← README](README.es.md)

# Arquitectura de BATEIA-1

El proyecto separa orquestación interactiva, procesamiento astrofísico, extracción visual y regresión.

## Entrada y entorno

- `bateia_exec.py`: menús de configuración, análisis, extracción, división por cúmulos, entrenamiento e informes.
- `environment.yml`: receta Conda; las versiones no están bloqueadas.

## Módulos centrales: `bateia/`

- `config.py`: raíz de datos (`BASE_DIR`, opcionalmente definida mediante `BATEIA_ROOT`), generación, remodelado de características y constantes de entrenamiento.
- `config_data/clusters.py`: metadatos de referencia de los cúmulos.
- `config_data/hyperparams.py`: espacios de búsqueda Optuna.
- `config_data/paper_data.py`: referencias para informes de la publicación.
- `real_clusters.py`, `synthetic_clusters.py`, `synthetic_analysis.py`: análisis real, generación y análisis sintético.
- `heatmaps_features.py`: extracción visual y mapas de calor de diagnóstico.

## Modelos y extractores

- `models/architectures.py`: definiciones PyTorch de MLP, LSTM, GRU y 1D-CNN.
- `models/core_trainer.py`: entrenamiento PyTorch e integración Optuna.
- `models/core_trainer_sklearn.py`: entrenamiento de regresores scikit-learn.
- `models/*_model.py`: entrenamiento y funciones objetivo específicos de cada modelo.
- `models/registry.py`, `features/registry.py`: registro y metadatos de modelos/extractores.

## Utilidades: `bateia/tools/`

- `conversor_dados.py`, `gdrive_downloader.py`, `verificador.py`: conversión, descargas autenticadas y comprobación local.
- `data_loader.py`, `inference.py`: carga de características/etiquetas, artefactos guardados e inferencia.
- `paper_utils.py`, `results_bateia.py`: tablas de publicación y resúmenes de resultados.
- `correction.py`, `verify_split.py`: mantenimiento de métricas e inspección de divisiones.
- `reset.py`: limpieza de datos/modelos generados; las operaciones de reinicio pueden eliminar salidas.

## Lecturas siguientes

Consulte [instalación y uso](USAGE.es.md), [modelos](MODELS_GUIDE.es.md) y [paralelismo](GPU_PARALLELISM.es.md). Existen módulos internos, pero la interfaz documentada sigue siendo el menú de terminal; no hay una API simplificada para notebooks implementada.

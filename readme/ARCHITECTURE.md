[English](ARCHITECTURE.md) · [Português (Brasil)](ARQUITETURA.md) · [Español](ARCHITECTURE.es.md)

[← README](../README.md)

# BATEIA-1 architecture

The project separates interactive orchestration, astrophysical processing, visual feature extraction and regression.

## Entry point and environment

- `bateia_exec.py`: interactive menus for data setup, analysis, feature extraction, cluster splits, training and reports.
- `environment.yml`: Conda environment recipe; dependency versions are not locked.

## Core modules: `bateia/`

- `config.py`: data root (`BASE_DIR`, optionally set through `BATEIA_ROOT`), generation settings, feature reshaping and training constants.
- `config_data/clusters.py`: cluster reference metadata.
- `config_data/hyperparams.py`: Optuna search spaces.
- `config_data/paper_data.py`: reference data used for publication reports.
- `real_clusters.py`, `synthetic_clusters.py`, `synthetic_analysis.py`: observed-cluster analysis, synthetic generation and synthetic analysis.
- `heatmaps_features.py`: visual feature extraction and diagnostic heatmaps.

## Models and extractors

- `models/architectures.py`: PyTorch MLP, LSTM, GRU and 1D-CNN definitions.
- `models/core_trainer.py`: PyTorch training and Optuna integration.
- `models/core_trainer_sklearn.py`: training for scikit-learn regressors.
- `models/*_model.py`: model-specific training and objective functions.
- `models/registry.py`, `features/registry.py`: model and extractor metadata/registration.

## Utilities: `bateia/tools/`

- `conversor_dados.py`, `gdrive_downloader.py`, `verificador.py`: conversion, authenticated downloads and local-data checks.
- `data_loader.py`, `inference.py`: feature/target loading, saved artifacts and inference utilities.
- `paper_utils.py`, `results_bateia.py`: publication tables and result summaries.
- `correction.py`, `verify_split.py`: metric maintenance and split inspection.
- `reset.py`: generated-data/model cleanup; reset operations can delete outputs.

## Further reading

See [installation and usage](USAGE.md), [models](MODELS_GUIDE.md) and [parallelism](GPU_PARALLELISM.md). Internal modules exist, but the documented interface remains the terminal menu; a simplified notebook API is not implemented.

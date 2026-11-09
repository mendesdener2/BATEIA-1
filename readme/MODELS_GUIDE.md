[English](MODELS_GUIDE.md) · [Português (Brasil)](MODELS_GUIDE.pt-BR.md) · [Español](MODELS_GUIDE.es.md)

[← README](../README.md)

# Models and optimization

## Feature representations

The VGG, ResNet and DenseNet extractors provide two feature modes:

| Mode | Saved features | Sequential input |
| --- | --- | --- |
| GAP | `.pt` | Global vector, sequence length 1 (`config.RESHAPE_CONFIG`) |
| No GAP | `_semgap.pt` | Spatial 7 × 7 map, sequence length 49 (`config.RESHAPE_CONFIG_SEQ`) |

The 49 positions represent spatial locations, not physical time measurements. Match the feature suffix and extractor to the training configuration.

## Regressors

| Model | Implementation | Operation |
| --- | --- | --- |
| MLP | `models/architectures.py` | Dense layers with nonlinearities and dropout |
| LSTM / GRU | `models/architectures.py` | Prediction from the last sequence-step output |
| 1D-CNN | `models/architectures.py` | 1D convolutions followed by `AdaptiveAvgPool1d(1)` |
| SVR / Random Forest | `models/core_trainer_sklearn.py` and model modules | scikit-learn regression |

## Optuna workflow

1. Load features and target values for the chosen cluster split.
2. Evaluate hyperparameters from `bateia/config_data/hyperparams.py`, minimizing the validation objective (RMSE in the model objectives).
3. Use the best configuration for final training and evaluate on the selected test set. Preserve the corresponding input/output scalers alongside the model.

The menu currently sets `config_geral_treino['n_trials']` to **1**. Increase the configured budget if a broader search is intended; multiple trials are not guaranteed by simply starting the menu. Training and epoch constants are configured separately.

## Evaluation scope

- **AM:** explicit cluster-level train/validation/test separation. The manual option uses 15 training and 4 validation clusters, with remaining clusters assigned to test; the stratified option lets you select test clusters and uses a 70/30 development split, subject to group-size constraints.
- **BM:** uses all data; results do not establish generalization to unseen clusters.
- A single test cluster is not a universal requirement of the current menu.

The article evaluates MLP and SVR; the dissertation/source include further regressors. See [usage](USAGE.md) for data requirements and [parallelism](GPU_PARALLELISM.md) for execution settings.

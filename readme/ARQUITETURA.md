[English](ARCHITECTURE.md) · [Português (Brasil)](ARQUITETURA.md) · [Español](ARCHITECTURE.es.md)

[← README](README.pt-BR.md)

# Arquitetura do BATEIA-1

O projeto separa orquestração interativa, processamento astrofísico, extração de características visuais e regressão.

## Entrada e ambiente

- `bateia_exec.py`: menus de configuração, análise, extração, divisão por aglomerados, treinamento e relatórios.
- `environment.yml`: receita de ambiente Conda; as versões não estão travadas.

## Módulos centrais: `bateia/`

- `config.py`: raiz dos dados (`BASE_DIR`, opcionalmente definida por `BATEIA_ROOT`), geração, remodelagem das características e constantes de treinamento.
- `config_data/clusters.py`: metadados de referência dos aglomerados.
- `config_data/hyperparams.py`: espaços de busca Optuna.
- `config_data/paper_data.py`: referências usadas nos relatórios da publicação.
- `real_clusters.py`, `synthetic_clusters.py`, `synthetic_analysis.py`: análise real, geração e análise sintética.
- `heatmaps_features.py`: extração visual e mapas de calor de diagnóstico.

## Modelos e extratoras

- `models/architectures.py`: definições PyTorch de MLP, LSTM, GRU e 1D-CNN.
- `models/core_trainer.py`: treinamento PyTorch e integração Optuna.
- `models/core_trainer_sklearn.py`: treinamento de regressores scikit-learn.
- `models/*_model.py`: treinamento e funções objetivo específicos dos modelos.
- `models/registry.py`, `features/registry.py`: registro e metadados de modelos/extratoras.

## Utilitários: `bateia/tools/`

- `conversor_dados.py`, `gdrive_downloader.py`, `verificador.py`: conversão, downloads autenticados e verificação local.
- `data_loader.py`, `inference.py`: carregamento de características/alvos, artefatos salvos e inferência.
- `paper_utils.py`, `results_bateia.py`: tabelas da publicação e resumos de resultados.
- `correction.py`, `verify_split.py`: manutenção de métricas e inspeção das divisões.
- `reset.py`: limpeza de dados/modelos gerados; operações de reset podem apagar saídas.

## Próximas leituras

Consulte [instalação e uso](USAGE.pt-BR.md), [modelos](MODELS_GUIDE.pt-BR.md) e [paralelismo](PARALELISMO-GPUS.md). Existem módulos internos, mas a interface documentada continua sendo o menu no terminal; uma API simplificada para notebooks não está implementada.

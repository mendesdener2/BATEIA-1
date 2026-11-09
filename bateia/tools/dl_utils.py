# bateia/tools/dl_utils.py
"""
Módulo de Utilitários de Deep Learning para o projeto bateia.
(Refatorado: As funções foram delegadas para módulos específicos para aderir ao SRP)
"""

from .data_loader import Logger, load_data, salvar_resultados_finais
from .trainer_backend import train_model_pytorch, test_model_pytorch, gerar_relatorio_consistencia
from .inference import predict_on_real_clusters, predict_on_real_clusters_sklearn

__all__ = [
    'Logger',
    'load_data',
    'salvar_resultados_finais',
    'train_model_pytorch',
    'test_model_pytorch',
    'gerar_relatorio_consistencia',
    'predict_on_real_clusters',
    'predict_on_real_clusters_sklearn'
]

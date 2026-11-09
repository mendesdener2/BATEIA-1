# bateia/models/__init__.py
"""
Módulo de Modelos do Projeto BATEIA.

Este módulo exporta todos os modelos registrados e o registry central.
"""

from .registry import (
    ModelRegistry,
    BaseModel,
    ModelInfo
)

# Importa os modelos existentes para garantir registro
# A ordem de importação é importante
from bateia.models import mlp_model
from bateia.models import svr_model
from bateia.models import randomforest_model
from bateia.models import lstm_model
from bateia.models import gru_model
from bateia.models import cnn1d_model

# Exporta as funções de treino dos modelos existentes para o registry
# Isso permite que o ModelRegistry tenha acesso às funções reais
ModelRegistry.register_function("MLP", mlp_model.executar_treino_mlp)
ModelRegistry.register_function("SVR", svr_model.executar_treino_svr)
ModelRegistry.register_function("RandomForest", randomforest_model.executar_treino_random_forest)
ModelRegistry.register_function("LSTM", lstm_model.executar_treino_lstm)
ModelRegistry.register_function("GRU", gru_model.executar_treino_gru)
ModelRegistry.register_function("1DCNN", cnn1d_model.executar_treino_1dcnn)

__all__ = [
    "ModelRegistry",
    "BaseModel", 
    "ModelInfo",
    "mlp_model",
    "svr_model", 
    "randomforest_model",
    "lstm_model",
    "gru_model",
    "cnn1d_model",
]

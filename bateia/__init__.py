# bateia/__init__.py
"""
Projeto BATEIA-1 - Machine Learning para Análise de Aglomerados Estelares

Este módulo exporta os principais componentes do projeto:
- Modelos de regressão (MLP, SVR, LSTM, etc)
- Extratoras de features (VGG, ResNet, DenseNet)
- Ferramentas de processamento
"""

# Models
from bateia.models import ModelRegistry, BaseModel, ModelInfo

# Features (Extratoras)
from bateia.features import ExtractorRegistry, ExtractorInfo

__version__ = "1.0.0"
__author__ = "BATEIA Team"

__all__ = [
    "ModelRegistry",
    "BaseModel",
    "ModelInfo",
    "ExtractorRegistry", 
    "ExtractorInfo",
]

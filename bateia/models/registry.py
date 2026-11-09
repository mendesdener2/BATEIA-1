# bateia/models/registry.py
"""
Sistema de Registro Centralizado de Modelos BATEIA.

Este módulo fornece um registry central para todos os modelos de regressão.
Para adicionar um novo modelo, basta usar o decorator @register.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Classe base abstrata que define a interface padrão para todos os modelos.
    Não é obrigatória sua implementação, mas é recomendada para padronização.
    """
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Treina o modelo com os dados fornecidos."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Faz predições com o modelo treinado."""
        pass
    
    def save(self, path: str):
        """Salva o modelo em disco."""
        pass
    
    def load(self, path: str):
        """Carrega o modelo do disco."""
        pass


class ModelInfo:
    """Armazena informações sobre um modelo registrado."""
    
    def __init__(self, name: str, hyperparams: Optional[Dict[str, Any]] = None,
                 description: str = "", needs_gpu: bool = False):
        self.name = name
        self.hyperparams = hyperparams or {}
        self.description = description
        self.needs_gpu = needs_gpu


class ModelRegistry:
    """
    Registry centralizado para modelos de regressão do BATEIA.
    
    Uso:
        @ModelRegistry.register("NomeDoModelo", hyperparams={...})
        class MeuModelo:
            ...
    
    Ou simplesmente registrar uma função existente:
        ModelRegistry.register_function("MLP", mlp_model.executar_treino_mlp, {...})
    """
    
    _models: Dict[str, ModelInfo] = {}
    _functions: Dict[str, Any] = {}
    _classes: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, hyperparams: Optional[Dict[str, Any]] = None,
                 description: str = "", needs_gpu: bool = False):
        """
        Decorator para registrar um modelo.
        
        Args:
            name: Nome único do modelo (ex: "MLP", "LSTM", "Transformer")
            hyperparams: Dicionário com configuração do Optuna para esse modelo
            description: Descrição breve do modelo
            needs_gpu: Se o modelo requer GPU para treino
        """
        def decorator(model_class_or_func):
            cls._models[name] = ModelInfo(name, hyperparams, description, needs_gpu)
            if callable(model_class_or_func) and not isinstance(model_class_or_func, type):
                cls._functions[name] = model_class_or_func
            elif isinstance(model_class_or_func, type) and issubclass(model_class_or_func, BaseModel):
                cls._classes[name] = model_class_or_func
            logger.info(f"Modelo registrado: {name}")
            return model_class_or_func
        return decorator
    
    @classmethod
    def register_function(cls, name: str, func: Any, 
                         hyperparams: Optional[Dict[str, Any]] = None,
                         description: str = "", needs_gpu: bool = False):
        """
        Registra uma função de treino existente.
        
        Args:
            name: Nome do modelo
            func: Função de treino existente
            hyperparams: Configuração do Optuna
            description: Descrição
            needs_gpu: Requer GPU
        """
        cls._models[name] = ModelInfo(name, hyperparams, description, needs_gpu)
        cls._functions[name] = func
        logger.info(f"Função registrada: {name}")
    
    @classmethod
    def register_class(cls, name: str, model_class: Type[BaseModel],
                       hyperparams: Optional[Dict[str, Any]] = None,
                       description: str = "", needs_gpu: bool = False):
        """
        Registra uma classe de modelo.
        """
        cls._models[name] = ModelInfo(name, hyperparams, description, needs_gpu)
        cls._classes[name] = model_class
        logger.info(f"Classe registrada: {name}")
    
    @classmethod
    def get(cls, name: str) -> Any:
        """
        Retorna a função ou classe do modelo pelo nome.
        
        Raises:
            ValueError: Se o modelo não existir
        """
        if name not in cls._models:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Modelo '{name}' não encontrado.\n"
                f"Modelos disponíveis: {available}"
            )
        
        if name in cls._classes:
            return cls._classes[name]
        return cls._functions.get(name)
    
    @classmethod
    def get_info(cls, name: str) -> Optional[ModelInfo]:
        """Retorna as informações de um modelo."""
        return cls._models.get(name)
    
    @classmethod
    def get_hyperparams(cls, name: str) -> Dict[str, Any]:
        """Retorna a configuração de hiperparâmetros do modelo."""
        info = cls._models.get(name)
        return info.hyperparams if info else {}
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Lista todos os nomes de modelos registrados."""
        return sorted(cls._models.keys())
    
    @classmethod
    def list_gpu_models(cls) -> List[str]:
        """Lista modelos que requerem GPU."""
        return sorted([name for name, info in cls._models.items() if info.needs_gpu])
    
    @classmethod
    def list_cpu_models(cls) -> List[str]:
        """Lista modelos que rodam em CPU."""
        return sorted([name for name, info in cls._models.items() if not info.needs_gpu])
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """Verifica se um modelo está registrado."""
        return name in cls._models
    
    @classmethod
    def get_all_hyperparams(cls) -> Dict[str, Dict[str, Any]]:
        """Retorna todos os hiperparâmetros registrados."""
        return {name: info.hyperparams for name, info in cls._models.items() if info.hyperparams}


# ==============================================================================
# REGISTRO DOS MODELOS EXISTENTES
# ==============================================================================

# Os modelos existentes são registrados aqui com seus hiperparâmetros
# As funções reais permanecem nos seus arquivos originais

from bateia.config import OPTUNA_HYPERPARAMETERS

# MLP
ModelRegistry.register_function(
    "MLP",
    None,  # Função será importada dinamicamente
    hyperparams=OPTUNA_HYPERPARAMETERS.get("MLP_f0"),  # Usa f0 como base
    description="Multi-Layer Perceptron com camadas densas",
    needs_gpu=True
)

# SVR
ModelRegistry.register_function(
    "SVR",
    None,
    hyperparams=OPTUNA_HYPERPARAMETERS.get("SVR"),
    description="Support Vector Regression",
    needs_gpu=False
)

# RandomForest
ModelRegistry.register_function(
    "RandomForest",
    None,
    hyperparams=OPTUNA_HYPERPARAMETERS.get("RandomForest"),
    description="Random Forest Regressor",
    needs_gpu=False
)

# LSTM
ModelRegistry.register_function(
    "LSTM",
    None,
    hyperparams=OPTUNA_HYPERPARAMETERS.get("LSTM_f0"),
    description="Long Short-Term Memory Network",
    needs_gpu=True
)

# GRU
ModelRegistry.register_function(
    "GRU",
    None,
    hyperparams=OPTUNA_HYPERPARAMETERS.get("GRU_f0"),
    description="Gated Recurrent Unit",
    needs_gpu=True
)

# 1DCNN
ModelRegistry.register_function(
    "1DCNN",
    None,
    hyperparams=OPTUNA_HYPERPARAMETERS.get("1DCNN_f0"),
    description="1-Dimensional Convolutional Neural Network",
    needs_gpu=True
)

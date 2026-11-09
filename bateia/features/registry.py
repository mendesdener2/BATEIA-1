# bateia/features/registry.py
"""
Sistema de Registro Centralizado de Extratoras de Features (CNNs).

Este módulo fornece um registry para extratoras de features baseadas em
redes neurais convolucionais pré-treinadas (VGG, ResNet, DenseNet, EfficientNet, ViT, etc).

Para adicionar uma nova extratora, basta usar o decorator @register com a configuração apropriada.

Autor: BATEIA Team
Versão: 1.0.0
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from torchvision import models
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ExtractorInfo:
    """
    Armazena informações completas sobre uma extratora registrada.
    
    Atributos:
        name: Nome único da extratora
        family: Família da rede (VGG, RESNET, DENSENET, etc)
        model_func: Função que retorna a instância do modelo
        output_features: Número de features de saída
        weights: Pesos pré-treinados a usar
        target_layer_name: Nome da camada-alvo para XAI (ex: 'layer4', 'features[-1]')
        input_size: Tamanho de entrada esperado (default: 224)
        preprocessing: Dicionário com mean e std para normalização
        description: Descrição breve do modelo
    """
    
    def __init__(
        self, 
        name: str, 
        family: str, 
        model_func: Callable,
        output_features: int = 512, 
        weights: str = "IMAGENET1K_V1",
        target_layer_name: str = "",
        input_size: int = 224,
        preprocessing: Optional[Dict[str, Any]] = None,
        description: str = ""
    ):
        self.name = name
        self.family = family
        self.model_func = model_func
        self.output_features = output_features
        self.weights = weights
        self.target_layer_name = target_layer_name
        self.input_size = input_size
        self.preprocessing = preprocessing or {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        self.description = description


class ExtractorRegistry:
    """
    Registry centralizado para extratoras de features do BATEIA.
    
    Uso - Para registrar uma nova extratora:
        
        @ExtractorRegistry.register(
            name="MinhaExtratora",
            family="MINHA_FAMILIA",
            output_features=512,
            target_layer_name="minha_camada",
            description="Minha nova extratora"
        )
        def get_minha_extratora():
            return models.minha_extratora(weights='IMAGENET1K_V1')
    
    Ou simplesmente listar os modelos disponíveis:
        ExtractorRegistry.list_all()
        ExtractorRegistry.list_by_family("VGG")
    """
    
    _extractors: Dict[str, ExtractorInfo] = {}
    _families: Dict[str, List[str]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        family: str = "Other",
        output_features: int = 512,
        weights: str = "IMAGENET1K_V1",
        target_layer_name: str = "",
        input_size: int = 224,
        preprocessing: Optional[Dict[str, Any]] = None,
        description: str = ""
    ):
        """
        Decorator para registrar uma extratora.
        
        Args:
            name: Nome único (ex: "VGG16", "EfficientNetB0")
            family: Família (ex: "VGG", "RESNET", "EfficientNet")
            output_features: Número de features de saída
            weights: Pesos ('IMAGENET1K_V1', 'DEFAULT', etc)
            target_layer_name: Nome da camada para XAI (opcional)
            input_size: Tamanho de entrada (default 224)
            preprocessing: Dicionário com 'mean' e 'std'
            description: Descrição do modelo
        """
        def decorator(model_func):
            info = ExtractorInfo(
                name=name,
                family=family,
                model_func=model_func,
                output_features=output_features,
                weights=weights,
                target_layer_name=target_layer_name,
                input_size=input_size,
                preprocessing=preprocessing,
                description=description
            )
            cls._extractors[name] = info
            cls._families.setdefault(family, []).append(name)
            logger.info(f"Extratora registrada: {name} (família: {family}, features: {output_features})")
            return model_func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> ExtractorInfo:
        """
        Retorna as informações da extratora pelo nome.
        
        Raises:
            ValueError: Se a extratora não existir
        """
        if name not in cls._extractors:
            available = ", ".join(cls.list_all())
            raise ValueError(
                f"Extratora '{name}' não encontrada.\n"
                f"Extratoras disponíveis: {available}"
            )
        return cls._extractors[name]
    
    @classmethod
    def get_model(cls, name: str) -> nn.Module:
        """Retorna uma instância do modelo."""
        info = cls.get(name)
        return info.model_func()
    
    @classmethod
    def get_output_features(cls, name: str) -> int:
        """Retorna o número de features de saída."""
        return cls.get(name).output_features
    
    @classmethod
    def get_target_layer(cls, model: nn.Module, name: str) -> List[nn.Module]:
        """
        Retorna a camada-alvo para XAI (Grad-CAM, etc).
        
        Args:
            model: Instância do modelo
            name: Nome da extratora
            
        Returns:
            Lista com a camada-alvo
        """
        info = cls.get(name)
        family = info.family
        
        # Mapeamento padrão de famílias para camadas-alvo
        if "VGG" in family:
            return [model.features[-1]]
        elif "RESNET" in family:
            return [model.layer4[-1]]
        elif "DENSENET" in family:
            return [model.features.norm5]
        elif "EFFICIENTNET" in family:
            return [model.features[-1]]
        elif "VIT" in family:
            return [model.encoder.layer[-1]]
        elif "MOBILENET" in family:
            return [model.features[-1]]
        
        # Fallback para casos não mapeados
        if info.target_layer_name:
            parts = info.target_layer_name.split('.')
            layer = model
            for part in parts:
                layer = getattr(layer, part)
            return [layer]
        
        # Padrão genérico
        return [model]
    
    @classmethod
    def list_all(cls) -> List[str]:
        """Lista todas as extratoras registradas."""
        return sorted(cls._extractors.keys())
    
    @classmethod
    def list_by_family(cls, family: str) -> List[str]:
        """Lista extratoras de uma família específica."""
        return sorted(cls._families.get(family, []))
    
    @classmethod
    def list_families(cls) -> List[str]:
        """Lista todas as famílias registradas."""
        return sorted(cls._families.keys())
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """Verifica se uma extratora está registrada."""
        return name in cls._extractors
    
    @classmethod
    def get_family(cls, name: str) -> Optional[str]:
        """Retorna a família de uma extratora."""
        return cls.get(name).family
    
    @classmethod
    def get_all_by_family(cls) -> Dict[str, List[str]]:
        """Retorna dicionário com todas as famílias e suas extratoras."""
        return dict(cls._families)
    
    @classmethod
    def get_preprocessing(cls, name: str) -> Dict[str, Any]:
        """Retorna configuração de preprocessing."""
        return cls.get(name).preprocessing
    
    @classmethod
    def create_preprocess_transform(cls, name: str) -> nn.Module:
        """
        Cria o transforms.Compose para preprocessing.
        
        Args:
            name: Nome da extratora
            
        Returns:
            transforms.Compose configurado
        """
        from torchvision import transforms
        info = cls.get(name)
        
        return transforms.Compose([
            transforms.Resize((info.input_size, info.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=info.preprocessing["mean"],
                std=info.preprocessing["std"]
            ),
        ])


# ==============================================================================
# REGISTRO DAS EXTRATORAS PADRÃO (IMAGENET)
# ==============================================================================

@ExtractorRegistry.register(
    name="VGG13", 
    family="VGG", 
    output_features=512,
    target_layer_name="features[-1]",
    description="VGG-13 with batch normalization"
)
def get_vgg13():
    return models.vgg13_bn(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="VGG16", 
    family="VGG", 
    output_features=512,
    target_layer_name="features[-1]",
    description="VGG-16 with batch normalization"
)
def get_vgg16():
    return models.vgg16_bn(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="VGG19", 
    family="VGG", 
    output_features=512,
    target_layer_name="features[-1]",
    description="VGG-19 with batch normalization"
)
def get_vgg19():
    return models.vgg19_bn(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="RESNET50", 
    family="RESNET", 
    output_features=2048,
    target_layer_name="layer4[-1]",
    description="ResNet-50"
)
def get_resnet50():
    return models.resnet50(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="RESNET101", 
    family="RESNET", 
    output_features=2048,
    target_layer_name="layer4[-1]",
    description="ResNet-101"
)
def get_resnet101():
    return models.resnet101(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="RESNET152", 
    family="RESNET", 
    output_features=2048,
    target_layer_name="layer4[-1]",
    description="ResNet-152"
)
def get_resnet152():
    return models.resnet152(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="DENSENET121", 
    family="DENSENET", 
    output_features=1024,
    target_layer_name="features.norm5",
    description="DenseNet-121"
)
def get_densenet121():
    return models.densenet121(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="DENSENET161", 
    family="DENSENET", 
    output_features=2208,
    target_layer_name="features.norm5",
    description="DenseNet-161"
)
def get_densenet161():
    return models.densenet161(weights='IMAGENET1K_V1')

@ExtractorRegistry.register(
    name="DENSENET201", 
    family="DENSENET", 
    output_features=1920,
    target_layer_name="features.norm5",
    description="DenseNet-201"
)
def get_densenet201():
    return models.densenet201(weights='IMAGENET1K_V1')


# ==============================================================================
# EXEMPLOS: Como adicionar novas extratoras (comentado para referência)
# ==============================================================================

# -------------------------------------------------------------------------
# Exemplo 1: EfficientNet (precisa de torchvision >= 0.11)
# -------------------------------------------------------------------------
# @ExtractorRegistry.register(
#     name="EfficientNetB0",
#     family="EFFICIENTNET",
#     output_features=1280,
#     target_layer_name="features[-1]",
#     input_size=224,
#     description="EfficientNet-B0"
# )
# def get_efficientnet_b0():
#     return models.efficientnet_b0(weights='IMAGENET1K_V1')

# @ExtractorRegistry.register(
#     name="EfficientNetB4",
#     family="EFFICIENTNET",
#     output_features=1792,
#     target_layer_name="features[-1]",
#     input_size=380,
#     description="EfficientNet-B4"
# )
# def get_efficientnet_b4():
#     return models.efficientnet_b4(weights='IMAGENET1K_V1')

# -------------------------------------------------------------------------
# Exemplo 2: Vision Transformer (precisa de torchvision >= 0.12)
# -------------------------------------------------------------------------
# @ExtractorRegistry.register(
#     name="ViT-B-16",
#     family="VIT",
#     output_features=768,
#     target_layer_name="encoder.layer[-1]",
#     input_size=224,
#     description="Vision Transformer Base (16x16 patch)"
# )
# def get_vit_b_16():
#     return models.vit_b_16(weights='IMAGENET1K_V1')

# -------------------------------------------------------------------------
# Exemplo 3: MobileNet (leve, ideal para dispositivos-limited)
# -------------------------------------------------------------------------
# @ExtractorRegistry.register(
#     name="MobileNetV3-Large",
#     family="MOBILENET",
#     output_features=960,
#     target_layer_name="features[-1]",
#     input_size=224,
#     description="MobileNetV3-Large"
# )
# def get_mobilenet_v3_large():
#     return models.mobilenet_v3_large(weights='IMAGENET1K_V1')

# -------------------------------------------------------------------------
# Exemplo 4: ConvNeXt (state-of-the-art moderno)
# -------------------------------------------------------------------------
# @ExtractorRegistry.register(
#     name="ConvNeXt-Tiny",
#     family="CONVNEXT",
#     output_features=768,
#     target_layer_name="features[-1]",
#     input_size=224,
#     description="ConvNeXt-Tiny"
# )
# def get_convnext_tiny():
#     return models.convnext_tiny(weights='IMAGENET1K_V1')

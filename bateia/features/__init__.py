# bateia/features/__init__.py
"""
Módulo de Extratoras de Features do Projeto BATEIA.

Este módulo exporta todas as extratoras registradas e o registry central.
"""

from .registry import (
    ExtractorRegistry,
    ExtractorInfo
)

__all__ = [
    "ExtractorRegistry",
    "ExtractorInfo",
]

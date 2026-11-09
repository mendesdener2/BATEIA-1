# bateia/config.py
"""
Arquivo de Configuração Central do Projeto BATEIA.

Design:
    - Bloco 1: Imports leves (stdlib) + resolução de BASE_DIR
    - Bloco 2: Constantes puras de configuração
    - Bloco 3: Imports de dados internos do projeto
    - Bloco 4: Imports pesados (torch, matplotlib, sklearn, etc.)
              carregados no topo do módulo para manter compatibilidade
              com `from bateia.config import *`, mas isolados logicamente.
"""

# ==============================================================================
# BLOCO 1: IMPORTS LEVES E RESOLUÇÃO DE CAMINHOS (CROSS-PLATFORM)
# ==============================================================================
import os
import sys
import logging
from pathlib import Path


def _resolve_base_dir() -> str:
    """
    Resolve o diretório base do projeto de forma multiplataforma.

    Prioridade:
        1. Variável de ambiente BATEIA_ROOT (para CI/CD ou máquinas de terceiros)
        2. Caminho relativo à raiz do repositório (padrão para desenvolvimento)

    Returns:
        String com o caminho absoluto do diretório 'database'.
    """
    env_root = os.environ.get("BATEIA_ROOT")
    if env_root:
        base = Path(env_root)
    else:
        # bateia/config.py → bateia/ → bateia-doc/ (raiz do repo)
        repo_root = Path(__file__).resolve().parent.parent
        base = repo_root / "database"

    if not base.exists():
        logging.warning(
            f"BASE_DIR não encontrado em '{base}'. "
            f"Defina a variável de ambiente BATEIA_ROOT ou crie o diretório."
        )

    return str(base)


BASE_DIR = _resolve_base_dir()

# ==============================================================================
# BLOCO 2: CONSTANTES PURAS DE CONFIGURAÇÃO
# ==============================================================================

# --- Dados Internos do Projeto ---
from .config_data.clusters import PARAMETROS_AJUSTE
from .config_data.hyperparams import OPTUNA_HYPERPARAMETERS

# --- Geração de Imagem e Dados Sintéticos ---
FORMATO_IMAGEM = 'jpg'
TAMANHO_FONTE = 18
IMAGE_SIZE_PX = 512
DPI = 100

T_MAX_SINTETICOS = 1000
SYNTHETIC_RANGES = {
    (1, 250): (-0.05, 0.05), (251, 500): (-0.10, 0.10),
    (501, 750): (-0.15, 0.15), (751, 1000): (-0.20, 0.20)
}
SCALE_VARIATION = 0

# --- XAI (Explainable AI) ---
# NOTA: HEATMAP_TYPES_* dependem de pytorch_grad_cam, importados abaixo.

# --- Treinamento ---
TRAINING_CONSTANTS = {
    "optuna_epochs": 20,  # Épocas para cada trial do Optuna = 150
    "final_epochs": 20,   # Épocas para o treino do modelo final com melhores params = 300
}

N_JOBS_ML = -1
OPTUNA_N_JOBS_CPU = N_JOBS_ML

# --- Early Stopping por Parâmetro ---
PARAMS_CONFIG = {
    "f0":  {"patience": 15, "min_delta": 0, "final_patience": 20},
    "Rc":  {"patience": 15, "min_delta": 0, "final_patience": 20},
    "Rt":  {"patience": 15, "min_delta": 0, "final_patience": 20},
    "fbg": {"patience": 15, "min_delta": 0, "final_patience": 20},
}

# --- Reshape para Modelos Sequenciais (GAP: vetor 1D) ---
RESHAPE_CONFIG = {
    512:  (1, 512),    # VGGs
    2048: (1, 2048),   # ResNets
    1024: (1, 1024),   # DenseNet121
    2208: (1, 2208),   # DenseNet161
    1920: (1, 1920),   # DenseNet201
}

# --- Reshape para Modelos Sequenciais (SEM GAP: sequência espacial 7x7) ---
RESHAPE_CONFIG_SEQ = {
    512 * 7 * 7:  (49, 512),    # VGGs
    2048 * 7 * 7: (49, 2048),   # ResNets
    1024 * 7 * 7: (49, 1024),   # DenseNet121
    2208 * 7 * 7: (49, 2208),   # DenseNet161
    1920 * 7 * 7: (49, 1920),   # DenseNet201
}

# --- Heatmaps ---
SINTETICOS_PARA_HEATMAPS = [1, 200, 400, 600, 800, 1000]
HEATMAP_FONT_TITLE = 26
HEATMAP_FONT_LABELS = 18
HEATMAP_FONT_TICKS = 18
HEATMAP_FONT_COLORBAR_LABEL = 18
HEATMAP_OUTPUT_SIZE = 512

# ==============================================================================
# BLOCO 3: API DE CARREGAMENTO LAZY (BIBLIOTECAS PESADAS)
# ==============================================================================

def get_heavy_imports():
    """
    Retorna as bibliotecas pesadas carregadas sob demanda.
    Útil para scripts que precisam de performance no startup.
    """
    import torch
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import ScoreCAM, AblationCAM
    
    return {
        'torch': torch,
        'cv2': cv2,
        'np': np,
        'pd': pd,
        'plt': plt,
        'ScoreCAM': ScoreCAM,
        'AblationCAM': AblationCAM
    }

# NOTA: Para manter a compatibilidade com o código atual que importa estas constantes,
# elas permanecem no topo mas agora são definidas como propriedades ou dicionários 
# carregados sob demanda em tempo de execução pelos módulos que as utilizam.
# No entanto, como o objetivo é profissionalismo, os módulos downstream (heatmaps)
# devem preferencialmente usar as classes diretamente.

def get_heatmap_types():
    from pytorch_grad_cam import ScoreCAM, AblationCAM
    return {
        'grad': {'Score-CAM': ScoreCAM},
        'grad_free': {'Ablation-CAM': AblationCAM},
        'comparison': {'Score-CAM': ScoreCAM, 'Ablation-CAM': AblationCAM}
    }

# Constantes para compatibilidade legada (serão removidas em versões futuras)
# Atualmente heatmaps_features.py ainda as importa.
HEATMAP_TYPES_GRAD = { 'Score-CAM': None } # Placeholder para evitar erro de import
HEATMAP_TYPES_GRAD_FREE = { 'Ablation-CAM': None }

def _fix_heatmap_placeholders():
    """Resolve os placeholders de heatmaps (chamado internamente)."""
    global HEATMAP_TYPES_GRAD, HEATMAP_TYPES_GRAD_FREE
    from pytorch_grad_cam import ScoreCAM, AblationCAM
    HEATMAP_TYPES_GRAD['Score-CAM'] = ScoreCAM
    HEATMAP_TYPES_GRAD_FREE['Ablation-CAM'] = AblationCAM

# Inicialização tardia se necessário, ou deixar para os módulos chamarem.
# Para este projeto, vamos manter as definições mas os módulos importam as libs.

from bateia.models.architectures import *

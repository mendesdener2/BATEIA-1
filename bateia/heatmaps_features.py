# BATEIA/heatmaps_features.py
# ----------------------------------------------
# Refatorado para usar ExtractorRegistry
# ----------------------------------------------

# --- Importações Centralizadas ---
import os
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pytorch_grad_cam import ScoreCAM, AblationCAM
from .config import (
    BASE_DIR, T_MAX_SINTETICOS, SINTETICOS_PARA_HEATMAPS, DPI
)

# Definições de métodos GradCAM (Movidas do config para cá por serem específicas deste módulo)
HEATMAP_TYPES_GRAD = {'Score-CAM': ScoreCAM}
HEATMAP_TYPES_GRAD_FREE = {'Ablation-CAM': AblationCAM}
from .features.registry import ExtractorRegistry

# Configurações globais de bibliotecas
plt.switch_backend('Agg')
cv2.setNumThreads(0) 
torch.set_num_threads(4)

# ==============================================================================
# DEFINIÇÕES E FUNÇÕES AUXILIARES GLOBAIS (REFATORADAS)
# ==============================================================================

def get_target_layer(model_name, model):
    """
    Retorna a camada-alvo para XAI (Grad-CAM, etc).
    
    Agora usa o ExtractorRegistry para resolver automaticamente.
    """
    try:
        return ExtractorRegistry.get_target_layer(model, model_name)
    except ValueError:
        # Fallback para compatibilidade com modelos não registrados
        if "VGG" in model_name: return [model.features[-1]]
        if "RESNET" in model_name: return [model.layer4[-1]]
        if "DENSENET" in model_name: return [model.features.norm5]
        raise ValueError(f"Arquitetura {model_name} não suportada.")

def get_preprocess_transform(model_name=None):
    """
    Retorna o transform de preprocessing.
    
    Args:
        model_name: Nome da extratora (opcional). Se None, usa padrão ImageNet.
    
    Returns:
        transforms.Compose configurado
    """
    if model_name and ExtractorRegistry.exists(model_name):
        return ExtractorRegistry.create_preprocess_transform(model_name)
    
    # Fallback: padrão ImageNet
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Preprocess transform padrão
preprocess_transform = get_preprocess_transform()

features = {}
def get_features_hook(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# ==============================================================================
# FUNÇÕES AUXILIARES DE PLOTAGEM
# ==============================================================================

def find_font_file(font_name='arialbd.ttf'):
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for path in font_paths:
        if os.path.basename(path).lower() == font_name.lower():
            return path
    for fallback in ['DejaVuSans-Bold.ttf', 'LiberationSans-Bold.ttf', 'arial.ttf']:
        for path in font_paths:
            if os.path.basename(path).lower() == fallback.lower():
                return path
    return None

def overlay_manual(rgb_img, grayscale_cam, alpha=0.5):
    cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap_jet = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    heatmap_float = np.float32(heatmap_rgb) / 255
    visualization = (heatmap_float * alpha) + (np.float32(rgb_img) * (1 - alpha))
    visualization = visualization / np.max(visualization)
    return np.uint8(255 * visualization)

def salvar_heatmap_estilo_bateia(cluster_name, extratora_label, cam_label, grayscale_cam, output_path):
    """Gera heatmap individual 512x512 com estrelas (scatter) e eixos em arcmin."""
    coords_path = os.path.join(BASE_DIR, cluster_name, "Posicoes", f"{cluster_name}_coords_processed.csv")
    if not os.path.exists(coords_path): return
    df = pd.read_csv(coords_path, sep=';')
    
    x_data, y_data = df['X'] * 60, df['Y'] * 60
    limite = max(x_data.abs().max(), y_data.abs().max()) * 1.01

    fig = Figure(figsize=(512 / DPI, 512 / DPI), dpi=DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.scatter(x_data, y_data, color='black', alpha=1, s=0.8, zorder=1)
    
    extent = [-limite, limite, -limite, limite]
    ax.imshow(grayscale_cam, cmap='jet', alpha=0.45, extent=extent, aspect='equal', origin='lower', zorder=2)

    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.set_xlabel(r'$\Delta \alpha * \cos(\delta)$ [arcmin]', fontsize=16)
    ax.set_ylabel(r'$\Delta \delta$ [arcmin]', fontsize=16)
    ax.tick_params(labelsize=12)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap='jet'), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_path, format='jpg', bbox_inches='tight', pad_inches=0.1)

# ==============================================================================
# LÓGICA PRINCIPAL REFATORADA COM EXTRACTORREGISTRY
# ==============================================================================

def extrair_features_sinteticas(clusters_para_processar, suffix=".pt", apply_gap=False):
    """
    Extrai features dos aglomerados sintéticos.
    
    Args:
        suffix (str): Sufixo para o arquivo salvo (ex: ".pt" ou "_semgap.pt").
        apply_gap (bool): Se True, aplica GAP (AdaptiveAvgPool2d(1,1)) e Flatten.
                          Se False, salva o tensor espacial (C, 7, 7).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando o dispositivo: {device} para extração de features sintéticas (Suffix: {suffix}, GAP: {apply_gap}).")
    warnings.filterwarnings("ignore", category=UserWarning)
    
    gap_layer = nn.AdaptiveAvgPool2d((1, 1))

    logging.info("\n--- ETAPA 5A: Iniciando extração de features para todos os sintéticos ---")
    for cluster_name in clusters_para_processar:
        logging.info(f"\nProcessando Features de: {cluster_name}")
        
        for extratora_name in ExtractorRegistry.list_all():
            logging.info(f"  - Extrator: {extratora_name}")
            
            model = ExtractorRegistry.get_model(extratora_name).to(device)
            model.eval()
            target_layer = get_target_layer(extratora_name, model)
            
            feature_dir = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "Features", extratora_name)
            os.makedirs(feature_dir, exist_ok=True)
            
            preprocess = get_preprocess_transform(extratora_name)
            
            for synth_id in tqdm(range(1, T_MAX_SINTETICOS + 1), desc=f"    {extratora_name}"):
                img_path = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "CNN", f"{cluster_name}_Sintetico_{synth_id}.jpg")
                if not os.path.exists(img_path):
                    continue

                try:
                    hook_handle = target_layer[0].register_forward_hook(get_features_hook('features'))
                    pil_image = Image.open(img_path).convert('RGB')
                    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                    _ = model(input_tensor)
                    
                    feature_tensor = features['features'].squeeze() 

                    if apply_gap:
                        if len(feature_tensor.shape) == 3: # (C, H, W)
                             feature_tensor = gap_layer(feature_tensor.unsqueeze(0)).squeeze() # (C,)

                    filename = f"features_sintetico_{cluster_name}_{synth_id}{suffix}"
                    torch.save(feature_tensor, os.path.join(feature_dir, filename))
                    hook_handle.remove(); features.clear()
                except Exception as e:
                    logging.error(f"ERRO ao extrair features de {extratora_name} para a imagem {synth_id}: {e}")
                    if 'hook_handle' in locals() and hook_handle: hook_handle.remove()
                    features.clear()

def _processar_heatmaps_para_imagem(model, target_layer, input_tensor, rgb_img_224, device):
    all_heatmaps = {}
    output = model(input_tensor)
    targets = [ClassifierOutputTarget(torch.argmax(output))]
    
    combined_cam_methods = {**HEATMAP_TYPES_GRAD, **HEATMAP_TYPES_GRAD_FREE}

    for name, method_class in combined_cam_methods.items():
        try:
            with method_class(model=model, target_layers=target_layer) as cam_method:
                if hasattr(cam_method, 'batch_size'):
                    cam_method.batch_size = 128 

                grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)[0, :]
                
                min_val, max_val = np.min(grayscale_cam), np.max(grayscale_cam)
                grayscale_cam_normalized = (grayscale_cam - min_val) / (max_val - min_val) if max_val > min_val else grayscale_cam
                
                visualization_rgb = overlay_manual(rgb_img_224, grayscale_cam_normalized)
                all_heatmaps[name] = cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            logging.error(f"  ERRO ao calcular {name}: {e}")
            all_heatmaps[name] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        finally:
            torch.cuda.empty_cache()
            
    return all_heatmaps

def gerar_grids_heatmaps_sinteticas(clusters_para_processar):
    """Gera heatmaps individuais para aglomerados sintéticos selecionados."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n--- ETAPA 6B: Gerando Heatmaps Sintéticos (Individual 512px) ---")
    
    for cluster_name in clusters_para_processar:
        # Usa o registry para obter famílias
        for family_name in ExtractorRegistry.list_families():
            models_in_family = ExtractorRegistry.list_by_family(family_name)
            if not models_in_family:
                continue
                
            output_dir = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "Resultados", "Heatmaps", family_name)
            os.makedirs(output_dir, exist_ok=True)

            for synth_id in SINTETICOS_PARA_HEATMAPS:
                img_path = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "CNN", f"{cluster_name}_Sintetico_{synth_id}.jpg")
                if not os.path.exists(img_path): continue
                
                pil_image = Image.open(img_path).convert('RGB')
                
                for extratora_name in models_in_family:
                    preprocess = get_preprocess_transform(extratora_name)
                    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                    
                    ext_label = extratora_name.replace('VGG', 'VGG-').replace('RESNET', 'RESNET-').replace('DENSENET', 'DENSENET-')
                    try:
                        model = ExtractorRegistry.get_model(extratora_name).to(device)
                        model.eval()
                        target_layer = get_target_layer(extratora_name, model)
                        output = model(input_tensor); targets = [ClassifierOutputTarget(torch.argmax(output))]
                        
                        methods = {**HEATMAP_TYPES_GRAD, **HEATMAP_TYPES_GRAD_FREE}
                        for cam_label, method_class in methods.items():
                            with method_class(model=model, target_layers=target_layer) as cam_method:
                                if hasattr(cam_method, 'batch_size'): cam_method.batch_size = 32
                                grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)[0, :]
                                
                                file_name = f"{cluster_name}_{ext_label}_{cam_label}_synth_{synth_id}.jpg"
                                salvar_heatmap_estilo_bateia(cluster_name, ext_label, cam_label, grayscale_cam, os.path.join(output_dir, file_name))
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logging.error(f"  ERRO Sintético {synth_id} | {extratora_name}: {e}")

def executar_analise_reais(clusters_para_processar, suffix=".pt", apply_gap=False):
    """
    Extrai features dos aglomerados reais.
    
    Args:
        suffix (str): Sufixo para o arquivo salvo (ex: ".pt" ou "_semgap.pt").
        apply_gap (bool): Se True, aplica GAP.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n--- ETAPA 5C: Iniciando extração de features para Aglomerados Reais ---")
    logging.info(f"Usando o dispositivo: {device} (Suffix: {suffix}, GAP: {apply_gap})")
    warnings.filterwarnings("ignore", category=UserWarning)

    gap_layer = nn.AdaptiveAvgPool2d((1, 1))

    for cluster_name in clusters_para_processar:
        img_path = os.path.join(BASE_DIR, cluster_name, "CNN", f"{cluster_name}_1.jpg")
        if not os.path.exists(img_path): continue
        
        pil_image = Image.open(img_path).convert('RGB')
        preprocess = get_preprocess_transform()
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        for extratora_name in ExtractorRegistry.list_all():
            feature_dir = os.path.join(BASE_DIR, cluster_name, "Features", extratora_name)
            os.makedirs(feature_dir, exist_ok=True)
            
            model = ExtractorRegistry.get_model(extratora_name).to(device)
            model.eval()
            target_layer = get_target_layer(extratora_name, model)
            
            try:
                hook_handle = target_layer[0].register_forward_hook(get_features_hook('features'))
                _ = model(input_tensor)
                feature_tensor = features['features'].squeeze()
                
                if apply_gap:
                    if len(feature_tensor.shape) == 3:
                         feature_tensor = gap_layer(feature_tensor.unsqueeze(0)).squeeze()
                
                filename = f"features_real_{cluster_name}{suffix}"
                torch.save(feature_tensor, os.path.join(feature_dir, filename))
                
                hook_handle.remove(); features.clear()
            except Exception as e:
                logging.error(f"  ERRO ao processar features {extratora_name}: {e}")

def gerar_grids_heatmaps_reais(clusters_para_processar):
    """Gera heatmaps individuais para aglomerados reais."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n--- ETAPA 6: Gerando Heatmaps Reais (Individual 512px) ---")
    
    for cluster_name in clusters_para_processar:
        img_path = os.path.join(BASE_DIR, cluster_name, "CNN", f"{cluster_name}_1.jpg")
        if not os.path.exists(img_path): continue
        
        pil_image = Image.open(img_path).convert('RGB')

        for family_name in ExtractorRegistry.list_families():
            models_in_family = ExtractorRegistry.list_by_family(family_name)
            if not models_in_family:
                continue
                
            output_dir = os.path.join(BASE_DIR, cluster_name, "Resultados", "Heatmaps", family_name)
            os.makedirs(output_dir, exist_ok=True)

            for extratora_name in models_in_family:
                logging.info(f"  -> Processando: {cluster_name} | {extratora_name}")
                ext_label = extratora_name.replace('VGG', 'VGG-').replace('RESNET', 'RESNET-').replace('DENSENET', 'DENSENET-')
                
                model = ExtractorRegistry.get_model(extratora_name).to(device)
                model.eval()
                target_layer = get_target_layer(extratora_name, model)
                preprocess = get_preprocess_transform(extratora_name)
                
                try:
                    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                    output = model(input_tensor)
                    targets = [ClassifierOutputTarget(torch.argmax(output))]
                    
                    methods = {**HEATMAP_TYPES_GRAD, **HEATMAP_TYPES_GRAD_FREE}
                    for cam_label, method_class in methods.items():
                        with method_class(model=model, target_layers=target_layer) as cam_method:
                            if hasattr(cam_method, 'batch_size'): cam_method.batch_size = 32
                            grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)[0, :]
                            
                            file_name = f"{cluster_name}_{ext_label}_{cam_label}.jpg"
                            salvar_heatmap_estilo_bateia(cluster_name, ext_label, cam_label, grayscale_cam, os.path.join(output_dir, file_name))
                            
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"  ERRO em {extratora_name}: {e}")

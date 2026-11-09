import os
import shutil
from bateia import config


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def _remover_pasta(caminho):
    """Remove uma pasta (com todo o conteúdo) se ela existir."""
    if os.path.exists(caminho):
        try:
            shutil.rmtree(caminho)
            print(f"✅ Pasta removida: {caminho}")
        except Exception as e:
            print(f"❌ Erro ao remover {caminho}: {e}")
    else:
        print(f"➡️ Pasta não encontrada: {caminho}")


def _limpar_subpastas(base_path, pastas_para_remover):
    """Remove subpastas específicas dentro de cada aglomerado."""
    if not os.path.exists(base_path):
        print(f"⚠️ Diretório base '{base_path}' não encontrado.")
        return

    clusters = [n for n in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, n))]
    if not clusters:
        print("Nenhum aglomerado encontrado.")
        return

    for cluster in clusters:
        # Ignora a própria pasta de Redes Neurais para não processá-la como um aglomerado
        if cluster == "Redes Neurais":
            continue
        cluster_path = os.path.join(base_path, cluster)
        print(f"\n📂 Processando aglomerado: {cluster}")
        for pasta in pastas_para_remover:
            caminho_alvo = os.path.join(cluster_path, pasta)
            _remover_pasta(caminho_alvo)

# ==============================================================================
# OPÇÃO 1 - RESET TOTAL
# ==============================================================================

def reset_total():
    """
    Apaga completamente a pasta 'Aglomerados' e todo o seu conteúdo.
    """
    from bateia import config  # Import local para evitar dependência circular
    base_path = config.BASE_DIR

    print("\n--- RESET TOTAL ---")
    print(f"⚠️  Atenção: A pasta '{base_path}' e TODO o conteúdo serão excluídos!")

    confirm = input("Tem certeza que deseja continuar? (s/n): ").strip().lower()
    if confirm != 's':
        print("Operação cancelada.")
        return

    _remover_pasta(base_path)
    print("✅ Reset total concluído!")


# ==============================================================================
# OPÇÃO 2 - RESET HEATMAPS E FEATURES
# ==============================================================================

def reset_heatmaps_e_features():
    """
    Remove apenas as pastas relacionadas a heatmaps e features dentro de cada aglomerado.
    """
    from bateia import config
    base_path = config.BASE_DIR

    print("\n--- RESET: HEATMAPS e FEATURES ---")
    pastas_para_remover = [
        "Heatmaps", 
        "Features", 
        os.path.join("Resultados", "Heatmaps"),
        os.path.join("Sinteticos", "Features"),
        os.path.join("Sinteticos", "Resultados", "Heatmaps")
    ]
    _limpar_subpastas(base_path, pastas_para_remover)
    print("✅ Reset de Heatmaps e Features concluído!")


# ==============================================================================
# OPÇÃO 3 - RESET DADOS SINTÉTICOS
# ==============================================================================

def reset_sinteticos():
    """
    Remove apenas pastas e arquivos de dados sintéticos (CSV e imagens geradas).
    """
    from bateia import config
    base_path = config.BASE_DIR

    print("\n--- RESET: DADOS SINTÉTICOS ---")
    pastas_para_remover = ["Sinteticos", "Analises_Sinteticas", "Heatmaps_Sinteticos"]
    _limpar_subpastas(base_path, pastas_para_remover)
    print("✅ Reset de dados sintéticos concluído!")


# ==============================================================================
# OPÇÃO 4 - RESET RESULTADOS DE ANÁLISE
# ==============================================================================

def reset_analises():
    """
    Remove apenas resultados de análises (reais e sintéticos).
    """
    from bateia import config
    base_path = config.BASE_DIR

    print("\n--- RESET: RESULTADOS DE ANÁLISE ---")
    pastas_para_remover = ["Analises_Reais", "Analises_Sinteticas"]
    _limpar_subpastas(base_path, pastas_para_remover)
    print("✅ Reset de análises concluído!")


# ==============================================================================
# OPÇÃO 5 - RESET MODELOS E LOGS
# ==============================================================================

def reset_ia():
    """
    Remove a pasta 'Redes Neurais' e todo o seu conteúdo.
    """
    base_path = config.BASE_DIR
    print("\n--- RESET: PASTA REDES NEURAIS ---")
    
    # Constrói o caminho direto para a pasta 'Redes Neurais'
    caminho_alvo = os.path.join(base_path, "Redes Neurais")
    
    # Chama a função de remoção diretamente neste caminho
    _remover_pasta(caminho_alvo)
    
    print("✅ Reset de modelos de IA concluído!")


# ==============================================================================
# FIM DO MÓDULO
# ==============================================================================

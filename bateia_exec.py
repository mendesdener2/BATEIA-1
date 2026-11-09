# bateia_exec.py

# --- ADICIONE ESTAS LINHAS NO TOPO ---
import sys
import concurrent.futures
import time
import multiprocessing
import math
print("Starting BATEIA... Loading libraries (please wait)...")
sys.stdout.flush() 
# -------------------------------------
import os
import logging
import random
import gc
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# --- Módulos do Projeto BATEIA ---
from bateia import (
    config,
    real_clusters,
    synthetic_clusters,
    synthetic_analysis,
    heatmaps_features,
    readme
)
from bateia.config import BASE_DIR, T_MAX_SINTETICOS, SINTETICOS_PARA_HEATMAPS
from bateia.tools import(
    conversor_dados, 
    gdrive_downloader, 
    verificador,
    reset, 
    dl_utils
)
from bateia.models import mlp_model, svr_model, randomforest_model, lstm_model, gru_model, cnn1d_model

# ==============================================================================
# 1. FUNÇÕES DE UTILIDADE E CONTROLE DE MENU (Sem alterações)
# ==============================================================================

def limpar_tela():
    """Limpa a tela do terminal. Funciona em Windows, Linux e macOS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def _baixar_aglomerados_interativo(servico_gdrive):
    """
    Função auxiliar para gerenciar o menu interativo de download de aglomerados.
    """
    while True:
        limpar_tela()
        print("\n--- Cluster Download Menu ---")
        print("\nFetching cluster lists from Google Drive and locally...")
        
        lista_gdrive = gdrive_downloader.listar_arquivos_dat(servico_gdrive)
        lista_local = verificador.listar_aglomerados_locais(config.BASE_DIR)

        disponiveis = sorted([nome for nome in lista_gdrive if nome not in lista_local])
        presentes = sorted([nome for nome in lista_gdrive if nome in lista_local])

        print("\nAvailable for Download:")
        if not disponiveis:
            print("  No new clusters to download. All are present locally.")
        else:
            for i, nome in enumerate(disponiveis, 1):
                print(f"  [{i}] {nome}")
            print(f"  [{len(disponiveis) + 1}] Download All Available")
        
        print("\nPresent in local sample:")
        if not presentes:
            print("  No clusters found locally.")
        else:
            for nome in presentes:
                print(f"  - {nome}")
        
        escolha = input("\nEnter cluster number, 'all', or 'back': ").strip().lower()
        
        if escolha == 'back':
            break
        elif escolha == 'all' or (escolha.isdigit() and int(escolha) == len(disponiveis) + 1):
            print("\nStarting download of all available clusters...")
            for i, nome_cluster in enumerate(disponiveis):
                print(f"\n--- Processing {i+1}/{len(disponiveis)}: {nome_cluster} ---")
                caminho_local_dat = os.path.join(config.BASE_DIR, nome_cluster, "Posicoes", f"{nome_cluster}_coords.dat")
                if gdrive_downloader.baixar_arquivo_dat(servico_gdrive, nome_cluster, caminho_local_dat):
                    conversor_dados.processar_um_aglomerado(nome_cluster)
            print("\n--- Download of all clusters completed! ---")
            input("Press Enter to continue...")
        elif escolha.isdigit() and 1 <= int(escolha) <= len(disponiveis):
            num_escolha = int(escolha)
            nome_cluster = disponiveis[num_escolha - 1]
            print(f"\nYou selected: {nome_cluster}")
            caminho_local_dat = os.path.join(config.BASE_DIR, nome_cluster, "Posicoes", f"{nome_cluster}_coords.dat")
            if gdrive_downloader.baixar_arquivo_dat(servico_gdrive, nome_cluster, caminho_local_dat):
                conversor_dados.processar_um_aglomerado(nome_cluster)
            input("\nPress Enter to continue...")
        else:
            input("Error: Invalid option. Press Enter to try again.")

def setup_interativo():
    """
    Orquestra o fluxo interativo com um novo submenu para downloads e configuração.
    """
    limpar_tela()
    print("Authenticating with Google Drive...")
    servico_gdrive = gdrive_downloader.autenticar()
    if not servico_gdrive:
        logging.error("Failed to authenticate with Google Drive. Aborting setup.")
        input("\nPress Enter to continue...")
        return

    while True:
        limpar_tela()
        print("\n--- Setup and Download Menu ---")
        print("[1] Download/Update Literature Parameters")
        print("[2] Populate Literature Parameters (requires .csv)")
        print("[3] Download Cluster Data Files (.dat)")
        print("[4] Back to Main Menu")
        
        escolha = input("\nSelect an option: ").strip()

        if escolha == '1':
            print("\n--- Downloading literature parameters file ---")
            os.makedirs(config.BASE_DIR, exist_ok=True)
            caminho_csv_parametros = os.path.join(config.BASE_DIR, "cluster_params.csv")
            gdrive_downloader.baixar_arquivo_csv_raiz(servico_gdrive, "cluster_params.csv", caminho_csv_parametros)
            input("\nProcess completed. Press Enter to continue...")

        elif escolha == '2':
            print("\n--- Populating literature parameters from .csv ---")
            conversor_dados.salvar_parametros_literatura()
            input("\nProcess completed. Press Enter to continue...")

        elif escolha == '3':
            _baixar_aglomerados_interativo(servico_gdrive)

        elif escolha == '4':
            print("Returning to main menu...")
            break
        else:
            input("Invalid option. Press Enter to try again...")

def gerenciar_reset():
    """Submenu para gerenciar a limpeza do ambiente."""
    while True:
        limpar_tela()
        print("--- Reset Workspace ---\n")
        print("This action will delete processed data. Choose carefully:\n")
        print("[1] DELETE EVERYTHING")
        print("[2] Delete HEATMAPS and FEATURES only")
        print("[3] Delete SYNTHETIC DATA (CSVs and Images) only")
        print("[4] Delete ANALYSIS RESULTS (Real and Synthetic) only")
        print("[5] Delete NEURAL NETWORKS only")
        print("[6] Back to Main Menu")
        
        escolha = input("\nSelect an option: ").strip()
        
        if escolha == '6':
            break
        
        funcoes_reset = {
            '1': reset.reset_total,
            '2': reset.reset_heatmaps_e_features,
            '3': reset.reset_sinteticos,
            '4': reset.reset_analises,
            '5': reset.reset_ia
        }
        
        if escolha in funcoes_reset:
            confirmacao = input(f"Are you sure you want to execute option {escolha}? (y/n): ").lower()
            if confirmacao == 'y':
                funcoes_reset[escolha]()
                input("\nReset operation completed. Press Enter to continue...")
            else:
                input("\nOperation cancelled. Press Enter to continue...")
        else:
            input("Invalid option. Press Enter to try again...")

# ==============================================================================
# 2. CONFIGURAÇÕES GLOBAIS PARA A EXECUÇÃO (Atualizado para usar Registry)
# ==============================================================================
from bateia.features.registry import ExtractorRegistry

PARAMETROS_A_RODAR = ["f0", "Rc", "Rt", "fbg"]
# Agora usa o registry para listar extratoras automaticamente
EXTRATORAS_A_RODAR = ExtractorRegistry.list_all()

config_geral_treino = {
    "n_trials": 1,
    "device": "cuda" if config.torch.cuda.is_available() else "cpu"
}

MODEL_FOLDER_MAP = {
    "MLP": "MLPerceptron",
    "SVR": "SVR",
    "RandomForest": "RandomForest",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "1DCNN": "1DCNN"
}

# ==============================================================================
# 3. FUNÇÕES DE EXECUÇÃO DO PIPELINE E TREINAMENTO DE IA (AJUSTADAS)
# ==============================================================================
def _rodar_etapa_5_worker(clusters_chunk, gpu_id, modo_extracao="ambos"):
    """Worker para rodar a Etapa 5 isolado em uma GPU específica (Apenas Extração Pura)."""
    if gpu_id is not None:
        import torch
        torch.cuda.set_device(gpu_id)
        
    logging.info(f"\n--- Starting Extraction (GPU ID: {gpu_id if gpu_id is not None else 'Default'}) | Clusters in this chunk: {len(clusters_chunk)} ---")
    
    # 1. Versão Padrão (COM GAP)
    if modo_extracao in ["gap", "ambos"]:
        logging.info("  -> Extracting version WITH GAP (.pt)")
        heatmaps_features.executar_analise_reais(clusters_chunk, suffix=".pt", apply_gap=True)
        heatmaps_features.extrair_features_sinteticas(clusters_chunk, suffix=".pt", apply_gap=True)
    
    # 2. Spatial Version (WITHOUT GAP)
    if modo_extracao in ["semgap", "ambos"]:
        logging.info("  -> Extracting version WITHOUT GAP (_semgap.pt)")
        heatmaps_features.executar_analise_reais(clusters_chunk, suffix="_semgap.pt", apply_gap=False)
        heatmaps_features.extrair_features_sinteticas(clusters_chunk, suffix="_semgap.pt", apply_gap=False)
    
    return True

def executar_etapa_selecionada(escolha, clusters_para_processar, modo_extracao="ambos"):
    """Executa uma etapa específica do pipeline com base na escolha do usuário."""
    if not clusters_para_processar:
        logging.warning("No clusters selected for processing.")
        return False

    etapas = {
        '1': ("STAGE 1: Cluster data processing", conversor_dados.processar_todos_aglomerados),
        '2': ("STAGE 2: Real cluster analysis (Least Squares)", real_clusters.processar_aglomerados_reais),
        '3': ("STAGE 3: Synthetic cluster generation", synthetic_clusters.processar_aglomerados_sinteticos),
        '4': ("STAGE 4: Synthetic cluster analysis", synthetic_analysis.processar_analise_sintetica),
        '5': ("STAGE 5: Feature Extraction", None), # Extraction Only
        '6': ("STAGE 6: Heatmap Generation", None),  # Heatmaps Only
    }
    
    try:
        if escolha in etapas:
            nome_etapa, funcao_etapa = etapas[escolha]
            clusters_str = ", ".join(clusters_para_processar)
            logging.info(f"\nStarting {nome_etapa} for: {clusters_str}")

            # --- MULTI-GPU LOGIC FOR STAGE 5 (EXTRACTION) ---
            if escolha == '5':
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                
                if num_gpus > 1 and len(clusters_para_processar) > 1:
                    print(f"  -> {num_gpus} GPUs detected! Splitting FEATURE EXTRACTION equitably...")
                    
                    tamanho_chunk = math.ceil(len(clusters_para_processar) / num_gpus)
                    chunks = [clusters_para_processar[i:i + tamanho_chunk] for i in range(0, len(clusters_para_processar), tamanho_chunk)]
                    
                    ctx = multiprocessing.get_context('spawn')
                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
                        futuros = []
                        for gpu_idx, chunk in enumerate(chunks):
                            if chunk: 
                                futuros.append(executor.submit(_rodar_etapa_5_worker, chunk, gpu_idx, modo_extracao))
                        
                        for futuro in concurrent.futures.as_completed(futuros):
                            futuro.result()
                else:
                    print("  -> Using default GPU processing (1 GPU or sequential) for EXTRACTION...")
                    _rodar_etapa_5_worker(clusters_para_processar, None, modo_extracao)
            
            # --- LOGIC FOR STAGE 6 (HEATMAPS) ---
            elif escolha == '6':
                print("  -> Starting generation of Real and Synthetic Heatmaps (Sequential Process)...")
                heatmaps_features.gerar_grids_heatmaps_reais(clusters_para_processar)
                heatmaps_features.gerar_grids_heatmaps_sinteticas(clusters_para_processar)
            
            # --- CPU LOGIC FOR STAGES 1, 2, 3 and 4 ---
            else:
                max_cpus = max(1, int(multiprocessing.cpu_count() * 0.85))
                print(f"  -> Accelerating: Distributing clusters across {max_cpus} CPU cores...")
                
                ctx = multiprocessing.get_context('spawn')
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpus, mp_context=ctx) as executor:
                    futuros = [executor.submit(funcao_etapa, [cluster]) for cluster in clusters_para_processar]
                    for futuro in concurrent.futures.as_completed(futuros):
                        futuro.result() 

            logging.info(f">>> {nome_etapa}: Completed.")
            return True
        else:
            logging.warning("Invalid stage option.")
            return False
            
    except Exception as e:
        logging.error(f"A critical error occurred during the execution of stage {escolha}: {e}", exc_info=True)
        return False
        
        
        
# --- FUNÇÃO DE SPLIT MANUAL (15/4/1) - (Mantida como Opção A) ---
def selecionar_clusters_para_split(clusters_disponiveis, n_treino=15, n_validacao=4):
    """Função interativa para o usuário montar os splits de treino, validação e teste."""
    
    total_necessario = n_treino + n_validacao # 19
    
    if len(clusters_disponiveis) < total_necessario + 1: # Needs 20
        logging.error(f"Not enough clusters. At least {total_necessario + 1} are required.")
        return None

    disponiveis = list(clusters_disponiveis)
    random.shuffle(disponiveis)
    
    treino, validacao, teste = [], [], []

    while len(treino) < n_treino:
        limpar_tela()
        print(f"--- Select {n_treino} Clusters for TRAINING ({len(treino)}/{n_treino}) ---\n")
        for i, nome in enumerate(disponiveis, 1):
            print(f"  [{i}] {nome}")
        try:
            escolha = int(input("\nChoose a number: ")) - 1
            if 0 <= escolha < len(disponiveis):
                selecionado = disponiveis.pop(escolha)
                treino.append(selecionado)
            else: input("Invalid option. Press Enter...")
        except ValueError: input("Invalid input. Press Enter...")

    while len(validacao) < n_validacao:
        limpar_tela()
        print(f"--- Select {n_validacao} Clusters for VALIDATION ({len(validacao)}/{n_validacao}) ---\n")
        print(f"Training Clusters: {', '.join(sorted(treino))}\n")
        for i, nome in enumerate(disponiveis, 1):
            print(f"  [{i}] {nome}")
        try:
            escolha = int(input("\nChoose a number: ")) - 1
            if 0 <= escolha < len(disponiveis):
                selecionado = disponiveis.pop(escolha)
                validacao.append(selecionado)
            else: input("Invalid option. Press Enter...")
        except ValueError: input("Invalid input. Press Enter...")
    
    teste = disponiveis
    split = {"train": sorted(treino), "val": sorted(validacao), "test": sorted(teste)}
    
    limpar_tela()
    print("--- Data Split Configured ---\n")
    print(f"✅ Train ({len(split['train'])}): {', '.join(split['train'])}")
    print(f"✅ Validation ({len(split['val'])}): {', '.join(split['val'])}")
    print(f"✅ Test ({len(split['test'])}): {', '.join(split['test'])}")
    print("\n" + "="*50)
    
    return [split]

# --- FUNÇÕES DE SPLIT ESTRATIFICADO (Opção B) ---

def _get_stratification_groups_for_all(clusters_locais_todos):
    """
    (Função re-adicionada)
    Carrega cluster_params.csv e retorna os perfis K-Means (K=3) para TODOS os clusters.
    """
    params_csv_path = os.path.join(config.BASE_DIR, "cluster_params.csv")
    if not os.path.exists(params_csv_path):
        logging.error(f"CRITICAL ERROR: cluster_params.csv not found in {config.BASE_DIR}")
        return None, None
    
    try:
        df_params = pd.read_csv(params_csv_path)
    except Exception as e:
        logging.error(f"Error reading cluster_params.csv: {e}")
        return None, None

    clusters_no_csv = set(df_params['Cluster'])
    if not set(clusters_locais_todos).issubset(clusters_no_csv):
        logging.error("Error: Some local clusters were not found in cluster_params.csv.")
        return None, None

    feature_cols = ['Rt', 'Rc', 'f0', 'fbg']
    df_params = df_params[df_params['Cluster'].isin(clusters_locais_todos)]
    df_features_all = df_params.set_index('Cluster')[feature_cols]
    
    scaler = StandardScaler()
    scaled_features_array = scaler.fit_transform(df_features_all)
    
    k_clusters = 3 
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=50) 
    group_labels = kmeans.fit_predict(scaled_features_array)
    df_features_all['Group'] = group_labels

    y_stratify_groups = df_features_all['Group'].values
    return df_features_all, y_stratify_groups

def selecionar_clusters_para_teste_auto(clusters_locais_todos):
    """
    (NOVA FUNÇÃO)
    Menu estilo "checklist" para selecionar aglomerados de TESTE.
    """
    limpar_tela()
    print("--- Configuring Stratified Split (Test Selection) ---")

    # 1. Obter os perfis de estratificação para exibir
    df_all_clusters, _ = _get_stratification_groups_for_all(clusters_locais_todos)
    if df_all_clusters is None:
        return None 
    
    todos_os_aglomerados = sorted(clusters_locais_todos)
    test_clusters_selecionados = []

    # --- Loop de Seleção de Teste (estilo menu de modelos) ---
    while True:
        limpar_tela()
        print("--- Select Clusters for the FINAL TEST ---\n")
        
        if not test_clusters_selecionados:
            print("Test Clusters: None")
        else:
            print(f"Test Clusters: {', '.join(sorted(test_clusters_selecionados))}")
        print("-" * 50)

        for i, nome_cluster in enumerate(todos_os_aglomerados, 1):
            marcador = "✅" if nome_cluster in test_clusters_selecionados else "  "
            perfil = df_all_clusters.loc[nome_cluster, 'Group']
            print(f" [{i}] {marcador} {nome_cluster} (Profile: {perfil})")
        
        print("\n" + "-"*50 + "\n")
        print(f" [{len(todos_os_aglomerados) + 1}] Confirm Selection and Start")
        print(f" [{len(todos_os_aglomerados) + 2}] Back")

        escolha = input("\nChoose a number to add/remove, or confirm: ").strip()

        if escolha.isdigit():
            num_escolha = int(escolha)
            # Adicionar/Remover aglomerado
            if 1 <= num_escolha <= len(todos_os_aglomerados):
                cluster = todos_os_aglomerados[num_escolha - 1]
                if cluster in test_clusters_selecionados:
                    test_clusters_selecionados.remove(cluster)
                else:
                    test_clusters_selecionados.append(cluster)
            # Confirmar Seleção
            elif num_escolha == len(todos_os_aglomerados) + 1:
                if not test_clusters_selecionados:
                    input("No test cluster selected. Press Enter...")
                    continue
                
                # Validate count
                n_test = len(test_clusters_selecionados)
                n_dev = len(clusters_locais_todos) - n_test
                if n_dev < 5: # (At least 5 for Train/Val)
                    print(f"Error: You selected {n_test} clusters, leaving only {n_dev} for Development.")
                    input("Please leave at least 5 clusters for Training/Validation. Press Enter...")
                    continue
                
                break # Sai do loop de seleção
            # Voltar
            elif num_escolha == len(todos_os_aglomerados) + 2:
                return None # Cancels operation
            else:
                input("Invalid option. Press Enter...")
        else:
            input("Invalid option. Press Enter...")

    # --- Fim do Loop ---
    
    test_clusters = sorted(test_clusters_selecionados)
    dev_clusters = sorted([c for c in clusters_locais_todos if c not in test_clusters])
    
    # 3. Pegar os perfis APENAS do conjunto de desenvolvimento
    df_dev_with_groups = df_all_clusters.loc[dev_clusters]
    y_stratify_dev = df_dev_with_groups['Group'].values
    
    print(f"\n✅ Final Test Set (N={len(test_clusters)}): {', '.join(test_clusters)}")
    print(f"✅ Development Set (N={len(dev_clusters)}): {len(dev_clusters)} clusters")
    print("\n" + "="*50)
    
    return dev_clusters, test_clusters, df_dev_with_groups, y_stratify_dev

def gerar_split_estratificado_70_30(dev_clusters, test_clusters, df_dev_with_groups, y_stratify_dev):
    """
    (NOVA FUNÇÃO)
    Gera o split 70/30 estratificado para Treino/Validação.
    """
    print("--- Generating Stratified Split 70% Train / 30% Validation ---")
    
    n_dev = len(dev_clusters)
    test_size_ratio = 0.30 # 30% for Validation
    n_val = int(round(n_dev * test_size_ratio)) # Round
    if n_val == 0: n_val = 1 # Guarantee at least 1 validation cluster
    n_train = n_dev - n_val
    
    dev_set_clusters_np = np.array(dev_clusters)

    print(f"\nSplitting N={n_dev} Development set into Train (N={n_train}) and Validation (N={n_val})...")

    try:
        train_clusters, val_clusters = train_test_split(
            dev_set_clusters_np,
            test_size=test_size_ratio,
            random_state=42, # Anchor for sub-split
            stratify=y_stratify_dev
        )
    except ValueError as e:
        logging.warning(f"Stratification failed for {n_train}/{n_val} split: {e}. Using non-stratified split.")
        train_clusters, val_clusters = train_test_split(
            dev_set_clusters_np,
            test_size=test_size_ratio,
            random_state=42
        )

    final_split = {
        "train": sorted(list(train_clusters)),
        "val":   sorted(list(val_clusters)),  
        "test":  test_clusters # Já está ordenado
    }
    
    print("\n" + "="*50)
    print("--- Data Split Configured ---")
    print(f"✅ Train ({len(final_split['train'])}): {', '.join(final_split['train'])}")
    print(f"✅ Validation ({len(final_split['val'])}): {', '.join(final_split['val'])}")
    print(f"✅ Final Test ({len(final_split['test'])}): {', '.join(final_split['test'])}")
    print("="*50)

    input("\nFinal Test configuration completed. Press Enter to continue...")
    return [final_split]

# No bateia_exec.py

def executar_treino_ia(metodologia, data_split_config, modelos_a_rodar, suffix=".pt", auto_confirm=False):
    """Orquestra o pipeline de treinamento e registra os tempos de execução."""
    limpar_tela()
    logging.info(f"\n{'='*60}\nSTARTING TRAINING PIPELINE - METHODOLOGY: {metodologia} (Suffix: {suffix})\n{'='*60}")
    
    if not auto_confirm:
        confirm = input("This process may be time-consuming. Do you want to continue? (y/n): ").lower()
        if confirm != 'y':
            logging.info("Training cancelled.")
            return

    base_neural_dir = os.path.join(config.BASE_DIR, "Redes Neurais")
    os.makedirs(base_neural_dir, exist_ok=True)
    
    # --- NOVA LÓGICA: BUSCA SEQUENCIAL DE SLOT LIVRE ---
    # Começa do 1 e sobe até achar uma pasta onde o trabalho solicitado AINDA NÃO FOI FEITO.
    current_try_id = 1
    
    # Definições auxiliares para verificação
    temp_gap_label = "NO-GAP" if "semgap" in suffix else "GAP"
    jobs_necessarios = [f"{p}_{e}" for p in PARAMETROS_A_RODAR for e in EXTRATORAS_A_RODAR]
    splits_check = range(1, len(data_split_config) + 1) if metodologia == 'AM' else [1]
    
    LOG_ABBREVIATIONS = {
        "MLP": "mlp", "SVR": "svr", "RandomForest": "randomforest",
        "LSTM": "lstm", "GRU": "gru", "1DCNN": "1dcnn"
    }

    while True:
        check_folder = os.path.join(base_neural_dir, f"TRY-{current_try_id}")
        
        # 1. Se a pasta nem existe, é aqui mesmo que vamos rodar.
        if not os.path.exists(check_folder):
            break
            
            # 2. Se a pasta existe, verificamos se o trabalho ATUAL (Metodologia + Modelos) já está feito nela.
        tarefa_atual_ja_concluida_nesta_pasta = True
        
        for mod in modelos_a_rodar:
            mod_folder_map = MODEL_FOLDER_MAP.get(mod)
            abbr = LOG_ABBREVIATIONS.get(mod, mod.lower())
            
            # Caminho da pasta do modelo
            path_check = os.path.join(check_folder, temp_gap_label, mod_folder_map, f"{metodologia}{mod}")
            
            # Nome do log de progresso (conforme core_trainer.py)
            log_name = f"bm_{abbr}_progress.log" if metodologia == 'BM' else f"am_{abbr}_progress.log"
            full_log_path = os.path.join(path_check, log_name)
            
            # Se o arquivo de log não existe, então a tarefa não foi feita nesta pasta.
            if not os.path.exists(full_log_path):
                tarefa_atual_ja_concluida_nesta_pasta = False
                break
            
            # Se o arquivo existe, verificamos se tem todos os jobs básicos (parametro_extratora)
            try:
                with open(full_log_path, 'r') as f:
                    conteudo_log = f.read()
                if not all(job in conteudo_log for job in jobs_necessarios):
                    tarefa_atual_ja_concluida_nesta_pasta = False
                    break
            except Exception:
                tarefa_atual_ja_concluida_nesta_pasta = False
                break
        
        if not tarefa_atual_ja_concluida_nesta_pasta:
            break
        else:
            current_try_id += 1

    # ----------------------------------------------------

    try_folder_name = f"TRY-{current_try_id}"
    execution_output_dir = os.path.join(base_neural_dir, try_folder_name)
    os.makedirs(execution_output_dir, exist_ok=True)

    print(f"Using/Continuing execution directory: {execution_output_dir}")
    logging.info(f"Using/Continuing execution directory: {try_folder_name}")

    execution_times = {model: {} for model in modelos_a_rodar}
    gap_label = "NO-GAP" if "semgap" in suffix else "GAP"

    try:
        mapa_de_modelos = {
            "MLP": mlp_model.executar_treino_mlp,
            "SVR": svr_model.executar_treino_svr,
            "RandomForest": randomforest_model.executar_treino_random_forest,
            "LSTM": lstm_model.executar_treino_lstm,
            "GRU": gru_model.executar_treino_gru,
            "1DCNN": cnn1d_model.executar_treino_1dcnn
        }
        
        for modelo_nome in modelos_a_rodar:
            model_total_start_time = time.time()
            model_base_folder = MODEL_FOLDER_MAP.get(modelo_nome)
            
            output_dir_modelo = os.path.join(execution_output_dir, gap_label, model_base_folder, f"{metodologia}{modelo_nome}")

            for parametro in PARAMETROS_A_RODAR:
                for extratora in EXTRATORAS_A_RODAR:
                    if modelo_nome in mapa_de_modelos:
                        extractor_start_time = time.time()
                        
                        mapa_de_modelos[modelo_nome](
                            metodologia=metodologia, parametro=parametro, extratora=extratora,
                            data_split_config=data_split_config, config_geral=config_geral_treino,
                            output_dir=output_dir_modelo, gap_label=gap_label, suffix=suffix
                        )
                        
                        extractor_end_time = time.time()
                        elapsed_seconds = extractor_end_time - extractor_start_time
                        time_key = f"{parametro}_{extratora}"
                        execution_times.setdefault(modelo_nome, {})[time_key] = elapsed_seconds

                        print("\nClearing memory before next extractor...")
                        gc.collect()
                        print("Memory cleanup completed.")

            model_total_end_time = time.time()
            execution_times[modelo_nome]['_total_'] = model_total_end_time - model_total_start_time
        
        logging.info(f"\n{'='*60}\nTRAINING PIPELINE ({metodologia}) COMPLETED\n{'='*60}")
        
    except Exception as e:
        logging.error(f"Critical error during training ({metodologia}): {e}", exc_info=True)
        
    finally:
        print("\nLogging execution times...")
        
        if metodologia == "AM":
            try:
                c_list = data_split_config[0].get("test", [])
                cluster_ref = c_list[0] if len(c_list) == 1 else f"Group_{len(c_list)}"
            except (IndexError, AttributeError, TypeError):
                cluster_ref = "Unknown"
            file_suffix_name = f"-{cluster_ref}"
        else:
            file_suffix_name = "-BM"

        for modelo_nome, model_times in execution_times.items():
            model_folder = MODEL_FOLDER_MAP.get(modelo_nome)
            if not model_folder or not model_times:
                continue
            
            specific_model_dir = os.path.join(execution_output_dir, gap_label, model_folder, f"{metodologia}{modelo_nome}")
            os.makedirs(specific_model_dir, exist_ok=True)

            # Nome padronizado para o log de resumo final
            tag_extra = suffix.replace('.pt', '')
            filename = f"{metodologia}{modelo_nome}-{gap_label}{file_suffix_name}{tag_extra}.txt"
            log_path = os.path.join(specific_model_dir, filename)

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Execution: {try_folder_name})\n") 
                f.write(f"Model: {modelo_nome} - Methodology: {metodologia} - Mode: {gap_label}\n\n")
                
                total_model_time = model_times.pop('_total_', 0)
                
                times_by_param = {}
                for key, seconds in model_times.items():
                    if '_' in key:
                        param, extr = key.split('_', 1)
                        if param not in times_by_param:
                            times_by_param[param] = {}
                        times_by_param[param][extr] = seconds
                
                for param, extr_times in times_by_param.items():
                       f.write(f" Parameter '{param}':\n")
                       for extratora, seconds in sorted(extr_times.items()):
                           f.write(f"   -- {extratora.upper()} = {seconds:.2f} seconds\n")
                       f.write("\n")

                f.write(f"- Total time recorded for the model in this run = {total_model_time:.2f} seconds\n")
                f.write(f"(Note: Total time includes setups. Individual times refer to parameter/extractor combinations.)\n")
                f.write(f"{'='*60}\n")
        
        print("\nExecution time logs saved to respective model folders.")
        if not auto_confirm:
            input("Process completed. Press Enter to continue...")

def gerenciar_etapas_analise(clusters_locais):
    """Submenu para selecionar aglomerados e etapas de 1 a 5."""
    clusters_selecionados = []

    # --- LOOP DE SELEÇÃO DE AGLOMERADOS ---
    while True:
        limpar_tela()
        print("--- Select Clusters for Analysis ---\n")
        
        if not clusters_selecionados:
            print("Selected Clusters: None")
        else:
            print(f"Selected Clusters: {', '.join(sorted(clusters_selecionados))}")
        print("-" * 50)

        for i, nome in enumerate(clusters_locais, 1):
            marcador = "✅" if nome in clusters_selecionados else "  "
            print(f" [{i}] {marcador} {nome}")

        print("\n" + "-"*50 + "\n")
        print(f" [{len(clusters_locais) + 1}] Select ALL clusters")
        print(f" [{len(clusters_locais) + 2}] Confirm Cluster Selection")
        print(f" [{len(clusters_locais) + 3}] Back to Main Menu")

        escolha_cluster = input("\nSelect an option: ").strip()

        if escolha_cluster.isdigit():
            num_escolha = int(escolha_cluster)
            if 1 <= num_escolha <= len(clusters_locais):
                cluster = clusters_locais[num_escolha - 1]
                if cluster in clusters_selecionados:
                    clusters_selecionados.remove(cluster)
                else:
                    clusters_selecionados.append(cluster)
            elif num_escolha == len(clusters_locais) + 1:
                clusters_selecionados = list(clusters_locais)
            elif num_escolha == len(clusters_locais) + 2:
                if not clusters_selecionados:
                    input("No cluster selected. Press Enter to continue...")
                    continue
                clusters_a_rodar = sorted(clusters_selecionados)
                break # Exit selection loop and proceed to stages
            elif num_escolha == len(clusters_locais) + 3:
                return # Cancel and back to main menu
            else:
                input("Invalid option. Press Enter...")
        else:
            input("Invalid option. Press Enter...")

    # --- LOOP DE SELEÇÃO DE ETAPAS ---
    todas_as_etapas = [
        ("1", "Data Processing (Ensure CSVs)"),
        ("2", "Real Cluster Analysis"),
        ("3", "Synthetic Cluster Generation"),
        ("4", "Synthetic Cluster Analysis"),
        ("5", "Feature Extraction (GPU Accelerated)"),
        ("6", "Heatmap Generation (Images)")
    ]
    etapas_selecionadas = []

    while True:
        limpar_tela()
        print(f"Cluster(s) to process: {', '.join(clusters_a_rodar)}\n")
        print("--- Select Analysis Stages ---\n")
        
        if not etapas_selecionadas:
            print("Selected Stages: None")
        else:
            print(f"Selected Stages: {', '.join(sorted(etapas_selecionadas))}")
        print("-" * 50)

        for i, (num_etapa, nome_etapa) in enumerate(todas_as_etapas, 1):
            marcador = "✅" if num_etapa in etapas_selecionadas else "  "
            print(f" [{i}] {marcador} Stage {num_etapa}: {nome_etapa}")

        print("\n" + "-"*50 + "\n")
        print(f" [{len(todas_as_etapas) + 1}] Select ALL stages")
        print(f" [{len(todas_as_etapas) + 2}] Start Execution")
        print(f" [{len(todas_as_etapas) + 3}] Back to Cluster Selection")

        escolha_etapa = input("\nSelect an option: ").strip()

        if escolha_etapa.isdigit():
            num_escolha = int(escolha_etapa)
            if 1 <= num_escolha <= len(todas_as_etapas):
                etapa = todas_as_etapas[num_escolha - 1][0]
                if etapa in etapas_selecionadas:
                    etapas_selecionadas.remove(etapa)
                else:
                    etapas_selecionadas.append(etapa)
            elif num_escolha == len(todas_as_etapas) + 1:
                etapas_selecionadas = [e[0] for e in todas_as_etapas]
            elif num_escolha == len(todas_as_etapas) + 2:
                if not etapas_selecionadas:
                    input("No stage selected. Press Enter to continue...")
                    continue
                
                # If Stage 5 (Extraction) is selected, ask for the mode
                modo_extracao = "ambos"
                if '5' in etapas_selecionadas:
                    print("\n--- Feature Extraction Configuration (Stage 5) ---")
                    print("[1] GAP Only (Global Vector .pt)")
                    print("[2] No-GAP Only (Spatial 7x7 _semgap.pt)")
                    print("[3] BOTH (Generate both file types)")
                    escolha_modo = input("Select an option: ").strip()
                    if escolha_modo == '1': modo_extracao = "gap"
                    elif escolha_modo == '2': modo_extracao = "semgap"
                    else: modo_extracao = "ambos"

                for etapa in sorted(etapas_selecionadas):
                    if not executar_etapa_selecionada(etapa, clusters_a_rodar, modo_extracao=modo_extracao):
                        logging.error(f"Pipeline interrupted at Stage {etapa}.")
                        break
                input("\nAnalysis pipeline completed. Press Enter to continue...")
                return # Back to main menu after completion
            elif num_escolha == len(todas_as_etapas) + 3:
                return gerenciar_etapas_analise(clusters_locais) # Restart function (back to start)
            else:
                input("Invalid option. Press Enter...")
        else:
            input("Invalid option. Press Enter...")
            
def gerenciar_treinamento_ia(clusters_locais):
    """Submenu para selecionar modelos, metodologia e iniciar o treinamento."""
    todos_os_modelos = ["MLP", "SVR", "RandomForest", "LSTM", "GRU", "1DCNN"]
    modelos_selecionados = []

    # --- Loop de Seleção de Modelos (sem alteração) ---
    while True:
        limpar_tela()
        print("--- Stage 6: AI Training ---")
        print("\n--- Model Selection Menu ---\n")
        
        if not modelos_selecionados:
            print("Selected Models: None")
        else:
            print(f"Selected Models: {', '.join(sorted(modelos_selecionados))}")
        print("-" * 50)

        for i, nome_modelo in enumerate(todos_os_modelos, 1):
            marcador = "✅" if nome_modelo in modelos_selecionados else "  "
            print(f" [{i}] {marcador} {nome_modelo}")
        
        print("\n" + "-"*50 + "\n")
        print(f" [{len(todos_os_modelos) + 1}] Select ALL models")
        print(f" [{len(todos_os_modelos) + 2}] Start Training with Selected Models")
        print(f" [{len(todos_os_modelos) + 3}] Back to Previous Menu")

        escolha_modelo = input("\nSelect an option: ").strip()

        if escolha_modelo.isdigit():
            num_escolha = int(escolha_modelo)
            if 1 <= num_escolha <= len(todos_os_modelos):
                modelo = todos_os_modelos[num_escolha - 1]
                if modelo in modelos_selecionados:
                    modelos_selecionados.remove(modelo)
                else:
                    modelos_selecionados.append(modelo)
            elif num_escolha == len(todos_os_modelos) + 1:
                modelos_selecionados = list(todos_os_modelos)
            elif num_escolha == len(todos_os_modelos) + 2:
                if not modelos_selecionados:
                    input("No model selected. Press Enter to continue...")
                    continue
                break 
            elif num_escolha == len(todos_os_modelos) + 3:
                return
            else:
                input("Invalid option. Press Enter...")
        else:
            input("Invalid option. Press Enter...")

    # --- Loop de Seleção de Metodologia (ALTERADO) ---
    while True:
        limpar_tela()
        print("--- Stage 6: AI Training ---\n")
        print(f"Models to train: {', '.join(sorted(modelos_selecionados))}\n")
        print("[1] AM Methodology (Uncontaminated)") 
        print("[2] BM Methodology (Contaminated) - Use All Data")
        print("[3] Back")
        
        escolha_metodo = input("\nSelect training methodology: ").strip()
    
        if escolha_metodo in ['1', '2']:
            
            # --- Data Input Choice ---
            print("\n--- Data Input Configuration ---")
            print("[1] With GAP (Default - Global Feature Vector)")
            print("[2] No-GAP (Spatial - 7x7 Sequence)")
            escolha_gap = input("Select an option: ").strip()
            
            suffix_escolhido = "_semgap.pt" if escolha_gap == '2' else ".pt"
            modo_txt = "NO-GAP" if escolha_gap == '2' else "WITH GAP"
            print(f" -> Selected mode: {modo_txt} (File: *{suffix_escolhido})")
            time.sleep(1)
            # ---------------------------------------------
            
            if escolha_metodo == '1': # AM
                
                # --- NEW LOGIC (AM SUB-MENU) ---
                limpar_tela()
                print("--- AM Methodology (Uncontaminated) ---")
                print("\nHow do you want to define the data split?\n")
                print("[1] Manual Split (15/4/1)")
                print("[2] Stratified Split (Select Test, 70/30)")
                print("[3] Back")
                
                escolha_split = input("\nSelect split type: ").strip()
                
                data_split = None
                if escolha_split == '1':
                    # Chama a função manual 15/4/1 (que já existe)
                    data_split = selecionar_clusters_para_split(clusters_locais, 15, 4) 
                
                elif escolha_split == '2':
                    # PASSO 1: Chama o menu de seleção de teste
                    split_info = selecionar_clusters_para_teste_auto(clusters_locais)
                    if split_info is None:
                        # Usuário cancelou
                        continue 
                    
                    dev_clusters, test_clusters, df_dev_with_groups, y_stratify_dev = split_info
                    
                    # PASSO 2: Chama a nova função de split 70/30
                    data_split = gerar_split_estratificado_70_30(dev_clusters, test_clusters, df_dev_with_groups, y_stratify_dev)
                
                elif escolha_split == '3':
                    continue # Back to methodology menu
                
                else:
                    input("Invalid option. Press Enter...")
                    continue
                # --- END NEW LOGIC ---
                
                if data_split:
                    executar_treino_ia("AM", data_split, modelos_selecionados, suffix=suffix_escolhido)

            elif escolha_metodo == '2': # BM
                executar_treino_ia("BM", [], modelos_selecionados, suffix=suffix_escolhido)
        
        elif escolha_metodo == '3':
            break 
        else:
            input("Invalid option. Press Enter...")

def gerenciar_execucao_pipeline():
    """Submenu principal para executar etapas de análise ou IA."""
    clusters_locais = sorted(verificador.listar_aglomerados_locais(config.BASE_DIR))
    if not clusters_locais or len(clusters_locais) < 5: 
        logging.warning(f"Número de aglomerados insuficiente (encontrado: {len(clusters_locais)}).")
        logging.warning("São necessários pelo menos ~5 aglomerados para o split estratificado funcionar.")
        input("Pressione Enter para voltar...")
        return
        
    while True:
        limpar_tela()
        print("--- Execution Menu ---\n")
        print("[1] Run Data Analysis Stages (1-5)")
        print("[2] Run AI Training (Stage 6)")
        print("[3] Back to Main Menu")

        escolha_menu = input("\nSelect an option: ").strip()

        if escolha_menu == '1':
            gerenciar_etapas_analise(clusters_locais)
        elif escolha_menu == '2':
            gerenciar_treinamento_ia(clusters_locais)
        elif escolha_menu == '3':
            break
        else:
            input("Invalid option. Press Enter...")

# ==============================================================================
# 4. PONTO DE ENTRADA PRINCIPAL DO PROGRAMA (Sem alterações)
# ==============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('bateia_log.log', mode='a'), logging.StreamHandler(sys.stdout)]
    )
    
    while True:
        limpar_tela()
        print("="*60 + "\n      BATEIA-1 APPROACH - CONTROL PANEL\n" + "="*60)
        print("\n--- MAIN MENU ---")
        print("[1] Setup and Data Download")
        print("[2] Run Analysis and AI Pipeline")
        print("[3] Verify Local Clusters")
        print("[4] Reset Workspace")
        print("[5] Generate Results and Paper Reports")
        print("[6] About")
        print("[7] Exit")
        
        escolha = input("\nSelect an option: ").strip()
        
        if escolha == '1':
            setup_interativo()
        elif escolha == '2':
            gerenciar_execucao_pipeline()
        elif escolha == '3':
            limpar_tela()
            verificador.verificar_aglomerados_locais(config.BASE_DIR)
        elif escolha == '4':
            gerenciar_reset()
        elif escolha == '5':
            import subprocess
            subprocess.run([sys.executable, "-m", "bateia.tools.results_bateia"])
            input("Pressione Enter para continuar...")
        elif escolha == '6':      
            readme.exibir_sobre()
        elif escolha == '7':
            print("Exiting program.")
            break
        else:
            input("Invalid option. Press Enter to try again...")

    limpar_tela()
    print("="*60 + "\n      BATEIA-1 Program Terminated.\n" + "="*60)

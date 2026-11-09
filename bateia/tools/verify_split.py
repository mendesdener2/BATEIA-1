# bateia/tools/verify_split.py
\"\"\"
Script de Verificação de Split e Estratificação.
Garante que a lógica de divisão de dados (Treino/Val/Teste) seja consistente.
\"\"\"

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings

# Importa a configuração central do projeto BATEIA
from bateia.config import BASE_DIR

# Ignora avisos do K-Means (n_init)
warnings.filterwarnings('ignore', category=FutureWarning)

def verificar_split_para_n_igual_1(base_dir_path):
    \"\"\"
    Executa a simulação de split para n_test=1.
    Garante n_init=10 no KMeans para ser consistente com o pipeline principal.
    \"\"\"
    print(\"\\n\" + \"=\"*60)
    print(\"--- BATEIA: Verificação de Split (Estratégia n_test=1) ---\")
    print(\"=\"*60)
    
    # --- Carregar CSV ---
    params_csv_path = os.path.join(base_dir_path, \"cluster_params.csv\")
    if not os.path.exists(params_csv_path):
        print(f\"ERRO CRÍTICO: cluster_params.csv não encontrado em {params_csv_path}\")
        return

    try:
        df_params = pd.read_csv(params_csv_path)
        clusters_locais_todos = df_params['Cluster'].tolist()
        print(f\"[INFO] {len(clusters_locais_todos)} aglomerados carregados de {params_csv_path}\")
    except Exception as e:
        print(f\"[ERRO] Falha ao ler cluster_params.csv: {e}\")
        return
        
    # --- Fase 1: Encontrar os Perfis e o Teste (N=1 \"Mais Central\") ---
    print(\"\\nFase 1: Identificando aglomerado de Teste (Mais Central)...\")

    feature_cols = ['Rt', 'Rc', 'f0', 'fbg']
    df_features_all = df_params.set_index('Cluster')[feature_cols]
    
    scaler = StandardScaler()
    scaled_features_array = scaler.fit_transform(df_features_all)
    
    k_clusters = 3
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    group_labels = kmeans_model.fit_predict(scaled_features_array)
    
    df_all_clusters = df_features_all.copy()
    df_all_clusters['Group'] = group_labels
    
    # Encontrar o perfil principal (o maior)
    main_group_id = df_all_clusters['Group'].value_counts().idxmax()
    main_centroid = kmeans_model.cluster_centers_[main_group_id]
    
    df_temp = pd.DataFrame(scaled_features_array, index=df_all_clusters.index, columns=feature_cols)
    df_temp['Group'] = df_all_clusters['Group']

    # Calcular a distância até o centroide
    df_main_group = df_temp[df_temp['Group'] == main_group_id][feature_cols]
    distances = np.linalg.norm(df_main_group.values - main_centroid, axis=1)
    df_main_group['Distance'] = distances

    central_cluster_name = df_main_group['Distance'].idxmin()
    test_clusters = [central_cluster_name]
    dev_clusters = sorted([c for c in clusters_locais_todos if c != central_cluster_name])
    
    print(f\"[OK] Aglomerado central para Teste: {test_clusters[0]}\")
    
    # --- Fase 2: Estratificar o Desenvolvimento (19 -> 15/4) ---
    print(\"Fase 2: Estratificando Desenvolvimento (15 Treino / 4 Validação)...\")
    
    df_dev_with_groups = df_all_clusters.loc[dev_clusters]
    y_stratify_dev = df_dev_with_groups['Group']
    dev_set_clusters_np = np.array(dev_clusters)
    
    n_val = 4
    test_size_ratio = n_val / float(len(dev_clusters))

    try:
        train_clusters, val_clusters = train_test_split(
            dev_set_clusters_np,
            test_size=test_size_ratio,
            random_state=42,
            stratify=y_stratify_dev
        )
    except ValueError:
        train_clusters, val_clusters = train_test_split(
            dev_set_clusters_np,
            test_size=test_size_ratio,
            random_state=42
        )

    # --- Resultado Final ---
    print(\"\\n\" + \"=\"*60)
    print(\"--- RESULTADO DO SPLIT (REPRODUTIBILIDADE) ---\")
    print(\"=\"*60)
    print(f\"✅ Treino ({len(train_clusters)}): {sorted(list(train_clusters))}\")
    print(f\"✅ Validação ({len(val_clusters)}): {sorted(list(val_clusters))}\")
    print(f\"✅ Teste Final ({len(test_clusters)}): {test_clusters}\")
    print(\"=\"*60 + \"\\n\")

if __name__ == \"__main__\":
    # Usa o BASE_DIR da configuração oficial do BATEIA
    verificar_split_para_n_igual_1(BASE_DIR)

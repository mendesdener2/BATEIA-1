# results_bateia.py
# Script unificado para análise e geração de resultados para o artigo BATEIA.
# Substitui o antigo 'resultados_paper_mlp_svr.py'.

import pandas as pd
from bateia.tools.plotting import BateiaPlotter
import os
from bateia.config import BASE_DIR
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import logging
from typing import List, Dict, Tuple, Any

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS DO PROJETO
# ==============================================================================

CONFIG = {
    # --- Caminhos Base ---
    "PATHS": {
        "base_aglomerados": BASE_DIR,
        "base_redes_neurais_root": os.path.join(BASE_DIR,  "Redes Neurais", "TRY-1"),
        "output_base": os.path.join(BASE_DIR,  "Resultados_Paper"),
        "output_no-gap": os.path.join(BASE_DIR,  "Resultados_Paper", "no-gap"),
        "output_gap": os.path.join(BASE_DIR,  "Resultados_Paper", "gap"),
        "output_imgs_finais": os.path.join(BASE_DIR,  "Resultados_Paper", "Imagens"),
        "output_comparacao": os.path.join(BASE_DIR,  "Resultados_Paper", "comparacao-gap") 
    },
    
    # --- Definição de Clusters ---
    "CLUSTERS_BM": [
        "Berkeley98", "Collinder277", "Haffner13", "M67", "NGC129", "NGC188", "NGC581",
        "NGC2266", "NGC2422", "NGC2425", "NGC2516", "NGC3766", "NGC3960", "NGC6531",
        "NGC6791", "NGC6811", "NGC6940", "NGC7082", "Ruprecht91", "Ruprecht171"
    ],
    
    "CLUSTERS_AM": [],

    # --- Configuração do BATEIA/BATEIA (Extração) ---
    "BATEIA_PARAMS": {
        "modelos_bm": [
            ('MLPerceptron', 'BMMLP', 'bm_gap'),
            ('SVR', 'BMSVR', 'bm_gap'),
            ('RandomForest', 'BMRandomForest', 'bm_gap'),
            ('LSTM', 'BMLSTM', 'bm_gap'),
            ('GRU', 'BMGRU', 'bm_gap'),
            ('1DCNN', 'BM1DCNN', 'bm_gap')
        ],
        "modelos_am": [
            ('MLPerceptron', 'AMMLP', 'gap'),
            ('SVR', 'AMSVR', 'gap'),
            ('RandomForest', 'AMRandomForest', 'gap'),
            ('LSTM', 'AMLSTM', 'gap'),
            ('GRU', 'AMGRU', 'gap'),
            ('1DCNN', 'AM1DCNN', 'gap')
        ],
        # Configuração específica para o No-GAP (Apenas Sequenciais)
        "modelos_no_gap": [
            ('LSTM', 'AMLSTM', 'no-gap'),
            ('GRU', 'AMGRU', 'no-gap'),
            ('1DCNN', 'AM1DCNN', 'no-gap')
        ],
        "extratoras": ['VGG13', 'VGG16', 'VGG19', 'RESNET50', 'RESNET101', 'RESNET152', 
                       'DENSENET121', 'DENSENET161', 'DENSENET201'],
        "param_map": {'f0': 'f0', 'Rc': 'Rc', 'Rt': 'Rt', 'fbg': 'fbg'}
    },

    # --- Configuração Visual (Plotagem Escalar) ---
    "PLOT_STYLE_SCALAR": {
        "single": {
            "style": "seaborn-v0_8-whitegrid", "figsize": (5.12, 5.12), "dpi": 200,
            "cluster_color": '#1f77b4', "cluster_marker": 'D', "marker_size": 10,
            "error_bar_capsize": 5, "bateia_error_linestyle": ':',
            "hollow_marker_fill_color": 'white', "zorder_errorbar": 1, "zorder_marker": 2,
            "label_fontsize": 16, "tick_fontsize": 16, "legend_fontsize": 14, "xtick_rotation": 0
        },
        "grouped": {
            "style": "seaborn-v0_8-whitegrid", "figsize": (5.12, 5.12), "dpi": 200,
            "cluster_colors": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            "cluster_markers": ['D', 's', 'o', 'H', 'p'],
            "marker_size": 8, "error_bar_capsize": 5, "bateia_error_linestyle": ':',
            "hollow_marker_fill_color": 'white', "zorder_errorbar": 1, "zorder_marker": 2,
            "label_fontsize": 16, "tick_fontsize": 16, "legend_fontsize": 14, "xtick_rotation": 40
        }
    },

    # --- Configuração Visual (Plotagem Perfil King) ---
    "PLOT_STYLE_PROFILE": {
        "image_size_px": 512,
        "dpi": 120,
        "font_size": 18,
        "format": "jpg"
    }
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Mapeamentos auxiliares para localização de arquivos de modelos
MODEL_TAG_MAP = {
    'MLPerceptron': 'mlp', 'SVR': 'svr', 'RandomForest': 'randomforest',
    'LSTM': 'lstm', 'GRU': 'gru', '1DCNN': '1dcnn'
}

MODEL_FOLDER_MAP = {
    'MLPerceptron': 'MLPerceptron', 'SVR': 'SVR', 'RandomForest': 'RandomForest',
    'LSTM': 'LSTM', 'GRU': 'GRU', '1DCNN': '1DCNN'
}

# ==============================================================================
# LEITURA DE DADOS (COM CORREÇÃO DE CSV)
# ==============================================================================

def get_lit_obs_data(cluster_name):
    base = CONFIG["PATHS"]["base_aglomerados"]
    csv_path = os.path.join(base, 'cluster_params.csv')

    params = {}
    errors = {}
    params_hunt = {}
    obs_data = None

    def safe_float(val):
        try:
            if pd.isna(val) or str(val).strip() == '': return np.nan
            val_str = str(val).strip().replace(',', '.')
            return float(val_str)
        except Exception as e:
            logging.warning(f"Erro ao converter {val} para float: {e}")
            return np.nan

    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
            if 'Cluster' in df.columns:
                df['Cluster'] = df['Cluster'].astype(str).str.strip()

            row = df[df['Cluster'] == cluster_name]

            if not row.empty:
                for p in ['f0', 'Rc', 'Rt', 'fbg']:
                    if p in row.columns:
                        params[p] = safe_float(row[p].values[0])
                    else:
                        params[p] = 0.0
                    
                    err_col = f"err_{p}"
                    if err_col in row.columns:
                        errors[p] = safe_float(row[err_col].values[0])
                    else:
                        errors[p] = 0.0

                    hunt_col = f"hunt_{p}"
                    if hunt_col in row.columns:
                        params_hunt[p] = safe_float(row[hunt_col].values[0])
            else:
                return None, None, None, None
        else:
            return None, None, None, None
            
    except Exception as e:
        logging.error(f"Erro leitura CSV cluster_params: {e}")
        return None, None, None, None

    try:
        div_path = os.path.join(base, cluster_name, "Target", f"best_division_info_{cluster_name}.csv")
        if os.path.exists(div_path):
            best_div_df = pd.read_csv(div_path)
            if not best_div_df.empty:
                best_div = int(best_div_df['Best Division'].iloc[0])
                con_path = os.path.join(base, cluster_name, "Resultados", f"{cluster_name}_parametros_king_consolidados.csv")
                if os.path.exists(con_path):
                    df_con = pd.read_csv(con_path)
                    row_topo = df_con.loc[df_con['Divisoes'] == best_div]
                    if not row_topo.empty:
                        topo_id = row_topo['Topologia_ID'].iloc[0]
                        drad_path = os.path.join(base, cluster_name, "Resultados", "Densidade radial", f"drad_{cluster_name}_{int(topo_id)}.csv")
                        if os.path.exists(drad_path):
                            df_drad = pd.read_csv(drad_path)
                            obs_data = (df_drad['Bin_Center (arcmin)'].values, df_drad['Density (stars/arcmin²)'].values)
    except: obs_data = None
    
    return params, errors, params_hunt, obs_data

# ==============================================================================
# MÓDULO 1: EXTRAÇÃO E RESUMO
# ==============================================================================

def load_literature_data(file_path: str) -> pd.DataFrame:
    print(f"\n--- Carregando Dados da Literatura ---")
    if not os.path.exists(file_path):
        logging.error(f"Arquivo de parâmetros '{file_path}' não encontrado.")
        return None
    try:
        df_lit_full = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
        df_lit_full.columns = df_lit_full.columns.str.strip()
        if 'Cluster' in df_lit_full.columns:
            df_lit_full['Cluster'] = df_lit_full['Cluster'].astype(str).str.strip()
            
        print(f"Dados da literatura carregados. Total de {len(df_lit_full)} clusters.")
        return df_lit_full
    except Exception as e:
        logging.error(f"Falha ao ler o arquivo de literatura '{file_path}': {e}")
        return None

def correct_literature_in_results(results_list: List[Dict], df_lit: pd.DataFrame, param_map: Dict):
    if not results_list or df_lit is None: return
    for item in results_list:
        c_name = item['Cluster']
        lit_row = df_lit[df_lit['Cluster'] == c_name]
        if not lit_row.empty:
            r = lit_row.iloc[0]
            for p_key in param_map.keys():
                if p_key in item and isinstance(item[p_key], dict):
                    val_lit = "0"
                    if p_key in r: val_lit = r[p_key]
                    elif p_key.lower() in r: val_lit = r[p_key.lower()]
                    try: v_float = float(str(val_lit).replace(',', '.'))
                    except: v_float = 0.0
                    item[p_key]['value_lit'] = v_float

def find_best_bateia_prediction(base_redes_neurais: str, cluster: str, param_bateia: str, 
                              modelos: List[Tuple[str, str, str]], 
                              extratoras: List[str]) -> Dict[str, Any]:
    best_result = {
        'value': np.nan, 'value_lit': np.nan, 'file': None, 'model': None, 
        'extractor': None, 'score': -1, 'error': float('inf')
    }

    for modelo, sigla_metodologia, sigla_split in modelos:
        # Determina o rótulo GAP/NO-GAP com base na sigla_split ou metodologia
        gap_folder = "NO-GAP" if "no-gap" in sigla_split.lower() else "GAP"
        model_base_folder = MODEL_FOLDER_MAP.get(modelo, modelo)
        
        for extratora in extratoras:
            # Novo caminho estruturado: TRY-1 / [GAP|NO-GAP] / [MLP|SVR|...] / [AMMLP|BMMLP|...] / Previsto / [VGG16|...]
            base_model_path = os.path.join(base_redes_neurais, gap_folder, model_base_folder, sigla_metodologia, 'Previsto', extratora.upper())
            if not os.path.exists(base_model_path): continue

            mod_tag = MODEL_TAG_MAP.get(modelo, modelo.lower())
            gap_tag = gap_folder.lower()

            # Padrão BM: predicoes_reais_f0_vgg16_bm_gap.csv
            file_name_bm = f"predicoes_reais_{param_bateia.lower()}_{extratora.lower()}_bm_{gap_tag}.csv"
            path_bm = os.path.join(base_model_path, file_name_bm)
            
            # Padrão AM: predicoes_reais_f0_vgg16_lstm_M67_gap.csv
            file_name_am = f"predicoes_reais_{param_bateia.lower()}_{extratora.lower()}_{mod_tag}_{cluster}_{gap_tag}.csv"
            path_am = os.path.join(base_model_path, file_name_am)

            file_path = None
            if os.path.exists(path_bm): file_path = path_bm
            elif os.path.exists(path_am): file_path = path_am
            
            if not file_path: continue
                
            try:
                # Leitura robusta com auto-detecção de separador e remoção de espaços
                df_pred = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
                df_pred.columns = df_pred.columns.str.strip()
                if 'Cluster' in df_pred.columns:
                    df_pred['Cluster'] = df_pred['Cluster'].astype(str).str.strip()

                cluster_row = df_pred[df_pred['Cluster'] == cluster]
                
                if cluster_row.empty: continue
                
                cluster_data = cluster_row.iloc[0]
                approved_status = cluster_data.get('Approved')
                error_perc = cluster_data.get('Error_%')
                
                col_name_specific = f"{extratora.lower()}_{param_bateia}" 
                predicted_val = cluster_data.get(col_name_specific)
                
                if pd.isna(predicted_val): predicted_val = cluster_data.get(param_bateia) 

                # Conversão segura para float (trata vírgula como decimal se necessário)
                try:
                    predicted_val = float(str(predicted_val).replace(',', '.'))
                except:
                    predicted_val = np.nan

                literature_val = 0.0 

                if pd.isna(predicted_val): continue

                current_score = 1 if approved_status == 'Yes' else 0
                
                # CORREÇÃO: Lê o erro percentual direto (sem multiplicar)
                current_error = float(str(error_perc).replace(',', '.')) if pd.notna(error_perc) else float('inf')

                if current_score > best_result['score']:
                    best_result.update({'value': predicted_val, 'value_lit': literature_val, 'file': os.path.basename(file_path),
                                        'model': sigla_metodologia, 'extractor': extratora.upper(), 'score': current_score, 'error': current_error})
                elif current_score == best_result['score'] and current_error < best_result['error']:
                    best_result.update({'value': predicted_val, 'value_lit': literature_val, 'file': os.path.basename(file_path),
                                        'model': sigla_metodologia, 'extractor': extratora.upper(), 'score': current_score, 'error': current_error})
                        
            except Exception as e: continue

    return best_result

def load_bateia_data_single_cluster(base_redes_neurais: str, cluster_alvo: str, 
                                  modelos: List[Tuple[str, str, str]], 
                                  extratoras: List[str], 
                                  param_map: Dict[str, str]) -> Dict[str, Any]:
    bateia_params_found = {'Cluster': cluster_alvo}
    for param_key in param_map.keys():
        param_bateia_file = param_map[param_key]
        best_val = find_best_bateia_prediction(base_redes_neurais, cluster_alvo, param_bateia_file, modelos, extratoras)
        bateia_params_found[param_key] = best_val  
    return bateia_params_found

def save_summary_csv(results_list: List[Dict], output_path: str, prefix: str, param_order: List[str]):
    if not results_list: return
    print(f"--- Preparando e salvando resumo {prefix.upper()} ---")
    flat_results_list = []
    for cluster_data in results_list:
        cluster_name = cluster_data['Cluster']
        for param_key, details in cluster_data.items():
            if param_key == 'Cluster' or not isinstance(details, dict): continue
            flat_results_list.append({
                'Cluster': cluster_name, 'Param': param_key,
                'Literature' : details.get('value_lit'), f'BATEIA-1 ({prefix.upper()})': details.get('value'),
                'Model': details.get('model'), 'Extractor': details.get('extractor'), 'Error_%': details.get('error')
            })
            
    df_resumo = pd.DataFrame(flat_results_list)
    cols_order = ['Cluster', 'Param', 'Literature', f'BATEIA-1 ({prefix.upper()})', 'Model', 'Extractor', 'Error_%']
    for col in cols_order:
        if col not in df_resumo.columns: df_resumo[col] = np.nan
    df_resumo = df_resumo[cols_order]
    
    try:
        os.makedirs(output_path, exist_ok=True)
        csv_filename = os.path.join(output_path, f"{prefix.lower()}_best_predictions_summary.csv")
        with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
            header = ';'.join(cols_order) + '\n'
            f.write(header)
            is_first_param = True
            for param_key in param_order:
                if not is_first_param: f.write('\n')
                else: is_first_param = False
                df_param = df_resumo[df_resumo['Param'] == param_key].sort_values('Cluster')
                df_param.to_csv(f, sep=';', decimal=',', index=False, header=False, lineterminator='\n')
        print(f"Resumo {prefix.upper()} salvo em: {csv_filename}")
    except Exception as e:
        logging.error(f"Falha ao salvar CSV {prefix.upper()}: {e}")

def save_best_extractors_report(results_list: List[Dict], param_order: List[str], prefix: str):
    target_dir = os.path.dirname(CONFIG["PATHS"]["output_base"]) 
    output_file = os.path.join(target_dir, f"{prefix.lower()}_best_extractors_count.txt")
    
    stats = {p: {} for p in param_order}

    for cluster_data in results_list:
        for param in param_order:
            if param in cluster_data and isinstance(cluster_data[param], dict):
                details = cluster_data[param]
                ext, mod = details.get('extractor'), details.get('model')
                err = details.get('error', 0)
                if not ext or not mod: continue
                
                combo_key = f"{ext} + {mod}"
                if combo_key not in stats[param]: stats[param][combo_key] = {'count': 0, 'sum_error': 0.0}
                stats[param][combo_key]['count'] += 1
                stats[param][combo_key]['sum_error'] += err

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"RELATORIO DE CONTAGEM (VENCEDORES INDIVIDUAIS) - {prefix.upper()}\n")
            f.write("="*60 + "\n\n")
            for param in param_order:
                f.write(f"PARAMETRO: {param}\n")
                for combo_key, metrics in stats[param].items():
                     metrics['avg_error'] = metrics['sum_error'] / metrics['count']
                candidates = sorted(stats[param].items(), key=lambda x: (-x[1]['count'], x[1]['avg_error']))
                if candidates:
                    best_name, best_metrics = candidates[0]
                    f.write(f" >> CAMPEAO: {best_name} ({best_metrics['count']} vitorias)\n")
                f.write("\n")
        print(f"Relatório de Contagem salvo em: {output_file}")
    except Exception as e: logging.error(f"Erro relatório contagem: {e}")

def generate_generalization_report_BM(clusters_bm: List[str], param_map: Dict[str, str]):
    print("\n>>> Gerando Relatório Detalhado de Generalização (BM - Per Cluster) <<<")
    
    base_nn = CONFIG["PATHS"]["base_redes_neurais_root"]
    modelos = CONFIG["BATEIA_PARAMS"]["modelos_bm"]
    extratoras = CONFIG["BATEIA_PARAMS"]["extratoras"]
    
    stats = {param: {} for param in param_map.keys()}
    
    for param_key, param_file in param_map.items():
        for modelo_pasta, sigla_metodologia, sigla_split in modelos:
            gap_folder = "NO-GAP" if "no-gap" in sigla_split.lower() else "GAP"
            model_base_folder = MODEL_FOLDER_MAP.get(modelo_pasta, modelo_pasta)
            
            for ext in extratoras:
                combo_key = f"{ext} + {sigla_metodologia}"
                gap_tag = gap_folder.lower()
                
                # Padrão BM: predicoes_reais_f0_vgg16_bm_gap.csv
                filename = f"predicoes_reais_{param_file.lower()}_{ext.lower()}_bm_{gap_tag}.csv"
                filepath = os.path.join(base_nn, gap_folder, model_base_folder, sigla_metodologia, 'Previsto', ext.upper(), filename)
                
                if not os.path.exists(filepath): continue
                
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8-sig')
                    df.columns = df.columns.str.strip()
                    if 'Cluster' in df.columns:
                        df['Cluster'] = df['Cluster'].astype(str).str.strip()

                    df_bm = df[df['Cluster'].isin(clusters_bm)]
                    
                    if df_bm.empty: continue
                    
                    cluster_errors = {}
                    for _, row in df_bm.iterrows():
                        c_name = row['Cluster']
                        err_val = row['Error_%']
                        try:
                            # CORREÇÃO: Lê erro direto
                            val_float = float(str(err_val).replace(',', '.')) if pd.notna(err_val) else None
                            if val_float is not None:
                                cluster_errors[c_name] = val_float
                        except: pass
                    
                    if cluster_errors:
                        stats[param_key][combo_key] = cluster_errors

                except Exception: continue

    target_dir = os.path.dirname(CONFIG["PATHS"]["output_imgs_finais"])
    output_file = os.path.join(target_dir, "bm_generalization_detailed.txt")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATORIO DETALHADO DE GENERALIZACAO - GRUPO BM\n")
            f.write("Objetivo: Identificar robustez e outliers (clusters discrepantes).\n")
            f.write("="*100 + "\n\n")
            
            for param in param_map.keys():
                f.write(f"PARAMETRO: {param}\n")
                f.write("-" * 40 + "\n")
                
                candidates = []
                for combo, err_dict in stats[param].items():
                    vals = list(err_dict.values())
                    if not vals: continue
                    avg = np.mean(vals)
                    std = np.std(vals)
                    worst = np.max(vals)
                    candidates.append((combo, avg, std, worst, err_dict))
                
                candidates.sort(key=lambda x: x[1])
                
                if not candidates:
                    f.write("  Nenhum dado processado.\n")
                else:
                    for i, (combo, avg, std, worst, err_dict) in enumerate(candidates[:5], 1):
                        f.write(f"  {i}. {combo}\n")
                        f.write(f"     [GLOBAL] Media: {avg:.2f}% | Desvio: {std:.2f} | Pior Caso: {worst:.2f}%\n")
                        f.write(f"     [DETALHE POR CLUSTER]:\n")
                        
                        sorted_clusters = sorted(clusters_bm)
                        buffer_line = []
                        for c_name in sorted_clusters:
                            val = err_dict.get(c_name, float('nan'))
                            val_str = f"{val:.2f}%" if pd.notna(val) else "N/A"
                            marker = " (!)" if (pd.notna(val) and val > 20.0) else "" 
                            buffer_line.append(f"{c_name}: {val_str}{marker}")
                            
                            if len(buffer_line) == 4:
                                f.write("         " + " | ".join(buffer_line) + "\n")
                                buffer_line = []
                        if buffer_line:
                            f.write("         " + " | ".join(buffer_line) + "\n")
                        
                        f.write("\n")
                f.write("\n" + "="*100 + "\n\n")
        
        print(f"Relatório detalhado salvo em: {output_file}")
            
    except Exception as e:
        logging.error(f"Erro ao salvar relatório detalhado BM: {e}")

def generate_generalization_report_AM(clusters_am: List[str], param_map: Dict[str, str]):
    print("\n>>> Gerando Relatório Detalhado de Generalização (AM - Per Cluster) <<<")
    
    base_nn = CONFIG["PATHS"]["base_redes_neurais_root"]
    modelos = CONFIG["BATEIA_PARAMS"]["modelos_am"]
    extratoras = CONFIG["BATEIA_PARAMS"]["extratoras"]
    
    stats = {param: {} for param in param_map.keys()}
    
    for param_key, param_file in param_map.items():
        for modelo_pasta, sigla_metodologia, sigla_split in modelos:
            mod_tag = MODEL_TAG_MAP.get(modelo_pasta, modelo_pasta.lower())
            gap_folder = "NO-GAP" if "no-gap" in sigla_split.lower() else "GAP"
            model_base_folder = MODEL_FOLDER_MAP.get(modelo_pasta, modelo_pasta)

            for ext in extratoras:
                combo_key = f"{ext} + {sigla_metodologia}"
                gap_tag = gap_folder.lower()
                
                base_model_path = os.path.join(base_nn, gap_folder, model_base_folder, sigla_metodologia, 'Previsto', ext.upper())
                
                if not os.path.exists(base_model_path): continue

                for cluster in clusters_am:
                    # Padrão AM: predicoes_reais_f0_vgg16_lstm_M67_gap.csv
                    filename = f"predicoes_reais_{param_file.lower()}_{ext.lower()}_{mod_tag}_{cluster}_{gap_tag}.csv"
                    filepath = os.path.join(base_model_path, filename)
                    
                    if not os.path.exists(filepath):
                         filename_fallback = f"predicoes_reais_{param_file.lower()}_{ext.lower()}_{sigla_split}.csv"
                         filepath_fallback = os.path.join(base_model_path, filename_fallback)
                         if os.path.exists(filepath_fallback):
                             filepath = filepath_fallback
                    
                    if not os.path.exists(filepath): continue
                    
                    try:
                        df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8-sig')
                        df.columns = df.columns.str.strip()
                        if 'Cluster' in df.columns:
                            df['Cluster'] = df['Cluster'].astype(str).str.strip()

                        row = df[df['Cluster'] == cluster]
                        
                        if not row.empty:
                            err_val = row['Error_%'].values[0]
                            if pd.notna(err_val):
                                try: 
                                    # CORREÇÃO: Lê erro direto
                                    err_float = float(str(err_val).replace(',', '.'))
                                    if combo_key not in stats[param_key]: stats[param_key][combo_key] = {}
                                    stats[param_key][combo_key][cluster] = err_float
                                except: pass
                    except Exception as e: continue

    target_dir = os.path.dirname(CONFIG["PATHS"]["output_imgs_finais"])
    output_file = os.path.join(target_dir, "am_generalization_detailed.txt")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATORIO DETALHADO DE GENERALIZACAO - METODOLOGIA AM\n")
            f.write(f"Clusters: {', '.join(clusters_am)}\n")
            f.write("Objetivo: Identificar robustez e outliers (clusters discrepantes).\n")
            f.write("="*100 + "\n\n")
            
            for param in param_map.keys():
                f.write(f"PARAMETRO: {param}\n")
                f.write("-" * 40 + "\n")
                
                candidates = []
                for combo, err_dict in stats[param].items():
                    vals = list(err_dict.values())
                    if not vals: continue

                    avg = np.mean(vals)
                    std = np.std(vals)
                    worst = np.max(vals)
                    count = len(vals)
                    candidates.append((combo, avg, std, worst, count, err_dict))
                
                candidates.sort(key=lambda x: (-x[4], x[1])) 
                
                if not candidates:
                    f.write("  Nenhum modelo encontrou predicoes validas.\n")
                else:
                    for i, (combo, avg, std, worst, count, err_dict) in enumerate(candidates[:10], 1):
                        f.write(f"  {i}. {combo} (Clusters: {count}/{len(clusters_am)})\n")
                        f.write(f"     [GLOBAL] Media: {avg:.2f}% | Desvio: {std:.2f} | Pior Caso: {worst:.2f}%\n")
                        
                        str_parts = []
                        for c_name in clusters_am:
                            val = err_dict.get(c_name)
                            if val is not None:
                                marker = " (!)" if val > 20.0 else ""
                                str_parts.append(f"{c_name}: {val:.2f}%{marker}")
                            else:
                                str_parts.append(f"{c_name}: --")
                        
                        f.write(f"     [DETALHE]: {' | '.join(str_parts)}\n\n")
                
                f.write("\n" + "="*100 + "\n\n")
        
        print(f"Relatório de Generalização AM salvo em: {output_file}")
            
    except Exception as e:
        logging.error(f"Erro ao salvar relatório de generalização AM: {e}")

# ==============================================================================
# MÓDULO 2: PLOTAGEM
# ==============================================================================

def funcao_king(r, f0, Rc, Rt, fbg):
    if Rc <= 0 or Rt <= Rc: return np.full_like(r, fbg if fbg > 0 else 0)
    try:
        termo_r, termo_t = np.sqrt(1 + (r / Rc)**2), np.sqrt(1 + (Rt / Rc)**2)
        mask = (r <= Rt)
        king_term = np.zeros_like(r, dtype=float)
        king_term[mask] = (1.0 / termo_r[mask]) - (1.0 / termo_t)
        king_term[king_term < 0] = 0
        return fbg + f0 * (king_term**2)
    except: return np.full_like(r, fbg if fbg > 0 else 0)

def get_bm_params(cluster_name):
    csv_path = os.path.join(CONFIG["PATHS"]["output_base"], "BM_Groups", "bm_best_predictions_summary.csv")
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        df_c = df[df['Cluster'] == cluster_name]
        if df_c.empty: return None
        params = {}
        for p in ['f0', 'Rc', 'Rt', 'fbg']:
            row = df_c[df_c['Param'] == p]
            if not row.empty: params[p] = float(row[f'BATEIA-1 (BM)'].values[0])
        return params if len(params)==4 else None
    except: return None

def get_am_params(cluster_name):
    csv_path = os.path.join(CONFIG["PATHS"]["output_base"], "AM_Single", cluster_name, "am_best_predictions_summary.csv")
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        if 'Cluster' in df.columns: df = df[df['Cluster'] == cluster_name]
        params = {}
        for p in ['f0', 'Rc', 'Rt', 'fbg']:
            row = df[df['Param'] == p]
            if not row.empty: params[p] = float(row[f'BATEIA-1 (AM)'].values[0])
        return params if len(params)==4 else None
    except: return None

def gerar_graficos_perfil(cluster, params_bateia, params_lit, errors_lit, params_hunt, dados_obs, label_bateia):
    if dados_obs is None: return
    bin_centers, density_obs = dados_obs
    r_max = bin_centers.max() * 1.05
    r_cont = np.linspace(0, r_max, 500)
    
    cfg = CONFIG["PLOT_STYLE_PROFILE"]
    out_folder = os.path.join(CONFIG["PATHS"]["output_imgs_finais"], cluster)
    os.makedirs(out_folder, exist_ok=True)
    suffix = "_BM" if "BM" in label_bateia else "_AM"
    
    y_bateia = funcao_king(r_cont, **params_bateia)
    y_lit = funcao_king(r_cont, **params_lit)
    y_lit_lo = funcao_king(r_cont, **{k: params_lit[k]-errors_lit[k] for k in params_lit})
    y_lit_hi = funcao_king(r_cont, **{k: params_lit[k]+errors_lit[k] for k in params_lit})
    
    figsize = (1.4 * cfg["image_size_px"]/cfg["dpi"], cfg["image_size_px"]/cfg["dpi"])
    
    props = {
        'Rc':  {'c': 'green', 'lbl': 'Core radius'},
        'Rt':  {'c': 'red',   'lbl': 'Tidal radius'},
        'fbg': {'c': 'gold',  'lbl': 'Background \n density'},
        'uncertainty': {'c': 'gray', 'lbl': 'Literature uncertainty'}
    }
    hunt_style = {'c': 'magenta', 'ls': (0, (1, 1)), 'lbl': 'Hunt et al.'} 

    # 1. King Fit Compare
    plt.figure(figsize=figsize, dpi=cfg["dpi"])
    plt.suptitle(f'{cluster}: King Fit (BATEIA-1 and literature)', fontsize=cfg["font_size"]-2, x=0.5, ha='center')
    plt.plot(bin_centers, density_obs, 'ko', label='Observed', zorder=3)
    plt.plot(r_cont, y_bateia, 'b--', label=f'BATEIA-1 Fit', zorder=4)
    plt.plot(r_cont, y_lit, 'r-', label='Angelo et al. fit', zorder=4)
    plt.fill_between(r_cont, y_bateia, y_lit, color='gray', alpha=0.3, label='Difference', zorder=1)
    plt.xlabel('Radial distance (r) [arcmin]', fontsize=cfg["font_size"]); plt.ylabel('Surface density [stars / arcmin²]', fontsize=cfg["font_size"])
    plt.legend(fontsize=cfg["font_size"]-6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_folder, f"king_fit_compare_{cluster}{suffix}.{cfg['format']}"))
    plt.close()

    # 2. Params Compare
    fig, ax = plt.subplots(figsize=figsize, dpi=cfg["dpi"])
    fig.suptitle(f'{cluster}: Params (BATEIA-1, Angelo & Hunt)', fontsize=cfg["font_size"]-2, x=0.5, ha='center')
    ax.plot(bin_centers, density_obs, 'ko', alpha=1)
    for k in ['Rc', 'Rt', 'fbg']:
        c = props[k]['c']
        if k=='fbg': ax.axhline(params_bateia[k], color=c, ls='--', linewidth=1.5, alpha=1)
        else:        ax.axvline(params_bateia[k], color=c, ls='--', linewidth=1.5, alpha=1)
        if k=='fbg': ax.axhline(params_lit[k], color=c, ls='-', alpha=1)
        else:        ax.axvline(params_lit[k], color=c, ls='-', alpha=1)
        if params_hunt:
            if k=='fbg': ax.axhline(params_hunt[k], color=c, ls=hunt_style['ls'], linewidth=2, alpha=1)
            else:        ax.axvline(params_hunt[k], color=c, ls=hunt_style['ls'], linewidth=2, alpha=1)
    ax.set_xlabel('Radial distance (r) [arcmin]', fontsize=cfg["font_size"]); ax.set_ylabel('Surface density [stars / arcmin²]', fontsize=cfg["font_size"])
    legend_elements = [
        plt.Line2D([0], [0], color='black', ls='--', label='BATEIA-1 (Dashed)'),
        plt.Line2D([0], [0], color='black', ls='-', label='Angelo et al. (Solid)'),
        plt.Line2D([0], [0], color='black', ls=hunt_style['ls'], label='Hunt et al. (Dotted)'),
        plt.Line2D([0], [0], color=props['Rc']['c'], lw=2, label=props['Rc']['lbl']),
        plt.Line2D([0], [0], color=props['Rt']['c'], lw=2, label=props['Rt']['lbl']),
        plt.Line2D([0], [0], color=props['fbg']['c'], lw=2, label=props['fbg']['lbl'])
    ]
    ax.legend(handles=legend_elements, fontsize=cfg["font_size"]-6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_folder, f"params_compare_{cluster}{suffix}.{cfg['format']}"))
    plt.close()

    # 3. King Uncertainty
    plt.figure(figsize=figsize, dpi=cfg["dpi"])
    plt.suptitle(f'{cluster}: BATEIA-1 and literature', fontsize=cfg["font_size"]-2, x=0.5, ha='center')
    plt.plot(bin_centers, density_obs, 'ko', zorder=3)
    plt.fill_between(r_cont, y_lit_lo, y_lit_hi, color=props['uncertainty']['c'], alpha=0.3, zorder=1, label='Angelo et al. \n uncertainty')
    plt.plot(r_cont, y_bateia, 'b--', zorder=4, label='BATEIA-1 Fit')
    plt.legend(fontsize=cfg["font_size"]-6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Radial distance (r) [arcmin]', fontsize=cfg["font_size"]); plt.ylabel('Surface density [stars / arcmin²]', fontsize=cfg["font_size"])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_folder, f"king_uncertainty_{cluster}{suffix}.{cfg['format']}"))
    plt.close()
    
    # 4. Params Uncertainty
    fig, ax = plt.subplots(figsize=figsize, dpi=cfg["dpi"])
    fig.suptitle(f'{cluster}: BATEIA-1, Angelo and Hunt params', fontsize=cfg["font_size"]-2, x=0.5, ha='center')
    ax.plot(bin_centers, density_obs, 'ko', alpha=1)
    ax.axvspan(params_lit['Rc']-errors_lit['Rc'], params_lit['Rc']+errors_lit['Rc'], color=props['Rc']['c'], alpha=0.2)
    ax.axvspan(params_lit['Rt']-errors_lit['Rt'], params_lit['Rt']+errors_lit['Rt'], color=props['Rt']['c'], alpha=0.2)
    ax.axhspan(params_lit['fbg']-errors_lit['fbg'], params_lit['fbg']+errors_lit['fbg'], color=props['fbg']['c'], alpha=0.2)
    for k in ['Rc', 'Rt', 'fbg']:
        ax.axvline(params_bateia['Rc'], color=props['Rc']['c'], ls='--', alpha=1)
        ax.axvline(params_bateia['Rt'], color=props['Rt']['c'], ls='--', alpha=1)
        ax.axhline(params_bateia['fbg'], color=props['fbg']['c'], ls='--', alpha=1)
        if params_hunt:
             if k=='fbg': ax.axhline(params_hunt[k], color=props[k]['c'], ls=hunt_style['ls'], linewidth=2, alpha=0.8)
             else:        ax.axvline(params_hunt[k], color=props[k]['c'], ls=hunt_style['ls'], linewidth=2, alpha=0.8)
        if params_lit:
            if k=='fbg': ax.axhline(params_lit[k], color=props[k]['c'], ls='-', alpha=0.8)
            else:        ax.axvline(params_lit[k], color=props[k]['c'], ls='-', alpha=0.8)
    ax.set_xlabel('Radial distance (r) [arcmin]', fontsize=cfg["font_size"]); ax.set_ylabel('Surface density [stars / arcmin²]', fontsize=cfg["font_size"])
    legend_elements_4 = [
        plt.Line2D([0], [0], color=props['uncertainty']['c'], alpha=0.3, lw=4, label='Angelo et. al. \n uncertainty'),
        plt.Line2D([0], [0], color='black', ls='--', label='BATEIA-1 value'),
        plt.Line2D([0], [0], color='black', ls=hunt_style['ls'], label='Hunt et al.')
    ]
    ax.legend(handles=legend_elements_4, fontsize=cfg["font_size"]-6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_folder, f"params_uncertainty_{cluster}{suffix}.{cfg['format']}"))
    plt.close()
    
    print(f" -> Gráficos salvos para {cluster} (BATEIA-1 and literature)")

def generate_grouped_plot(param: str, df_grupo: pd.DataFrame, plot_config: Dict[str, Any], output_path: str, grupo_num: int, legend_label: str = "BATEIA-1"):
    ylabel_map = {'Rc': 'Core radius ($r_c$)', 'Rt': 'Tidal radius ($r_t$)', 'f0': r'Central density ($\sigma_0$)', 'fbg': r'Background density ($\sigma_{bg}$)'}
    try:
        plt.style.use(plot_config["style"])
        fig, ax = plt.subplots(figsize=plot_config["figsize"])
        offset = 0.1
        
        ax.errorbar([], [], color='gray', marker='s', ls='None', markersize=plot_config['marker_size'], label='Angelo et. al.')
        ax.errorbar([], [], color='gray', marker='s', ls='None', markersize=plot_config['marker_size'], mfc='white', mec='gray', label=legend_label)

        for j, (index, row) in enumerate(df_grupo.iterrows()):
            lit_val, bateia_val, lit_err = row.get(f'{param}_lit'), row.get(f'{param}_bateia'), row.get(f'err_{param}_lit')
            if pd.isna(lit_val) or pd.isna(bateia_val): continue
            
            bateia_err = (lit_err / lit_val) * bateia_val if lit_val > 0 else 0
            color = plot_config["cluster_colors"][j % len(plot_config["cluster_colors"])]
            marker = plot_config["cluster_markers"][j % len(plot_config["cluster_markers"])]
            
            ax.errorbar(j - offset, lit_val, yerr=lit_err, ls='None', marker=marker, color=color, markersize=plot_config["marker_size"], capsize=plot_config["error_bar_capsize"])
            ax.errorbar(j + offset, bateia_val, yerr=bateia_err, fmt='none', ecolor=color, capsize=plot_config["error_bar_capsize"], ls=plot_config["bateia_error_linestyle"], zorder=1)
            ax.plot(j + offset, bateia_val, marker=marker, ls='None', mfc='white', mec=color, markersize=plot_config["marker_size"], zorder=2)
        
        ax.set_ylabel(ylabel_map.get(param, param), fontsize=plot_config["label_fontsize"])
        ax.set_xticks(np.arange(len(df_grupo)))
        ax.set_xticklabels(df_grupo['Cluster'], rotation=plot_config["xtick_rotation"], ha='right', fontsize=plot_config["tick_fontsize"])
        fig.tight_layout()
        ax.legend(fontsize=plot_config["legend_fontsize"])
        out_dir = os.path.join(output_path, f"Group {grupo_num}")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'comparacao_bm_{param}_grupo_{grupo_num}.jpg'), dpi=plot_config["dpi"])
        plt.close(fig)
    except Exception:
        if 'fig' in locals(): plt.close(fig)

def generate_single_cluster_plot(param: str, lit_val: float, lit_err: float, bateia_val: float, cluster_alvo: str, 
                                 plot_config: Dict[str, Any], output_path: str, bateia_legend_label: str):
    ylabel_map = {'Rc': 'Core radius ($r_c$)', 'Rt': 'Tidal radius ($r_t$)', 'f0': r'Central density ($\sigma_0$)', 'fbg': r'Background density ($\sigma_{bg}$)'}
    try:
        bateia_err = (lit_err / lit_val) * bateia_val if lit_val > 0 else 0
        plt.style.use(plot_config["style"])
        fig, ax = plt.subplots(figsize=plot_config["figsize"])
        
        offset, x_pos = 0.15, 0
        color, marker = plot_config["cluster_color"], plot_config["cluster_marker"]

        ax.errorbar([], [], color='gray', marker=marker, ls='None', markersize=plot_config['marker_size'], label='Angelo et. al.')
        ax.errorbar([], [], color='gray', marker=marker, ls='None', markersize=plot_config['marker_size'], mfc='white', mec='gray', label=bateia_legend_label)

        ax.errorbar(x_pos - offset, lit_val, yerr=lit_err, ls='None', marker=marker, color=color, markersize=plot_config["marker_size"], capsize=plot_config["error_bar_capsize"])
        ax.errorbar(x_pos + offset, bateia_val, yerr=bateia_err, fmt='none', ecolor=color, capsize=plot_config["error_bar_capsize"], ls=plot_config["bateia_error_linestyle"], zorder=1)
        ax.plot(x_pos + offset, bateia_val, marker=marker, ls='None', mfc='white', mec=color, markersize=plot_config["marker_size"], zorder=2)
        
        ax.set_ylabel(ylabel_map.get(param, param), fontsize=plot_config["label_fontsize"])
        ax.set_xticks([x_pos]); ax.set_xticklabels([cluster_alvo], fontsize=plot_config["tick_fontsize"])
        ax.tick_params(axis='y', labelsize=plot_config["tick_fontsize"])
        ax.legend(fontsize=plot_config["legend_fontsize"])
        ax.xaxis.grid(False)
        fig.tight_layout()
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f'comparacao_am_{param}_{cluster_alvo}.jpg'), dpi=plot_config["dpi"])
        plt.close(fig)
    except Exception:
        if 'fig' in locals(): plt.close(fig)





# ==============================================================================
# MÓDULO 3: COMPARAÇÃO GAP VS NO-GAP (NOVO)
# ==============================================================================

def get_nogap_params(cluster_name):
    """
    Lê os parâmetros do CSV de resumo gerado especificamente para o NO-GAP.
    """
    csv_path = os.path.join(CONFIG["PATHS"]["output_comparacao"], "nogap_best_predictions_summary.csv")
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        if 'Cluster' in df.columns: df = df[df['Cluster'] == cluster_name]
        params = {}
        for p in ['f0', 'Rc', 'Rt', 'fbg']:
            row = df[df['Param'] == p]
            # Nota: o prefixo da coluna será 'BATEIA-1 (NOGAP)' baseado na função de salvamento
            if not row.empty: params[p] = float(row[f'BATEIA-1 (NOGAP)'].values[0])
        return params if len(params)==4 else None
    except: return None

def generate_gap_comparison_report(clusters_am: List[str], gap_data: List[Dict], nogap_data: List[Dict]):
    """
    Gera um relatório txt comparando o erro percentual do melhor modelo GAP vs melhor modelo No-GAP.
    """
    output_file = os.path.join(CONFIG["PATHS"]["output_comparacao"], "relatorio_rigoroso_gap_vs_nogap.txt")
    
    # Transforma listas em dicionários indexados por cluster
    dict_gap = {d['Cluster']: d for d in gap_data}
    dict_nogap = {d['Cluster']: d for d in nogap_data}
    
    params = CONFIG["BATEIA_PARAMS"]["param_map"].keys()
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATORIO RIGOROSO DE COMPARACAO: GAP vs NO-GAP\n")
            f.write("Escopo: Modelos Sequenciais (LSTM, GRU, 1DCNN) em Clusters AM.\n")
            f.write("Metrica: Erro Percentual Absoluto em relacao a Literatura.\n")
            f.write("="*90 + "\n\n")
            
            for param in params:
                f.write(f"PARAMETRO: {param}\n")
                f.write(f"{'Cluster':<15} | {'GAP Error':<12} | {'No-GAP Error':<14} | {'Delta':<10} | {'Veredito'}\n")
                f.write("-" * 90 + "\n")
                
                for cluster in clusters_am:
                    # Dados GAP
                    gap_info = dict_gap.get(cluster, {}).get(param, {})
                    err_gap = gap_info.get('error', float('inf'))
                    mod_gap = gap_info.get('model', 'N/A')
                    
                    # Dados No-GAP
                    nogap_info = dict_nogap.get(cluster, {}).get(param, {})
                    err_nogap = nogap_info.get('error', float('inf'))
                    mod_nogap = nogap_info.get('model', 'N/A')
                    
                    if err_gap == float('inf') and err_nogap == float('inf'):
                        line = f"{cluster:<15} | {'N/A':<12} | {'N/A':<14} | {'--':<10} | Sem dados"
                    else:
                        delta = err_gap - err_nogap # Positivo significa que GAP tinha mais erro (No-GAP melhorou)
                        
                        str_gap = f"{err_gap:.2f}% ({mod_gap})" if err_gap != float('inf') else "N/A"
                        str_nogap = f"{err_nogap:.2f}% ({mod_nogap})" if err_nogap != float('inf') else "N/A"
                        
                        if err_nogap < err_gap:
                            veredito = ">> No-GAP VENCEU"
                        elif err_nogap > err_gap:
                            veredito = "GAP Venceu"
                        else:
                            veredito = "Empate"
                            
                        line = f"{cluster:<15} | {str_gap:<12} | {str_nogap:<14} | {delta:+.2f}%      | {veredito}"
                    
                    f.write(line + "\n")
                f.write("\n")
        print(f"Relatório de Comparação GAP vs No-GAP salvo em: {output_file}")
    except Exception as e:
        logging.error(f"Erro ao gerar relatório comparativo: {e}")

def generate_grouped_plot_gap_comparison(param: str, df_merged: pd.DataFrame, plot_config: Dict[str, Any], output_path: str):
    ylabel_map = {'Rc': 'Core radius ($r_c$)', 'Rt': 'Tidal radius ($r_t$)', 'f0': r'Central density ($\sigma_0$)', 'fbg': r'Background density ($\sigma_{bg}$)'}
    try:
        plt.style.use(plot_config["style"])
        fig, ax = plt.subplots(figsize=plot_config["figsize"])
        offset = 0.2
        
        # Legendas Fictícias para consistência
        ax.errorbar([], [], color='gray', marker='s', ls='None', markersize=plot_config['marker_size'], label='Angelo et. al.')
        ax.errorbar([], [], color='#1f77b4', marker='D', ls='None', markersize=plot_config['marker_size'], mfc='white', mec='#1f77b4', label='BATEIA-1 (GAP)')
        ax.plot([], [], color='#d62728', marker='D', ls='None', fillstyle='left', markersize=plot_config['marker_size'], mfc='#d62728', mec='#d62728', label='BATEIA-1 (No-GAP)')

        for j, (index, row) in enumerate(df_merged.iterrows()):
            lit_val, lit_err = row.get(f'{param}_lit'), row.get(f'err_{param}_lit')
            gap_val = row.get(f'{param}_gap')
            nogap_val = row.get(f'{param}_nogap')
            
            if pd.isna(lit_val): continue
            
            # Lit (Centro)
            ax.errorbar(j, lit_val, yerr=lit_err, ls='None', marker='s', color='gray', markersize=plot_config["marker_size"], capsize=plot_config["error_bar_capsize"])
            
            # GAP (Esquerda - Offset)
            if pd.notna(gap_val):
                err_est_gap = (lit_err / lit_val) * gap_val if lit_val > 0 else 0
                ax.errorbar(j - offset, gap_val, yerr=err_est_gap, fmt='none', ecolor='#1f77b4', capsize=plot_config["error_bar_capsize"], ls=':')
                ax.plot(j - offset, gap_val, marker='D', ls='None', mfc='white', mec='#1f77b4', markersize=plot_config["marker_size"])

            # No-GAP (Direita + Offset)
            if pd.notna(nogap_val):
                err_est_nogap = (lit_err / lit_val) * nogap_val if lit_val > 0 else 0
                ax.errorbar(j + offset, nogap_val, yerr=err_est_nogap, fmt='none', ecolor='#d62728', capsize=plot_config["error_bar_capsize"], ls='-.')
                # AQUI ESTÁ O AJUSTE VISUAL: fillstyle='left'
                ax.plot(j + offset, nogap_val, marker='D', ls='None', fillstyle='left', mfc='#d62728', mec='#d62728', markersize=plot_config["marker_size"])
        
        ax.set_ylabel(ylabel_map.get(param, param), fontsize=plot_config["label_fontsize"])
        ax.set_xticks(np.arange(len(df_merged)))
        ax.set_xticklabels(df_merged['Cluster'], rotation=plot_config["xtick_rotation"], ha='right', fontsize=plot_config["tick_fontsize"])
        fig.tight_layout()
        ax.legend(fontsize=plot_config["legend_fontsize"])
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f'comparacao_gap_vs_nogap_{param}.jpg'), dpi=plot_config["dpi"])
        plt.close(fig)
    except Exception:
        if 'fig' in locals(): plt.close(fig)

def gerar_graficos_perfil_comparacao(cluster, params_gap, params_nogap, params_lit, errors_lit, dados_obs):
    """
    Gera gráfico de perfil King comparando Lit vs GAP vs No-GAP, mantendo
    a estrutura e dimensões do gráfico original de AM (King Fit Compare).
    """
    if dados_obs is None or params_gap is None or params_nogap is None: return
    bin_centers, density_obs = dados_obs
    r_max = bin_centers.max() * 1.05
    r_cont = np.linspace(0, r_max, 500)
    
    cfg = CONFIG["PLOT_STYLE_PROFILE"]
    out_folder = os.path.join(CONFIG["PATHS"]["output_comparacao"], "Imagens", cluster)
    os.makedirs(out_folder, exist_ok=True)
    
    y_gap = funcao_king(r_cont, **params_gap)
    y_nogap = funcao_king(r_cont, **params_nogap)
    y_lit = funcao_king(r_cont, **params_lit)
    
    figsize = (1.4 * cfg["image_size_px"]/cfg["dpi"], cfg["image_size_px"]/cfg["dpi"])
    
    plt.figure(figsize=figsize, dpi=cfg["dpi"])
    plt.title(f'{cluster}: GAP vs No-GAP Comparison', fontsize=cfg["font_size"]-2)
    
    # 1. Observado (Círculos pretos, sem alpha para igualar o original)
    plt.plot(bin_centers, density_obs, 'ko', label='Observed', zorder=3)
    
    # 2. Literatura (Vermelho Sólido - Igual ao original)
    plt.plot(r_cont, y_lit, 'r-', label='Angelo et al. fit', zorder=4)
    
    # 3. GAP (Azul Tracejado - Igual ao original)
    plt.plot(r_cont, y_gap, 'b--', label='BATEIA-1 (GAP)', zorder=4)
    
    # 4. No-GAP (NOVO: Verde Traço-Ponto)
    plt.plot(r_cont, y_nogap, color='green', linestyle='-.', linewidth=2, label='BATEIA-1 (No-GAP)', zorder=5)
    
    plt.xlabel('Radial distance (r) [arcmin]', fontsize=cfg["font_size"])
    plt.ylabel('Surface density [stars / arcmin²]', fontsize=cfg["font_size"])
    
    # Legenda Externa (Crucial para manter dimensões idênticas ao original)
    plt.legend(fontsize=cfg["font_size"]-6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(os.path.join(out_folder, f"king_fit_comparison_{cluster}.{cfg['format']}"))
    plt.close()

# ==============================================================================
# ORQUESTRADOR PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    print("="*60 + "\nPIPELINE UNIFICADO: ANALISE BATEIA/BATEIA-1 (EXTRAÇÃO + PLOTAGEM)\n" + "="*60)
    
    csv_path = os.path.join(CONFIG["PATHS"]["base_aglomerados"], 'cluster_params.csv')
    df_literatura = load_literature_data(csv_path)
    
    if df_literatura is not None:
        
        print("\n[PASSO 1] EXTRAÇÃO DE DADOS E GERAÇÃO DE RESUMOS")
        
        # --- A. Workflow BM (Todos os clusters) ---
        path_bm_source = os.path.join(CONFIG["PATHS"]["base_redes_neurais_root"])
        output_path_bm = os.path.join(CONFIG["PATHS"]["output_base"], "BM_Groups")
        
        bateia_data_list_bm = []
        for cluster_name in df_literatura['Cluster']:
            if pd.isna(cluster_name): continue
            
            bateia_data = load_bateia_data_single_cluster(
                path_bm_source, cluster_name, 
                CONFIG["BATEIA_PARAMS"]["modelos_bm"], CONFIG["BATEIA_PARAMS"]["extratoras"], CONFIG["BATEIA_PARAMS"]["param_map"]
            )
            bateia_data_list_bm.append(bateia_data)
        
        # CORREÇÃO CIRÚRGICA: Injetar Literatura Correta (BM)
        correct_literature_in_results(bateia_data_list_bm, df_literatura, CONFIG["BATEIA_PARAMS"]["param_map"])

        save_summary_csv(bateia_data_list_bm, output_path_bm, "BM", list(CONFIG["BATEIA_PARAMS"]["param_map"].keys()))
        generate_generalization_report_BM(CONFIG["CLUSTERS_BM"], CONFIG["BATEIA_PARAMS"]["param_map"])
        
        # --- B. Workflow AM (Clusters Alvo - Descoberta Automática) ---
        print(">>> Analisando diretórios para identificar Clusters AM...")
        am_clusters_found = set()
        base_nn_path = CONFIG["PATHS"]["base_redes_neurais_root"]
        gap_root = os.path.join(base_nn_path, "GAP")
        if os.path.exists(gap_root):
            for root, dirs, files in os.walk(gap_root):
                if "AM" in root and "Previsto" in root:
                    for f in files:
                        if f.startswith("predicoes_reais_") and f.endswith("_gap.csv"):
                            parts = f.replace(".csv", "").split("_")
                            if len(parts) >= 7: am_clusters_found.add(parts[-2])
        
        CONFIG["CLUSTERS_AM"] = sorted(list(am_clusters_found))
        if CONFIG["CLUSTERS_AM"]:
            print(f"Clusters AM detectados: {', '.join(CONFIG['CLUSTERS_AM'])}")
        else:
            print("AVISO: Nenhum resultado AM detectado nos diretórios.")

        am_results_list = []
        for am_cluster in CONFIG["CLUSTERS_AM"]:
            am_data = load_bateia_data_single_cluster(
                CONFIG["PATHS"]["base_redes_neurais_root"], 
                am_cluster, 
                CONFIG["BATEIA_PARAMS"]["modelos_am"], 
                CONFIG["BATEIA_PARAMS"]["extratoras"], 
                CONFIG["BATEIA_PARAMS"]["param_map"]
            )
            am_results_list.append(am_data)

        # CORREÇÃO CIRÚRGICA: Injetar Literatura Correta (AM)
        correct_literature_in_results(am_results_list, df_literatura, CONFIG["BATEIA_PARAMS"]["param_map"])

        # Agora salva e plota com os valores corretos
        for am_data in am_results_list:
            am_cluster = am_data['Cluster']
            output_path = os.path.join(CONFIG["PATHS"]["output_base"], "AM_Single", am_cluster)
            save_summary_csv([am_data], output_path, "AM", list(CONFIG["BATEIA_PARAMS"]["param_map"].keys()))
            
            # Gráficos Single
            lit_data = df_literatura[df_literatura['Cluster'] == am_cluster]
            if not lit_data.empty:
                lit_row = lit_data.iloc[0]
                for param in CONFIG["BATEIA_PARAMS"]["param_map"]:
                    # Leitura segura Lit
                    val_lit = "0"
                    if param in lit_row: val_lit = lit_row[param]
                    elif param.lower() in lit_row: val_lit = lit_row[param.lower()]
                    try: l_val = float(str(val_lit).replace(',', '.'))
                    except: l_val = 0.0
                    
                    err_col = f"err_{param}"
                    if err_col not in lit_row: err_col = f"err_{param.lower()}"
                    try: l_err = float(str(lit_row[err_col]).replace(',', '.')) if err_col in lit_row else 0.0
                    except: l_err = 0.0

                    a_val = am_data.get(param, {}).get('value')
                    
                    if pd.notna(l_val) and pd.notna(a_val):
                        generate_single_cluster_plot(param, l_val, l_err, a_val, am_cluster, 
                                                     CONFIG["PLOT_STYLE_SCALAR"]["single"], output_path, "BATEIA-1 (AM)")
        
        generate_generalization_report_AM(CONFIG["CLUSTERS_AM"], CONFIG["BATEIA_PARAMS"]["param_map"])
        
# --- C. Plotagem Grouped (Para BM e AM) ---
        def run_grouped_plots(results_list, df_lit, output_path, label):
            print(f"\n--- Gerando Gráficos Grouped e Scatter {label} ---")
            plot_data = []
            for item in results_list:
                row = {'Cluster': item['Cluster']}
                for p in CONFIG["BATEIA_PARAMS"]["param_map"]:
                     if p in item: row[p] = item[p]['value']
                plot_data.append(row)
            
            df_lit_float = df_lit.copy()
            for col in df_lit_float.columns:
                if col != 'Cluster':
                     try:
                        df_lit_float[col] = df_lit_float[col].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else np.nan)
                     except: pass

            df_merged = pd.merge(
                df_lit_float.rename(columns=lambda c: f"{c}_lit" if c != 'Cluster' else c),
                pd.DataFrame(plot_data).rename(columns=lambda c: f"{c}_bateia" if c != 'Cluster' else c),
                on='Cluster'
            )
            
            # ==============================================================================
            # --- NOVO: GERAR SCATTER PLOTS 1:1 PARA TODOS OS CLUSTERS (Pedido do Revisor) ---
            print(f">>> Gerando Scatter Plots 1:1 (Dispersão) para {label}...")
            scatter_out_path = os.path.join(output_path, "Scatter_Comparisons")
            plotter = BateiaPlotter(CONFIG["PLOT_STYLE_SCALAR"]["grouped"])
            for param in CONFIG["BATEIA_PARAMS"]["param_map"]:
                plotter.generate_scatter_comparison(
                    param=param,
                    df_dados=df_merged,
                    output_path=scatter_out_path,
                    lit_label="Angelo et al.",
                    bateia_label=f"BATEIA-1 ({label})",
                    numbered=False
                )
                plotter.generate_scatter_comparison(
                    param=param,
                    df_dados=df_merged,
                    output_path=scatter_out_path,
                    lit_label="Angelo et al.",
                    bateia_label=f"BATEIA-1 ({label})",
                    numbered=True
                )
            # ==============================================================================

            clusters_all = sorted(df_merged['Cluster'].unique())
            for i in range(0, len(clusters_all), 5):
                grupo = clusters_all[i:i+5]
                df_grupo = df_merged[df_merged['Cluster'].isin(grupo)].copy()
                if df_grupo.empty: continue
                for param in CONFIG["BATEIA_PARAMS"]["param_map"]:
                    generate_grouped_plot(param, df_grupo, CONFIG["PLOT_STYLE_SCALAR"]["grouped"], output_path, (i//5)+1, f"BATEIA-1 ({label})")
        run_grouped_plots(bateia_data_list_bm, df_literatura, os.path.join(CONFIG["PATHS"]["output_base"], "BM_Groups"), "BM")
        run_grouped_plots(am_results_list, df_literatura, os.path.join(CONFIG["PATHS"]["output_base"], "AM_Groups"), "AM")

        print("\n[PASSO 2] GERAÇÃO DE PERFIS DETALHADOS (KING FIT)")
        
        for cluster in CONFIG["CLUSTERS_BM"]:
            p_bm = get_bm_params(cluster)
            p_lit, e_lit, p_hunt, d_obs = get_lit_obs_data(cluster)
            if p_bm and p_lit and d_obs:
                gerar_graficos_perfil(cluster, p_bm, p_lit, e_lit, p_hunt, d_obs, "BATEIA-1 (BM)")
                
        for cluster in CONFIG["CLUSTERS_AM"]:
            p_am = get_am_params(cluster)
            p_lit, e_lit, p_hunt, d_obs = get_lit_obs_data(cluster)
            if p_am and p_lit and d_obs:
                gerar_graficos_perfil(cluster, p_am, p_lit, e_lit, p_hunt, d_obs, "BATEIA-1 (AM)")
        
        # ==========================================================================
        # EXECUÇÃO DO MÓDULO 3 (NO-GAP vs GAP)
        # ==========================================================================
        print("\n" + "="*40 + "\n[PASSO 3 - NOVO] ANALISE RIGOROSA: GAP vs NO-GAP\n" + "="*40)
        
        # 1. Carregar dados No-GAP (Apenas Sequenciais, sufixo 'no-gap')
        nogap_results_list = []
        print(">>> Carregando dados No-GAP para Clusters AM...")
        for am_cluster in CONFIG["CLUSTERS_AM"]:
            nogap_data = load_bateia_data_single_cluster(
                CONFIG["PATHS"]["base_redes_neurais_root"],
                am_cluster,
                CONFIG["BATEIA_PARAMS"]["modelos_no_gap"], 
                CONFIG["BATEIA_PARAMS"]["extratoras"],
                CONFIG["BATEIA_PARAMS"]["param_map"]
            )
            nogap_results_list.append(nogap_data)
        
        # CORREÇÃO CIRÚRGICA: Injetar Literatura Correta (NO-GAP)
        correct_literature_in_results(nogap_results_list, df_literatura, CONFIG["BATEIA_PARAMS"]["param_map"])

        # 2. Salvar CSV de Resumo No-GAP
        save_summary_csv(nogap_results_list, CONFIG["PATHS"]["output_comparacao"], "NOGAP", list(CONFIG["BATEIA_PARAMS"]["param_map"].keys()))
        
        # 3. Gerar Relatório de Texto Comparativo
        generate_gap_comparison_report(CONFIG["CLUSTERS_AM"], am_results_list, nogap_results_list)
        
        # 4. Gerar Gráficos Agrupados Comparativos (Scalar)
        data_gap = []
        for item in am_results_list:
            row = {'Cluster': item['Cluster']}
            for p in CONFIG["BATEIA_PARAMS"]["param_map"]:
                 if p in item: row[f"{p}_gap"] = item[p]['value']
            data_gap.append(row)
            
        data_nogap = []
        for item in nogap_results_list:
            row = {'Cluster': item['Cluster']}
            for p in CONFIG["BATEIA_PARAMS"]["param_map"]:
                 if p in item: row[f"{p}_nogap"] = item[p]['value']
            data_nogap.append(row)
            
        df_lit_float = df_literatura.copy()
        for col in df_lit_float.columns:
            if col != 'Cluster':
                 try:
                    df_lit_float[col] = df_lit_float[col].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else np.nan)
                 except: pass

        df_lit_renamed = df_lit_float.rename(columns=lambda c: f"{c}_lit" if c != 'Cluster' else c)
        
        df_m1 = pd.merge(df_lit_renamed, pd.DataFrame(data_gap), on='Cluster')
        df_final_comp = pd.merge(df_m1, pd.DataFrame(data_nogap), on='Cluster')
        df_final_comp = df_final_comp[df_final_comp['Cluster'].isin(CONFIG["CLUSTERS_AM"])]
        
        print(">>> Gerando Gráficos Agrupados (Comparação Tripla)...")
        for param in CONFIG["BATEIA_PARAMS"]["param_map"]:
            generate_grouped_plot_gap_comparison(
                param, df_final_comp, CONFIG["PLOT_STYLE_SCALAR"]["grouped"], CONFIG["PATHS"]["output_comparacao"]
            )
            
        # 5. Gerar Gráficos de Perfil (King) Comparativos
        print(">>> Gerando Imagens de Perfil (Lit vs GAP vs No-GAP)...")
        for cluster in CONFIG["CLUSTERS_AM"]:
            p_gap = get_am_params(cluster) 
            p_nogap = get_nogap_params(cluster) 
            p_lit, e_lit, _, d_obs = get_lit_obs_data(cluster)
            
            if p_gap and p_nogap and p_lit and d_obs:
                gerar_graficos_perfil_comparacao(cluster, p_gap, p_nogap, p_lit, e_lit, d_obs)
                print(f" -> Imagem salva: {cluster}")

    else:
        logging.error("Não foi possível carregar a literatura. Abortando pipeline.")

    print("\nPIPELINE FINALIZADO COM SUCESSO.")

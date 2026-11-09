# bateia/real_clusters.py

# ==============================================================================
# BIBLIOTECAS E CONFIGURAÇÕES
# ==============================================================================


import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .config import (
    BASE_DIR, PARAMETROS_AJUSTE, FORMATO_IMAGEM, TAMANHO_FONTE,
    IMAGE_SIZE_PX, DPI
)
from matplotlib.lines import Line2D # Importado para criar legendas customizadas

# ==============================================================================
# SEÇÃO DE FUNÇÕES AUXILIARES (ATUALIZADA)
# ==============================================================================

def funcao_king(r, f0, Rc, Rt, fbg):
    """Modelo teórico de perfil de King."""
    if Rc == 0 or Rt <= Rc: return np.full_like(r, fbg)
    termo_r = np.sqrt(1 + (r / Rc)**2)
    termo_Rt = np.sqrt(1 + (Rt / Rc)**2)
    # Garante que o termo dentro do parêntese não seja negativo
    termo_interior = (1 / termo_r) - (1 / termo_Rt)
    termo_interior[termo_interior < 0] = 0
    densidade_perfil = f0 * (termo_interior)**2 + fbg
    densidade = np.where(r <= Rt, densidade_perfil, fbg)
    return densidade

def calcular_area_anular(r_interno, r_externo):
    """Calcula a área de um anel circular."""
    return np.pi * (r_externo**2 - r_interno**2)

def calcular_densidade_central(density):
    for valor in density:
        if valor > 0: return valor
    return 0

def calcular_raio_nucleo(bin_centers, density, f0):
    if f0 == 0: return 0
    alvo = 0.5 * f0
    indice_mais_proximo = (np.abs(density - alvo)).argmin()
    return bin_centers[indice_mais_proximo]

def calcular_rt_inicial(bin_centers, density, f0):
    """Calcula um Rt inicial baseado em 10% da densidade central (lógica antiga de Rt)."""
    if f0 == 0: return 0
    alvo = 0.10 * f0
    indices_validos = np.where(density <= alvo)[0]
    if indices_validos.size > 0:
        primeiro_indice = indices_validos[0]
        return bin_centers[primeiro_indice]
    else:
        return bin_centers[-1]

def calcular_raio_mare(bin_centers, density, fbg):
    """Calcula o Rt final, baseado no primeiro bin <= fbg (nova lógica)."""
    indices_validos = np.where(density <= fbg)[0]
    if indices_validos.size > 0:
        primeiro_indice = indices_validos[0]
        if primeiro_indice == 0 and len(bin_centers) > 1:
            return bin_centers[1]
        return bin_centers[primeiro_indice]
    else:
        return bin_centers[-1]

def calcular_densidade_fundo(bin_centers, density, Rt, r_max):
    """Calcula fbg usando um Rt (inicial) para definir a região de fundo."""
    intervalo_fundo = (bin_centers >= Rt) & (bin_centers <= r_max)
    valores_fundo = density[intervalo_fundo]
    if len(valores_fundo) == 0:
        indice_rt = (np.abs(bin_centers - Rt)).argmin()
        valores_fundo = [density[indice_rt]]
    return np.mean(valores_fundo) if len(valores_fundo) > 0 else 0

# ==============================================================================
# ETAPA UNIFICADA: PROCESSAMENTO DOS AGLOMERADOS E AJUSTE DE KING
# ==============================================================================
# ==============================================================================
# ETAPA UNIFICADA: PROCESSAMENTO DOS AGLOMERADOS E AJUSTE DE KING
# ==============================================================================

def _carregar_dados_cluster(cluster_name):
    """Fase 1: Configuração e Leitura de Dados."""
    coords_path = os.path.join(BASE_DIR, cluster_name, "Posicoes", f"{cluster_name}_coords_processed.csv")
    if not os.path.exists(coords_path):
        print(f"  WARNING: Coordinates file not found for {cluster_name}. Skipping.")
        return None

    ajustes = PARAMETROS_AJUSTE.get(cluster_name)
    if not ajustes:
        print(f"  WARNING: Adjustment parameters not found for {cluster_name}. Skipping.")
        return None

    data = pd.read_csv(coords_path, sep=';')
    r = np.sqrt((data['X'] - 0)**2 + (data['Y'] - 0)**2) * 60  # Em arcmin
    r_sorted = np.sort(r)
    r_max = (r_sorted.max() * np.sqrt(2)) / 2
    
    return {
        'r_sorted': r_sorted,
        'r_max': r_max,
        'x_coords_arcmin': data['X'] * 60,
        'y_coords_arcmin': data['Y'] * 60,
        'ajustes': ajustes
    }

def _calcular_pipeline_divisoes(cluster_name, cluster_data):
    """Fase 2: Cálculo (Loop por todas as divisões)."""
    r_sorted = cluster_data['r_sorted']
    r_max = cluster_data['r_max']
    ajustes = cluster_data['ajustes']
    x_coords_arcmin = cluster_data['x_coords_arcmin']
    y_coords_arcmin = cluster_data['y_coords_arcmin']
    
    fator_central = ajustes.get('AjusteCentral', 1.0)
    fator_restante = ajustes.get('AjusteRestante', 1.0)
    fator_fundo = ajustes.get('AjusteFundoBG', 1.0)
    t_limit = ajustes.get('t', 41)
    
    resultados_cluster = []
    print(f"  Calculating parameters and residuals for {t_limit - 5} divisions...")

    for i, num_bins in enumerate(range(5, t_limit), start=1):
        espessura_bin = r_max / num_bins
        bin_edges = np.arange(0, r_max + espessura_bin, espessura_bin)
        if len(bin_edges) > num_bins + 1: bin_edges = bin_edges[:num_bins + 1]
        
        star_counts, _ = np.histogram(r_sorted, bins=bin_edges)
        if np.sum(star_counts > 0) < 4: continue
            
        bin_areas = np.array([calcular_area_anular(bin_edges[j], bin_edges[j+1]) for j in range(len(bin_edges) - 1)])
        density = np.divide(star_counts, bin_areas, out=np.zeros_like(star_counts, dtype=float), where=bin_areas != 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        f0 = calcular_densidade_central(density)
        if f0 == 0: continue
        
        Rc = calcular_raio_nucleo(bin_centers, density, f0)
        Rt_inicial = calcular_rt_inicial(bin_centers, density, f0)
        fbg = calcular_densidade_fundo(bin_centers, density, Rt_inicial, r_max)
        Rt = calcular_raio_mare(bin_centers, density, fbg)

        if Rt <= Rc: continue
        
        f0_adj = f0 * fator_central
        Rc_adj = Rc * fator_restante
        Rt_adj = Rt * fator_restante
        fbg_adj = fbg * fator_fundo
        
        densidade_ajustada = funcao_king(bin_centers, f0_adj, Rc_adj, Rt_adj, fbg_adj)
        residuos = np.abs(density - densidade_ajustada)
        
        resultados_cluster.append({
            'Topologia_ID': i, 'Divisoes': num_bins, 'Espessura (arcmin)': espessura_bin,
            'f0 (stars/arcmin²)': f0, 'Rc (arcmin)': Rc, 'Rt (arcmin)': Rt,
            'fbg (stars/arcmin²)': fbg, 'Mean Residual': np.mean(residuos)
        })
        
        # Salva Densidade Radial
        output_folder_drad = os.path.join(BASE_DIR, cluster_name, "Resultados", "Densidade radial")
        os.makedirs(output_folder_drad, exist_ok=True)
        pd.DataFrame({
            'Bin_Center (arcmin)': bin_centers, 
            'Density (stars/arcmin²)': density, 
            'Espessura (arcmin)': np.full_like(bin_centers, espessura_bin)
        }).to_csv(os.path.join(output_folder_drad, f"drad_{cluster_name}_{i}.csv"), index=False)
        
        # Gera Gráficos Individuais de cada divisão
        _gerar_graficos_divisao(cluster_name, num_bins, i, bin_centers, bin_edges, density, x_coords_arcmin, y_coords_arcmin)
        
    return resultados_cluster

def _gerar_graficos_divisao(cluster_name, num_bins, topologia_id, bin_centers, bin_edges, density, x_coords, y_coords):
    """Gera gráficos de RDP e distribuição espacial para uma divisão específica."""
    tamanho_figura_polegadas = (IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI)
    output_folder_graficos = os.path.join(BASE_DIR, cluster_name, "Resultados", "Graficos_Densidade")
    os.makedirs(output_folder_graficos, exist_ok=True)

    # 1. RDP Plot
    plt.figure(figsize=tamanho_figura_polegadas, dpi=DPI)
    plt.plot(bin_centers, density, marker='o', linestyle=' ', color='black')
    plt.title(f'{cluster_name}: RDP (Bins = {num_bins}) ', fontsize=TAMANHO_FONTE - 1)
    plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
    plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder_graficos, f"densidade_{cluster_name}_div_{num_bins}.{FORMATO_IMAGEM}"), bbox_inches='tight')
    plt.close()

    # 2. Spatial Distribution Plot
    plt.figure(figsize=tamanho_figura_polegadas, dpi=DPI)
    plt.scatter(x_coords, y_coords, color='black', s=0.8, alpha=0.8)
    theta = np.linspace(0, 2 * np.pi, 200)
    for radius in bin_edges:
        if radius > 0:
            x_circle, y_circle = radius * np.cos(theta), radius * np.sin(theta)
            plt.plot(x_circle, y_circle, linestyle='--', color='blue', linewidth=0.8)
    plot_limit = bin_edges[-1]
    plt.xlim(-plot_limit, plot_limit)
    plt.ylim(-plot_limit, plot_limit)
    plt.title(f'{cluster_name}: Bins = {num_bins}', fontsize=TAMANHO_FONTE - 1)
    plt.xlabel(r'$\Delta \alpha * \cos(\delta)$ [arcmin]', fontsize=TAMANHO_FONTE)
    plt.ylabel(r'$\Delta \delta$ [arcmin]', fontsize=TAMANHO_FONTE)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder_graficos, f"distribuicao_espacial_{cluster_name}_div_{num_bins}.{FORMATO_IMAGEM}"), bbox_inches='tight')
    plt.close()

def _analisar_e_salvar_resultados(cluster_name, df_principal, ajustes):
    """Fase 3: Análise e Salvamento de Dados."""
    print(f"  Analyzing {len(df_principal)} valid divisions and saving results...")
    
    fator_central = ajustes.get('AjusteCentral', 1.0)
    fator_restante = ajustes.get('AjusteRestante', 1.0)
    fator_fundo = ajustes.get('AjusteFundoBG', 1.0)

    output_folder_res = os.path.join(BASE_DIR, cluster_name, "Resultados")
    os.makedirs(output_folder_res, exist_ok=True)
    
    # Salva consolidado
    df_principal_para_salvar = df_principal.drop(columns=['Mean Residual'])
    df_principal_para_salvar.to_csv(os.path.join(output_folder_res, f"{cluster_name}_parametros_king_consolidados.csv"), index=False)
    
    # Análise de resíduos
    output_dir_analise_residuos = os.path.join(output_folder_res, "Ajuste de King", "Residuos")
    os.makedirs(output_dir_analise_residuos, exist_ok=True)
    
    resultados_top10 = df_principal.sort_values(by='Mean Residual').head(10)
    resultados_top10[['Divisoes', 'Mean Residual']].to_csv(os.path.join(output_dir_analise_residuos, f"Residuos_{cluster_name}.csv"), index=False)
    
    df_filtered = df_principal[df_principal['Divisoes'] >= 8]
    if not df_filtered.empty:
        best_row = df_filtered.loc[df_filtered['Mean Residual'].idxmin()]
        best_division_info = {'Division': int(best_row['Divisoes']), 'Mean Residual': best_row['Mean Residual']}
        pd.DataFrame([best_division_info]).to_csv(os.path.join(output_dir_analise_residuos, f"Menor_residuo_{cluster_name}.csv"), index=False)
        
        # Salva Target Parameters
        output_dir_target = os.path.join(BASE_DIR, cluster_name, "Target")
        os.makedirs(output_dir_target, exist_ok=True)
        
        final_params = {
            'f0': best_row['f0 (stars/arcmin²)'] * fator_central,
            'Rc': best_row['Rc (arcmin)'] * fator_restante,
            'Rt': best_row['Rt (arcmin)'] * fator_restante,
            'fbg': best_row['fbg (stars/arcmin²)'] * fator_fundo
        }
        for name, val in final_params.items():
            pd.DataFrame({name: [val]}).to_csv(os.path.join(output_dir_target, f"{name}_otm_{cluster_name}.csv"), index=False)
            
        pd.DataFrame({
            'Best Division': [best_division_info['Division']], 
            'Thickness (arcmin)': [best_row['Espessura (arcmin)']], 
            'Mean Residual': [best_division_info['Mean Residual']]
        }).to_csv(os.path.join(output_dir_target, f"best_division_info_{cluster_name}.csv"), index=False)
        
        return best_row, best_division_info
    
    return None, None

def _gerar_graficos_finais(cluster_name, df_principal, best_division_info, ajustes, r_max):
    """Fase 4 e 5: Geração de Gráficos Finais e Cópias Target."""
    print("  Generating final analysis graphs...")
    
    output_folder_res = os.path.join(BASE_DIR, cluster_name, "Resultados")
    output_folder_analise = os.path.join(output_folder_res, "Graficos_Analise_Final")
    os.makedirs(output_folder_analise, exist_ok=True)

    fator_central = ajustes.get('AjusteCentral', 1.0)
    fator_restante = ajustes.get('AjusteRestante', 1.0)
    fator_fundo = ajustes.get('AjusteFundoBG', 1.0)
    inct_f0 = ajustes.get('inct_f0', 1.0)
    inct_rc = ajustes.get('inct_rc', 1.0)
    inct_rt = ajustes.get('inct_rt', 1.0)
    inct_fbg = ajustes.get('inct_fbg', 1.0)

    # 1. Residuals Bar Plot
    plt.figure(figsize=(512 / DPI, 512 / DPI), dpi=DPI)
    cores_barras = ['gray'] * len(df_principal)
    if best_division_info:
        try:
            min_index_pos = df_principal.index[df_principal['Divisoes'] == best_division_info['Division']].tolist()[0]
            cores_barras[min_index_pos] = 'gold'
        except (IndexError, KeyError):
            pass
    plt.bar(df_principal['Divisoes'], df_principal['Mean Residual'], color=cores_barras)
    plt.title(f'{cluster_name}: Residuals and divisions', fontsize=TAMANHO_FONTE - 1)
    plt.xlabel('Number of divisions', fontsize=TAMANHO_FONTE)
    plt.ylabel('Mean residual [stars / arcmin²]', fontsize=TAMANHO_FONTE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder_analise, f"analise_residuos_{cluster_name}.{FORMATO_IMAGEM}"), bbox_inches='tight')
    plt.close()

    # 2. Individual Fit Plots for each division (already done in loop, but here we can add consolidated logic if needed)
    # Actually the original code does more plots here in Fase 4.
    r_continuo = np.linspace(0, r_max, 500)
    for index, row in df_principal.iterrows():
        num_divisions = int(row['Divisoes'])
        drad_path = os.path.join(output_folder_res, "Densidade radial", f"drad_{cluster_name}_{int(row['Topologia_ID'])}.csv")
        if not os.path.exists(drad_path): continue
        
        densidade_observada_df = pd.read_csv(drad_path)
        bin_centers = densidade_observada_df['Bin_Center (arcmin)'].values
        density_observada = densidade_observada_df['Density (stars/arcmin²)'].values
        tamanho_fig_individual = (IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI)

        # Parameters Plot
        plt.figure(figsize=tamanho_fig_individual, dpi=DPI)
        plt.title(f'{cluster_name}: Parameters (Bins = {num_divisions})', fontsize=TAMANHO_FONTE - 1)
        plt.plot(bin_centers, density_observada, marker='o', color='black', alpha=1, linestyle='None', label='Observed Density')
        plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        
        rc_val, rt_val, fbg_val = row['Rc (arcmin)'], row['Rt (arcmin)'], row['fbg (stars/arcmin²)']
        plt.axvline(rc_val, color='green', linestyle='-', label="Core radius [arcmin]")
        plt.axvline(rc_val / inct_rc, color='green', linestyle=':', alpha=0.7)
        plt.axvline(rc_val * inct_rc, color='green', linestyle=':', alpha=0.7)
        plt.axvline(rt_val, color='red', linestyle='-', label="Tidal radius [arcmin]")
        plt.axvline(rt_val / inct_rt, color='red', linestyle=':', alpha=0.7)
        plt.axvline(rt_val * inct_rt, color='red', linestyle=':', alpha=0.7)
        plt.axhline(fbg_val, color='orange', linestyle='-', label="Background density [stars/arcmin²]")
        plt.axhline(fbg_val / inct_fbg, color='orange', linestyle=':', alpha=0.7)
        plt.axhline(fbg_val * inct_fbg, color='orange', linestyle=':', alpha=0.7)
        plt.legend(fontsize=TAMANHO_FONTE - 8)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_folder_params = os.path.join(output_folder_res, "Graficos_Parametros")
        os.makedirs(output_folder_params, exist_ok=True)
        plt.savefig(os.path.join(output_folder_params, f"parameters_{cluster_name}_div_{num_divisions}.{FORMATO_IMAGEM}"), bbox_inches='tight')
        plt.close()

        # King Fit Plot
        f0_adj = row['f0 (stars/arcmin²)'] * fator_central
        Rc_adj = row['Rc (arcmin)'] * fator_restante
        Rt_adj = row['Rt (arcmin)'] * fator_restante
        fbg_adj = row['fbg (stars/arcmin²)'] * fator_fundo
        
        densidade_king_continua = funcao_king(r_continuo, f0_adj, Rc_adj, Rt_adj, fbg_adj)
        densidade_king_lower = funcao_king(r_continuo, f0_adj / inct_f0, Rc_adj / inct_rc, Rt_adj / inct_rt, fbg_adj / inct_fbg)
        densidade_king_upper = funcao_king(r_continuo, f0_adj * inct_f0, Rc_adj * inct_rc, Rt_adj * inct_rt, fbg_adj * inct_fbg)
        
        densidade_king_nos_pontos = funcao_king(bin_centers, f0_adj, Rc_adj, Rt_adj, fbg_adj)
        residuos = densidade_king_nos_pontos - density_observada
        error_bars = [np.maximum(0, -residuos), np.maximum(0, residuos)]

        plt.figure(figsize=tamanho_fig_individual, dpi=DPI)
        plt.title(f'{cluster_name}: King\'s fit (Bins = {num_divisions} )', fontsize=TAMANHO_FONTE - 1)
        plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        plt.plot(bin_centers, density_observada, marker='o', color='black', alpha=1, linestyle='None', label='Observed Density', zorder=3)
        plt.plot(r_continuo, densidade_king_continua, linestyle='-', color='magenta', label='King Model Fit', zorder=4)
        plt.fill_between(r_continuo, densidade_king_lower, densidade_king_upper, color='gray', alpha=0.3, label='Uncertainty', zorder=1)
        plt.errorbar(bin_centers, density_observada, yerr=error_bars, fmt='none', ecolor='black', elinewidth=1.5, capsize=0.5, label='Residual', zorder=2)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_folder_fit = os.path.join(output_folder_res, "Graficos_King_Fit")
        os.makedirs(output_folder_fit, exist_ok=True)
        plt.savefig(os.path.join(output_folder_fit, f"king_fit_{cluster_name}_div_{num_divisions}.{FORMATO_IMAGEM}"), bbox_inches='tight')
        plt.close()

    # Fase 5: Salvar cópias dos melhores gráficos
    if best_division_info:
        _copiar_melhores_graficos(cluster_name, best_division_info['Division'], output_folder_res)

def _copiar_melhores_graficos(cluster_name, best_div_num, output_folder_res):
    """Copia os gráficos da melhor divisão para a pasta Target."""
    print("  Saving copies of the best graphs to the Target folder...")
    output_dir_target = os.path.join(BASE_DIR, cluster_name, "Target")
    os.makedirs(output_dir_target, exist_ok=True)
    
    graficos_a_copiar = {
        "Graficos_Densidade/densidade": "density_profile_best",
        "Graficos_Densidade/distribuicao_espacial": "spatial_dist_best",
        "Graficos_Parametros/parameters": "parameters_best",
        "Graficos_King_Fit/king_fit": "king_fit_best"
    }
    for caminho_parcial, nome_final in graficos_a_copiar.items():
        nome_arquivo_origem = f"{os.path.basename(caminho_parcial)}_{cluster_name}_div_{best_div_num}.{FORMATO_IMAGEM}"
        caminho_origem = os.path.join(output_folder_res, os.path.dirname(caminho_parcial), nome_arquivo_origem)
        nome_arquivo_destino = f"{cluster_name}_{nome_final}.{FORMATO_IMAGEM}"
        caminho_destino = os.path.join(output_dir_target, nome_arquivo_destino)
        try:
            shutil.copy2(caminho_origem, caminho_destino)
            print(f"    - Copied: {nome_arquivo_destino}")
        except FileNotFoundError:
            print(f"    - WARNING: Could not find graph to copy: {caminho_origem}")

def _gerar_comparacao_literatura(cluster_name, df_principal, best_division_info, ajustes, r_max):
    """Fase 6 e 7: Geração de Gráficos de Comparação com a Literatura."""
    if not best_division_info:
        return

    print("  Generating comparison graphs with literature...")
    output_folder_res = os.path.join(BASE_DIR, cluster_name, "Resultados")
    output_dir_target = os.path.join(BASE_DIR, cluster_name, "Target")
    output_dir_comparacao = os.path.join(output_folder_res, "Graficos_Comparacao_Literatura")
    os.makedirs(output_dir_comparacao, exist_ok=True)
    
    # Parâmetros finais calculados
    fator_central = ajustes.get('AjusteCentral', 1.0)
    fator_restante = ajustes.get('AjusteRestante', 1.0)
    fator_fundo = ajustes.get('AjusteFundoBG', 1.0)
    
    best_row = df_principal[df_principal['Divisoes'] == best_division_info['Division']].iloc[0]
    f0_final = best_row['f0 (stars/arcmin²)'] * fator_central
    Rc_final = best_row['Rc (arcmin)'] * fator_restante
    Rt_final = best_row['Rt (arcmin)'] * fator_restante
    fbg_final = best_row['fbg (stars/arcmin²)'] * fator_fundo

    params_angelo = {}
    params_angelo_err = {}
    can_compare = True
    param_names = ['f0', 'Rc', 'Rt', 'fbg']
    
    for p_name in param_names:
        lit_path = os.path.join(output_dir_target, f"mat_{p_name}_otm_{cluster_name}.csv")
        err_path = os.path.join(output_dir_target, f"mat_err_{p_name}_otm_{cluster_name}.csv")
        try:
            params_angelo[p_name] = pd.read_csv(lit_path)[p_name].iloc[0]
            if os.path.exists(err_path):
                params_angelo_err[p_name] = pd.read_csv(err_path, skiprows=1, header=None).iloc[0, 0]
            else:
                params_angelo_err[p_name] = 0
        except (FileNotFoundError, KeyError, IndexError) as e:
            print(f"  WARNING: Literature file not found or invalid for '{p_name}'. Skipping comparison. Error: {e}")
            can_compare = False
            break
    
    if can_compare:
        r_continuo = np.linspace(0, r_max, 500)
        best_topo_id = best_row['Topologia_ID']
        drad_path = os.path.join(output_folder_res, "Densidade radial", f"drad_{cluster_name}_{int(best_topo_id)}.csv")
        best_density_df = pd.read_csv(drad_path)
        bin_centers = best_density_df['Bin_Center (arcmin)'].values
        density_observada = best_density_df['Density (stars/arcmin²)'].values

        # King Fit Comparison
        curva_king_bateia = funcao_king(r_continuo, f0_final, Rc_final, Rt_final, fbg_final)
        curva_king_angelo = funcao_king(r_continuo, params_angelo['f0'], params_angelo['Rc'], params_angelo['Rt'], params_angelo['fbg'])
        
        plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        plt.title(f'{cluster_name}: King Fit\n(Preliminary estimates and Angelo et al.)', fontsize=TAMANHO_FONTE - 4)
        plt.plot(bin_centers, density_observada, marker='o', color='black', linestyle='None', label='Observed density')
        plt.plot(r_continuo, curva_king_bateia, color='blue', linestyle='--', label='Preliminary estimates')
        plt.plot(r_continuo, curva_king_angelo, color='red', linestyle='-', label='Angelo et al. Fit')
        plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_comparacao, f"comparacao_king_fit_{cluster_name}.{FORMATO_IMAGEM}"), bbox_inches='tight')
        plt.close()

        # Parameters Comparison
        fig, ax = plt.subplots(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        ax.set_title(f'{cluster_name}: Parameters\n(Preliminary estimates and Angelo et al.)', fontsize=TAMANHO_FONTE - 4)
        ax.plot(bin_centers, density_observada, marker='o', color='black', linestyle='None', label='Observed density')
        ax.axvline(Rc_final, color='blue', linestyle='--')
        ax.axvline(Rt_final, color='blue', linestyle='--')
        ax.axhline(fbg_final, color='blue', linestyle='--')
        ax.axvline(params_angelo['Rc'], color='red', linestyle='-')
        ax.axvline(params_angelo['Rt'], color='red', linestyle='-')
        ax.axhline(params_angelo['fbg'], color='red', linestyle='-')
        ax.axvspan(min(Rc_final, params_angelo['Rc']), max(Rc_final, params_angelo['Rc']), color='gray', alpha=0.2)
        ax.axvspan(min(Rt_final, params_angelo['Rt']), max(Rt_final, params_angelo['Rt']), color='gray', alpha=0.2)
        ax.axhspan(min(fbg_final, params_angelo['fbg']), max(fbg_final, params_angelo['fbg']), color='gray', alpha=0.2)
        ax.set_xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        ax.set_ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        ax.grid(True, linestyle=':', alpha=0.7)
        legend_elements = [
            Line2D([0], [0], color='black', marker='o', linestyle='None', label='Observed density'),
            Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Preliminary estimates parameters'),
            Line2D([0], [0], color='red', lw=2, linestyle='-', label='Angelo et al. Parameters'),
            Line2D([0], [0], color='gray', alpha=0.2, lw=4, label='Difference')
        ]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_comparacao, f"comparacao_parametros_{cluster_name}.{FORMATO_IMAGEM}"), bbox_inches='tight')
        plt.close()

        # King uncertainty
        f0_lit_err, Rc_lit_err, Rt_lit_err, fbg_lit_err = params_angelo_err['f0'], params_angelo_err['Rc'], params_angelo_err['Rt'], params_angelo_err['fbg']
        curva_king_lit_lower = funcao_king(r_continuo, params_angelo['f0'] - f0_lit_err, params_angelo['Rc'] - Rc_lit_err, params_angelo['Rt'] - Rt_lit_err, params_angelo['fbg'] - fbg_lit_err)
        curva_king_lit_upper = funcao_king(r_continuo, params_angelo['f0'] + f0_lit_err, params_angelo['Rc'] + Rc_lit_err, params_angelo['Rt'] + Rt_lit_err, params_angelo['fbg'] + fbg_lit_err)
        
        plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        plt.title(f'{cluster_name}: Preliminary estimates\nfit and literature uncertainty', fontsize=TAMANHO_FONTE - 4)
        plt.plot(bin_centers, density_observada, marker='o', color='black', linestyle='None', label='Observed density')
        plt.fill_between(r_continuo, curva_king_lit_lower, curva_king_lit_upper, color='gray', alpha=0.4, label='Angelo et al. Uncertainty')
        plt.plot(r_continuo, curva_king_bateia, color='blue', linestyle='-', label='Preliminary estimates fit')
        plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_comparacao, f"comparacao_king_incerteza_lit_{cluster_name}.{FORMATO_IMAGEM}"), bbox_inches='tight')
        plt.close()

        # Parameters with uncertainty
        fig, ax = plt.subplots(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        ax.set_title(f'{cluster_name}: Preliminary estimates\nparameters and literature uncertainty', fontsize=TAMANHO_FONTE - 4)
        ax.plot(bin_centers, density_observada, marker='o', color='black', linestyle='None', label='Observed density')
        ax.axvspan(params_angelo['Rc'] - Rc_lit_err, params_angelo['Rc'] + Rc_lit_err, color='gray', alpha=0.3)
        ax.axvspan(params_angelo['Rt'] - Rt_lit_err, params_angelo['Rt'] + Rt_lit_err, color='gray', alpha=0.3)
        ax.axhspan(params_angelo['fbg'] - fbg_lit_err, params_angelo['fbg'] + fbg_lit_err, color='gray', alpha=0.3)
        ax.axvline(Rc_final, color='blue', linestyle='-')
        ax.axvline(Rt_final, color='blue', linestyle='-')
        ax.axhline(fbg_final, color='blue', linestyle='-')
        ax.set_xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        ax.set_ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        ax.grid(True, linestyle=':', alpha=0.7)
        legend_elements = [
            Line2D([0], [0], color='black', marker='o', linestyle='None', label='Observed density'),
            Line2D([0], [0], color='blue', lw=2, label='Preliminary estimates parameters'),
            Line2D([0], [0], color='gray', alpha=0.3, lw=4, label='Angelo et al. Uncertainty')
        ]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_comparacao, f"comparacao_parametros_incerteza_lit_{cluster_name}.{FORMATO_IMAGEM}"), bbox_inches='tight')
        plt.close()

        print(f"  Comparison graphs for {cluster_name} saved successfully.")

def processar_aglomerados_reais(clusters_a_processar):
    """
    Orquestrador público: Executa todo o pipeline de análise para os aglomerados reais.
    """
    print("\n" + "=" * 60)
    print("Initializing unified processing for real clusters...")
    print("=" * 60)

    for cluster_name in clusters_a_processar:
        print(f"\n--- Processing Cluster: {cluster_name} ---")

        # 1. Carregamento
        cluster_data = _carregar_dados_cluster(cluster_name)
        if not cluster_data: continue

        # 2. Pipeline de Divisões (Cálculos e Gráficos de Divisão)
        resultados_cluster = _calcular_pipeline_divisoes(cluster_name, cluster_data)
        if not resultados_cluster:
            print("  No valid divisions found. Finishing process for this cluster.")
            continue

        df_principal = pd.DataFrame(resultados_cluster)

        # 3. Análise e Salvamento (Encontra melhor divisão e salva targets)
        best_row, best_division_info = _analisar_e_salvar_resultados(cluster_name, df_principal, cluster_data['ajustes'])

        # 4. Gráficos Finais (Resíduos, Parâmetros, Fit e Cópias Target)
        _gerar_graficos_finais(cluster_name, df_principal, best_division_info, cluster_data['ajustes'], cluster_data['r_max'])

        # 5. Comparação com Literatura
        _gerar_comparacao_literatura(cluster_name, df_principal, best_division_info, cluster_data['ajustes'], cluster_data['r_max'])

        print(f"  Processing for {cluster_name} completed successfully.")

    print("\n" + "=" * 60)
    print("Unified processing for all real clusters is complete.")
    print("=" * 60)

# ==============================================================================
# PONTO DE ENTRADA DO SCRIPT (PAINEL DE CONTROLE)
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("     BATEIA APPROACH (VERSION 1.0) - PIPELINE EXECUTION")
    print("=" * 60)
    # Exemplo de uso direto se executado como script
    # processar_aglomerados_reais(["M67", "NGC188"]) 
    print("\n" + "=" * 60)
    print("     BATEIA APPROACH (VERSION 1.0) - PIPELINE FINISH")
    print("=" * 60)


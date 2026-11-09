# bateia/synthetic_analysis.py

# Importa as configurações e funções de cálculo do projeto BATEIA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import (
    BASE_DIR, PARAMETROS_AJUSTE, T_MAX_SINTETICOS, IMAGE_SIZE_PX, DPI,
    FORMATO_IMAGEM, TAMANHO_FONTE
)
from .real_clusters import (
    calcular_area_anular,
    calcular_densidade_central,
    calcular_raio_nucleo,
    calcular_rt_inicial,
    calcular_densidade_fundo,
    calcular_raio_mare,
    funcao_king
)

def processar_analise_sintetica(clusters_a_processar):
    """
    Executa o pipeline de análise para os aglomerados SINTÉTICOS,
    calculando os parâmetros de King para uma topologia fixa (t_ideal + 4)
    e aplicando os fatores de ajuste.
    """
    print("\n" + "="*60)
    print("Initializing processing for SYNTHETIC clusters...")
    print("="*60)

    for cluster_name in clusters_a_processar:
        print(f"\n--- Processing Synthetic Set for Cluster: {cluster_name} ---")

        # --- FASE 1: CONFIGURAÇÃO E DEFINIÇÃO DA TOPOLOGIA ---
        ajustes = PARAMETROS_AJUSTE.get(cluster_name)
        if not ajustes or 't_ideal' not in ajustes:
            print(f"  WARNING: 't_ideal' not found for {cluster_name} in config. Skipping.")
            continue
        
        num_bins = ajustes['t_ideal'] + 4
        print(f"  Using fixed topology: {num_bins} divisions (based on t_ideal={ajustes['t_ideal']})")

        fator_central = ajustes.get('AjusteCentral', 1.0)
        fator_restante = ajustes.get('AjusteRestante', 1.0)
        fator_fundo = ajustes.get('AjusteFundoBG', 1.0)
        print(f"  Adjustment factors loaded: Central={fator_central}, Rest={fator_restante}, BG={fator_fundo}")

        # --- INÍCIO DA ALTERAÇÃO 1/3: Carrega os fatores de incerteza ---
        inct_f0 = ajustes.get('inct_f0', 1.0)
        inct_rc = ajustes.get('inct_rc', 1.0)
        inct_rt = ajustes.get('inct_rt', 1.0)
        inct_fbg = ajustes.get('inct_fbg', 1.0)

        # --- FASE 1B: CALCULAR PERFIL REAL PARA COMPARAÇÃO ---
        real_coords_path = os.path.join(BASE_DIR, cluster_name, "Posicoes", f"{cluster_name}_coords_processed.csv")
        try:
            real_data = pd.read_csv(real_coords_path, sep=';', header=0)
            real_r = np.sqrt(real_data['X']**2 + real_data['Y']**2) * 60
            real_r_sorted = np.sort(real_r)
            real_r_max = (real_r_sorted.max() * np.sqrt(2)) / 2
            
            real_espessura_bin = real_r_max / num_bins
            real_bin_edges = np.arange(0, real_r_max + real_espessura_bin, real_espessura_bin)
            if len(real_bin_edges) > num_bins + 1: real_bin_edges = real_bin_edges[:num_bins + 1]

            real_star_counts, _ = np.histogram(real_r_sorted, bins=real_bin_edges)
            real_bin_areas = np.array([calcular_area_anular(real_bin_edges[j], real_bin_edges[j+1]) for j in range(len(real_bin_edges) - 1)])
            real_density = np.divide(real_star_counts, real_bin_areas, out=np.zeros_like(real_star_counts, dtype=float), where=real_bin_areas!=0)
            real_bin_centers = (real_bin_edges[:-1] + real_bin_edges[1:]) / 2

            real_f0 = calcular_densidade_central(real_density)
            real_Rc = calcular_raio_nucleo(real_bin_centers, real_density, real_f0)
            real_Rt_inicial = calcular_rt_inicial(real_bin_centers, real_density, real_f0)
            real_fbg = calcular_densidade_fundo(real_bin_centers, real_density, real_Rt_inicial, real_r_max)
            real_Rt = calcular_raio_mare(real_bin_centers, real_density, real_fbg)
            
            real_f0_adj = real_f0 * fator_central
            real_Rc_adj = real_Rc * fator_restante
            real_Rt_adj = real_Rt * fator_restante
            real_fbg_adj = real_fbg * fator_fundo

        except (FileNotFoundError, Exception) as e:
            print(f"  WARNING: Could not process real cluster profile for comparison. Skipping graph overlay. Error: {e}")
            real_density, real_bin_centers, real_r_max = None, None, None
            real_f0_adj, real_Rc_adj, real_Rt_adj, real_fbg_adj = 0,0,0,0

        resultados_sinteticos = []
        dados_de_densidade_sintetica = []
        dados_da_curva_de_king_sintetica = []

        # --- FASE 2: CÁLCULO (Loop por todos os aglomerados sintéticos) ---
        for i in tqdm(range(1, T_MAX_SINTETICOS + 1), desc=f"  Analyzing {cluster_name} synthetics"):
            coords_path = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "CSV", f"{cluster_name}_Sintetico_{i}.csv")
            if not os.path.exists(coords_path): continue
            
            data = pd.read_csv(coords_path, sep=';')
            if data.empty: continue

            r = np.sqrt(data['X']**2 + data['Y']**2)
            r_sorted = np.sort(r)
            r_max = (r_sorted.max() * np.sqrt(2)) / 2
            
            espessura_bin = r_max / num_bins
            bin_edges = np.arange(0, r_max + espessura_bin, espessura_bin)
            if len(bin_edges) > num_bins + 1: bin_edges = bin_edges[:num_bins + 1]
            
            star_counts, _ = np.histogram(r_sorted, bins=bin_edges)
            if np.sum(star_counts > 0) < 4: continue
                
            bin_areas = np.array([calcular_area_anular(bin_edges[j], bin_edges[j+1]) for j in range(len(bin_edges) - 1)])
            density = np.divide(star_counts, bin_areas, out=np.zeros_like(star_counts, dtype=float), where=bin_areas!=0)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            dados_de_densidade_sintetica.append({'bins': bin_centers, 'density': density})

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

            # --- INÍCIO DA ALTERAÇÃO 2/3: Calcula os valores de erro ---
            err_f0 = f0_adj * abs(inct_f0 - 1)
            err_Rc = Rc_adj * abs(inct_rc - 1)
            err_Rt = Rt_adj * abs(inct_rt - 1)
            err_fbg = fbg_adj * abs(inct_fbg - 1)
            
            resultados_sinteticos.append({
                'Synthetic_ID': i, 'Divisoes': num_bins,
                'f0_adj': f0_adj, 'Rc_adj': Rc_adj, 'Rt_adj': Rt_adj, 'fbg_adj': fbg_adj,
                'err_f0': err_f0, 'err_Rc': err_Rc, 'err_Rt': err_Rt, 'err_fbg': err_fbg
            })

            r_continuo_synth = np.linspace(0, r_max, 200)
            curva_king_synth = funcao_king(r_continuo_synth, f0_adj, Rc_adj, Rt_adj, fbg_adj)
            dados_da_curva_de_king_sintetica.append({'r': r_continuo_synth, 'curve': curva_king_synth})

        if not resultados_sinteticos:
            print(f"  No valid synthetic clusters could be processed for {cluster_name}.")
            continue

        output_folder_sintetico = os.path.join(BASE_DIR, cluster_name, "Sinteticos")
        os.makedirs(output_folder_sintetico, exist_ok=True)

        # --- FASE 3A: GRÁFICO DE DENSIDADE (SINTÉTICOS EM CINZA + REAL EM VERMELHO) ---
        print(f"\n  Generating overlaid density profile graph (Synthetics + Real)...")
        plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        for perfil in dados_de_densidade_sintetica:
            plt.plot(perfil['bins'], perfil['density'], marker='o', linestyle=' ', color='gray', alpha=0.05, zorder=1)
        plt.plot([], [], marker='o', linestyle=' ', color='gray', label='Synthetic density profile clusters')
        if real_density is not None:
            plt.plot(real_bin_centers, real_density, marker='o', linestyle=' ', color='red', label='Observed density profile', zorder=2)
        plt.title(f'{cluster_name}: Synthetic and\nobserved RDP (Bins = {num_bins})', fontsize=TAMANHO_FONTE-1)
        plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        graph_path = os.path.join(output_folder_sintetico, f"{cluster_name}_real_and_synthetic_rdp.{FORMATO_IMAGEM}")
        plt.savefig(graph_path)
        plt.close()
        print(f"  Overlay graph saved to: {graph_path}")

        # --- FASE 3B: GRÁFICO DE DENSIDADE (APENAS SINTÉTICOS EM AZUL) ---
        print(f"  Generating overlaid density profile graph (Synthetics Only)...")
        plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        for perfil in dados_de_densidade_sintetica:
            plt.plot(perfil['bins'], perfil['density'], marker='o', linestyle=' ', color='blue', alpha=0.05, label='Synthetic clusters')
        plt.title(f'{cluster_name}: Synthetic RDP\n(Bins = {num_bins})', fontsize=TAMANHO_FONTE-1)
        plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
        plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        graph_path_azul = os.path.join(output_folder_sintetico, f"{cluster_name}_synthetic_rdp.{FORMATO_IMAGEM}")
        plt.savefig(graph_path_azul)
        plt.close()
        print(f"  Synthetics-only graph saved to: {graph_path_azul}")

        # --- FASE 3C: GRÁFICOS DA CURVA DE KING ---
        if real_density is not None:
            print(f"  Generating King Fit graphs...")
            r_continuo_real = np.linspace(0, real_r_max, 500)
            curva_king_real = funcao_king(r_continuo_real, real_f0_adj, real_Rc_adj, real_Rt_adj, real_fbg_adj)            
            plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
            for curva in dados_da_curva_de_king_sintetica:
                plt.plot(curva['r'], curva['curve'], color='gray', alpha=0.05, zorder=1)            
            plt.plot([], [], color='gray', label='Preliminary synthetic King fit')
            plt.plot(r_continuo_real, curva_king_real, color='blue', linestyle='--', label='Preliminary King fit', zorder=2)
            plt.plot(real_bin_centers, real_density, marker='o', linestyle=' ', color='black', label='Observed density profile', zorder=3)            
            plt.title(f'{cluster_name}: Synthetic King fit\nestimated distribution (Bins = {num_bins})', fontsize=TAMANHO_FONTE-1)
            plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
            plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            graph_path_king = os.path.join(output_folder_sintetico, f"{cluster_name}_real_and_synthetic_king_fit.{FORMATO_IMAGEM}")
            plt.savefig(graph_path_king)
            plt.close()
            print(f"  King Fit overlay graph saved to: {graph_path_king}")

            plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
            for curva in dados_da_curva_de_king_sintetica:
                plt.plot(curva['r'], curva['curve'], color='magenta', alpha=0.05)
            for perfil in dados_de_densidade_sintetica:
                plt.plot(perfil['bins'], perfil['density'], marker='o', linestyle=' ', color='blue', alpha=0.05)
            plt.title(f'{cluster_name}: Synthetic King fit\ndistribution (Bins = {num_bins})', fontsize=TAMANHO_FONTE-1)
            plt.xlabel('Radial distance (r) [arcmin]', fontsize=TAMANHO_FONTE)
            plt.ylabel('Surface density [stars / arcmin²]', fontsize=TAMANHO_FONTE)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            graph_path_king_azul = os.path.join(output_folder_sintetico, f"{cluster_name}_synthetic_king_fit.{FORMATO_IMAGEM}")
            plt.savefig(graph_path_king_azul)
            plt.close()
            print(f"  Synthetics-only King Fit graph saved to: {graph_path_king_azul}")

        # --- INÍCIO DA ALTERAÇÃO ---
        # --- FASE 4: CONSOLIDAÇÃO E SALVAMENTO DOS DADOS ---
        df_full = pd.DataFrame(resultados_sinteticos)
        
        # Monta o DataFrame final para salvar, com parâmetros e seus respectivos erros
        df_to_save = pd.DataFrame({
            'Synthetic_ID': df_full['Synthetic_ID'],
            'Divisoes': df_full['Divisoes'],
            'f0': df_full['f0_adj'],
            'err_f0': df_full['err_f0'],
            'Rc': df_full['Rc_adj'],
            'err_Rc': df_full['err_Rc'],
            'Rt': df_full['Rt_adj'],
            'err_Rt': df_full['err_Rt'],
            'fbg': df_full['fbg_adj'],
            'err_fbg': df_full['err_fbg'],
        })
        
        output_folder_resultados = os.path.join(output_folder_sintetico, "Resultados")
        os.makedirs(output_folder_resultados, exist_ok=True)
        output_path = os.path.join(output_folder_resultados, f"{cluster_name}_synthetic_parameters.csv")
        
        df_to_save.to_csv(output_path, index=False, sep=';')
        
        print(f"\n  Analysis of {len(df_to_save)} synthetic clusters complete.")
        print(f"  Results saved to: {output_path}")
        # --- FIM DA ALTERAÇÃO ---

    print("\n" + "="*60)
    print("Unified processing for all SYNTHETIC clusters is complete.")
    print("="*60)

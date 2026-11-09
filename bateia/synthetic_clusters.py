# bateia/synthetic_clusters.py

# Importa as configurações e bibliotecas centralizadas do projeto BATEIA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import (
    BASE_DIR, T_MAX_SINTETICOS, SYNTHETIC_RANGES, SCALE_VARIATION,
    FORMATO_IMAGEM, TAMANHO_FONTE, IMAGE_SIZE_PX, DPI
)

def transform_point(x, y, scale_factor, x_max, y_max, displacement_range):
    """
    Aplica escala, deslocamento e reflexão de borda a um ponto individual.
    """
    # Aplicar a escala
    original_x, original_y = x * scale_factor, y * scale_factor
    
    # --- ALTERAÇÃO MATEMÁTICA ---
    # O deslocamento agora é PROPORCIONAL À POSIÇÃO da própria estrela,
    # para corresponder exatamente ao script de referência.
    dx = original_x * np.random.uniform(*displacement_range)
    dy = original_y * np.random.uniform(*displacement_range)
    
    # Aplicar o deslocamento recém-calculado
    x_new, y_new = original_x + dx, original_y + dy

    # Se o ponto ultrapassa os limites, refletir para dentro
    if x_new > x_max:
        x_new = x_max - (x_new - x_max)
    elif x_new < -x_max:
        x_new = -x_max + (-x_new - x_max)

    if y_new > y_max:
        y_new = y_max - (y_new - y_max)
    elif y_new < -y_max:
        y_new = -y_max + (-y_new - y_max)

    return x_new, y_new

def gerar_aglomerados_sinteticos(cluster_name):
    """
    Lê os dados de um aglomerado processado, gera os arquivos CSV para
    os aglomerados sintéticos e cria um gráfico de visualização geral.
    """
    print(f"Iniciando geração de dados sintéticos para: {cluster_name}")
    
    # Usa o arquivo CSV processado, que é mais limpo e confiável
    input_file = os.path.join(BASE_DIR, cluster_name, "Posicoes", f"{cluster_name}_coords_processed.csv")

    try:
        # Lê o arquivo CSV, esperando um cabeçalho na primeira linha (header=0)
        data = pd.read_csv(input_file, sep=';', header=0)
    except FileNotFoundError:
        print(f"Arquivo de entrada não encontrado: {input_file}")
        print("Certifique-se de que a ETAPA 1 (conversor_dados) foi executada para este aglomerado.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo {input_file}: {e}")
        return

    # Valida se as colunas necessárias existem
    if 'X' not in data.columns or 'Y' not in data.columns:
        print(f"Erro: O arquivo {input_file} não contém as colunas 'X' e 'Y' necessárias.")
        return

    # Cria os diretórios de saída
    jpg_output_dir = os.path.join(BASE_DIR, cluster_name, "Sinteticos")
    csv_output_dir = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "CSV")
    os.makedirs(jpg_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    # Calcula os limites máximos com base nos dados originais
    x_max = float(data['X'].max()) * 60
    y_max = float(data['Y'].max()) * 60

    synthetic_clusters_for_plotting = []
    # Loop principal para gerar cada aglomerado sintético
    for i in range(1, T_MAX_SINTETICOS + 1):
        synthetic_data = data.copy()

        # Define os fatores de transformação para este aglomerado
        scale_factor = 1 + np.random.uniform(-SCALE_VARIATION, SCALE_VARIATION)
        displacement_range = next(value for key, value in SYNTHETIC_RANGES.items() if key[0] <= i <= key[1])

        # A chamada para 'transform_point' continua a mesma, mas a lógica interna mudou
        synthetic_data[['X', 'Y']] = synthetic_data.apply(
            lambda row: transform_point(row['X'] * 60, row['Y'] * 60, scale_factor, x_max, y_max, displacement_range), 
            axis=1, 
            result_type='expand'
        )

        # Salva o novo aglomerado sintético em um arquivo CSV
        output_file_path = os.path.join(csv_output_dir, f"{cluster_name}_Sintetico_{i}.csv")
        synthetic_data.to_csv(output_file_path, index=False, sep=';')
        synthetic_clusters_for_plotting.append(synthetic_data)

    # --- Geração do Gráfico de Visualização ---
    plt.figure(figsize=(IMAGE_SIZE_PX/DPI, IMAGE_SIZE_PX/DPI))
    
    # Plota os dados originais
    plt.scatter(data['X'] * 60, data['Y'] * 60, color='black', alpha=1, s=2)
    
    # Plota todos os dados sintéticos com transparência
    for synth_data in synthetic_clusters_for_plotting:
        plt.scatter(synth_data['X'], synth_data['Y'], color='magenta', alpha=0.01, s=2)

    plt.xlabel(r'$\Delta \alpha * \cos(\delta)$ [arcmin]', fontsize=TAMANHO_FONTE)
    plt.ylabel(r'$\Delta \delta$ [arcmin]', fontsize=TAMANHO_FONTE)
    plt.xlim(-x_max, x_max)
    plt.ylim(-y_max, y_max)
    plt.tick_params(axis='both', labelsize=10)
    
    # Salva o gráfico
    jpg_output_path = os.path.join(jpg_output_dir, f"{cluster_name}_synthetics_overview.jpg")
    plt.savefig(jpg_output_path, format=FORMATO_IMAGEM, bbox_inches='tight', dpi=DPI)
    plt.close()
    print(f"Gráfico de visualização geral salvo em: {jpg_output_path}")

def gerar_imagens_sinteticas(cluster_name):
    """
    Gera imagens individuais para cada CSV sintético, prontas para a CNN.
    """
    print(f"Gerando imagens individuais para CNN do cluster: {cluster_name}")
    
    image_output_dir = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "CNN")
    csv_dir = os.path.join(BASE_DIR, cluster_name, "Sinteticos", "CSV")
    os.makedirs(image_output_dir, exist_ok=True)

    # Loop para processar cada arquivo CSV sintético
    for t in range(1, T_MAX_SINTETICOS + 1):
        csv_path = os.path.join(csv_dir, f"{cluster_name}_Sintetico_{t}.csv")

        if os.path.exists(csv_path):
            try:
                synthetic_data = pd.read_csv(csv_path, delimiter=';')
                if not synthetic_data.empty:
                    x_data = synthetic_data.iloc[:, 0]
                    y_data = synthetic_data.iloc[:, 1]
                    
                    # Usa limites simétricos para manter a consistência da imagem
                    abs_max = max(x_data.abs().max(), y_data.abs().max()) if not x_data.empty else 1

                    plt.figure(figsize=(IMAGE_SIZE_PX/DPI, IMAGE_SIZE_PX/DPI), dpi=DPI)
                    plt.scatter(x_data, y_data, color='black', alpha=1, s=0.8)
                    plt.xlim(-abs_max, abs_max)
                    plt.ylim(-abs_max, abs_max)
                    plt.axis('off') # Remove eixos e bordas

                    # Salva a imagem sem margens
                    image_name = f"{cluster_name}_Sintetico_{t}.{FORMATO_IMAGEM}"
                    image_path = os.path.join(image_output_dir, image_name)
                    plt.savefig(image_path, format=FORMATO_IMAGEM, bbox_inches='tight', pad_inches=0)
                    plt.close()
            except Exception as e:
                print(f"Erro ao processar imagem para {csv_path}: {e}")

def processar_aglomerados_sinteticos(lista_clusters):
    """
    Função orquestradora principal para a geração de dados e imagens sintéticas.
    """
    for cluster in lista_clusters:
        gerar_aglomerados_sinteticos(cluster)
        gerar_imagens_sinteticas(cluster)

if __name__ == '__main__':
    print("Este script é um módulo e deve ser chamado a partir do bateia_exec.py")

# bateia/conversor_dados.py
"""
Módulo com funções para configurar, converter e gerar imagens de aglomerados.
As funções são projetadas para operar em um aglomerado por vez.
"""
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import io
from bateia.config import BASE_DIR, IMAGE_SIZE_PX, DPI

def get_float_input(prompt_message):
    """Garante que o usuário digite um número de ponto flutuante."""
    while True:
        try:
            return float(input(prompt_message))
        except ValueError:
            print("  Erro: Por favor, digite um número válido.")

def configurar_parametros_aglomerado(nome_aglomerado):
    """Pede e salva os 8 parâmetros ótimos para um aglomerado específico."""
    print(f"\nConfigurando os 8 parâmetros para '{nome_aglomerado}':")
    # ATENÇÃO: Os nomes dos arquivos de erro devem corresponder ao que 'real_clusters.py' espera.
    # Ex: 'err_f0', 'err_Rc', etc.
    parametros_a_pedir = ['f0', 'err_f0', 'Rc', 'err_Rc', 'Rt', 'err_Rt', 'fbg', 'err_fbg']
    dados_coletados = {}

    for param in parametros_a_pedir:
        dados_coletados[param] = get_float_input(f"  - Digite o valor para '{param}': ")

    print("\nDados coletados. Salvando arquivos...")
    target_dir = os.path.join(BASE_DIR, nome_aglomerado, 'Target')
    os.makedirs(target_dir, exist_ok=True)

    for nome, valor in dados_coletados.items():
        # Define o prefixo 'mat_' para os arquivos de literatura
        prefixo = "mat_" if "err" in nome else "mat_"
        # Corrige o nome do arquivo de erro para o formato esperado
        nome_arquivo_salvar = nome.replace("err_","mat_err_") if "err" in nome else f"mat_{nome}"

        caminho_arquivo = os.path.join(target_dir, f"{nome_arquivo_salvar}_otm_{nome_aglomerado}.csv")
        
        # O cabeçalho do CSV deve corresponder ao nome do parâmetro para leitura fácil
        # Ex: err_Rc.csv -> cabeçalho 'Rc'
        nome_coluna = nome.replace("err_", "")
        pd.DataFrame({nome_coluna: [valor]}).to_csv(caminho_arquivo, index=False, sep=';')
        print(f"  -> Arquivo salvo: {os.path.basename(caminho_arquivo)}")
    
    print(f"\nSucesso! Parâmetros para '{nome_aglomerado}' foram salvos.")
    return True

def processar_um_aglomerado(nome_aglomerado):
    """Converte .dat para .csv e gera as imagens .jpg (_1 e _stars) para um único aglomerado."""
    print(f"\nProcessando arquivo de coordenadas para: {nome_aglomerado}")
    try:
        input_dir = os.path.join(BASE_DIR, nome_aglomerado, "Posicoes")
        input_filepath = os.path.join(input_dir, f"{nome_aglomerado}_coords.dat")
        output_filepath = os.path.join(input_dir, f"{nome_aglomerado}_coords_processed.csv")

        # 1. Conversão para CSV
        df = pd.read_csv(input_filepath, sep='\s+', skiprows=1, names=['ID', 'RAlin(J2016)', 'DEClin(J2016)', 'Gmag', 'BPmag', 'RPmag'])
        df['X'] = df['RAlin(J2016)']
        df['Y'] = df['DEClin(J2016)']

        df[['X', 'Y']].to_csv(output_filepath, index=False, sep=';')
        print(f"  - Sucesso na conversão para CSV.")

        # Criação do diretório de saída (usado para ambas as imagens)
        cnn_dir = os.path.join(BASE_DIR, nome_aglomerado, "CNN")
        os.makedirs(cnn_dir, exist_ok=True)

        # ---------------------------------------------------------
        # 2. Geração da Imagem 1 (Padrão CNN - Limpa, sem eixos)
        # ---------------------------------------------------------
        plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        plt.scatter(df['X'], df['Y'], color='black', s=1.0)
        
        limite = max(df['X'].abs().max(), df['Y'].abs().max()) * 1.01
        plt.xlim(-limite, limite)
        plt.ylim(-limite, limite)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        caminho_imagem_1 = os.path.join(cnn_dir, f"{nome_aglomerado}_1.jpg")
        plt.savefig(caminho_imagem_1, bbox_inches='tight', pad_inches=0)
        plt.close() # Importante fechar para não sobrepor na próxima
        print(f"  - Sucesso na geração da imagem CNN: {os.path.basename(caminho_imagem_1)}")

        # ---------------------------------------------------------
        # 3. Geração da Imagem 2 (_stars - Visual com eixos e labels)
        # ---------------------------------------------------------
        
        # Prepara dados (Convertendo para arcmin * 60 conforme solicitado)
        x_data = df['X'] * 60
        y_data = df['Y'] * 60

        # Calcula os limites baseados nos dados convertidos
        x_max = x_data.abs().max()
        y_max = y_data.abs().max() # Se quiser manter quadrado perfeito, use max(x_max, y_max) para ambos

        plt.figure(figsize=(IMAGE_SIZE_PX / DPI, IMAGE_SIZE_PX / DPI), dpi=DPI)
        plt.scatter(x_data, y_data, color='black', alpha=1, s=0.8)
        
        plt.xlim(-x_max, x_max)
        plt.ylim(-y_max, y_max)
        # plt.axis('off') # Removido para que os labels abaixo apareçam

        plt.xlabel(r'$\Delta \alpha * \cos(\delta)$ [arcmin]', fontsize=14)
        plt.ylabel(r'$\Delta \delta$ [arcmin]', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        caminho_imagem_stars = os.path.join(cnn_dir, f"{nome_aglomerado}_stars.jpg")
        # pad_inches aumentado levemente para não cortar os textos dos eixos
        plt.savefig(caminho_imagem_stars, format='jpg', bbox_inches='tight')
        plt.close()
        print(f"  - Sucesso na geração da imagem Visual: {os.path.basename(caminho_imagem_stars)}")

        return True

    except Exception as e:
        print(f"  ERRO ao processar '{nome_aglomerado}': {e}")
        return False

def processar_todos_aglomerados(lista_aglomerados):
    """Executa o processamento para uma lista de aglomerados."""
    for nome in lista_aglomerados:
        processar_um_aglomerado(nome)

# --- INÍCIO DA NOVA FUNÇÃO ---
def salvar_parametros_literatura():
    """
    Lê o arquivo 'cluster_params.csv' da pasta raiz e salva os parâmetros
    de literatura e de erro nos diretórios 'Target' de cada aglomerado.
    """
    caminho_csv_principal = os.path.join(BASE_DIR, "cluster_params.csv")
    
    if not os.path.exists(caminho_csv_principal):
        print(f"❌ ERRO: Arquivo '{os.path.basename(caminho_csv_principal)}' não encontrado na pasta raiz.")
        print("   Por favor, execute a opção de download primeiro.")
        return False

    print(f"\nLendo dados da literatura de '{os.path.basename(caminho_csv_principal)}'...")
    df = pd.read_csv(caminho_csv_principal)

    # Dicionário para mapear colunas do CSV para nomes de arquivos
    colunas_map = {
        'f0': 'mat_f0', 'err_f0': 'mat_err_f0',
        'Rc': 'mat_Rc', 'err_Rc': 'mat_err_Rc',
        'Rt': 'mat_Rt', 'err_Rt': 'mat_err_Rt',
        'fbg': 'mat_fbg', 'err_fbg': 'mat_err_fbg'
    }

    for index, row in df.iterrows():
        cluster_name = row['Cluster']
        print(f"\nProcessando parâmetros para: {cluster_name}")
        
        target_dir = os.path.join(BASE_DIR, cluster_name, 'Target')
        os.makedirs(target_dir, exist_ok=True)
        
        for col_nome, arquivo_prefixo in colunas_map.items():
            if col_nome not in row:
                print(f"  -> AVISO: Coluna '{col_nome}' não encontrada no CSV. Pulando.")
                continue
                
            valor = row[col_nome]
            
            # O nome da coluna no arquivo de saída é o nome base do parâmetro
            parametro_base = col_nome.replace('err_', '')
            
            file_name = f"{arquivo_prefixo}_otm_{cluster_name}.csv"
            full_file_path = os.path.join(target_dir, file_name)
            
            df_parametro = pd.DataFrame({parametro_base: [valor]})
            df_parametro.to_csv(full_file_path, index=False, sep=';')
            print(f"  -> Arquivo salvo: {file_name}")

    print("\nProcesso de popular parâmetros da literatura concluído com sucesso!")
    return True
# --- FIM DA NOVA FUNÇÃO ---

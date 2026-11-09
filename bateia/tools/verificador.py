# bateia/verificador.py
"""
Módulo responsável por verificar o estado dos dados locais dos aglomerados.
"""
import os
import logging

def listar_aglomerados_locais(base_dir):
    """
    Varre um diretório base e retorna uma lista de nomes de aglomerados
    que possuem um arquivo .dat válido.
    """
    aglomerados_encontrados = []
    try:
        if not os.path.isdir(base_dir):
            return []
            
        for nome_pasta in os.listdir(base_dir):
            caminho_completo = os.path.join(base_dir, nome_pasta)
            if os.path.isdir(caminho_completo):
                caminho_dat = os.path.join(caminho_completo, "Posicoes", f"{nome_pasta}_coords.dat")
                if os.path.isfile(caminho_dat):
                    aglomerados_encontrados.append(nome_pasta)
    except FileNotFoundError:
        return [] # Retorna lista vazia se o diretório base não existir
        
    return aglomerados_encontrados

def verificar_aglomerados_locais(base_dir):
    """
    Usa a função de listagem para encontrar e exibir os aglomerados locais
    em formato de checklist.
    """
    print("--- Verificação de Aglomerados Locais ---")
    print(f"Buscando em: {base_dir}\n")
    
    # Agora usa a função de listagem reutilizável
    aglomerados_encontrados = listar_aglomerados_locais(base_dir)

    if not aglomerados_encontrados:
        print("Nenhum aglomerado com arquivo '_coords.dat' válido foi encontrado nas subpastas 'Posicoes'.")
    else:
        print("Os seguintes aglomerados foram encontrados com seus arquivos .dat válidos:")
        aglomerados_encontrados.sort() # Ordena para uma exibição limpa
        for nome_aglomerado in aglomerados_encontrados:
            print(f"  [✓] {nome_aglomerado}")

    input("\nPressione Enter para voltar ao menu principal...")

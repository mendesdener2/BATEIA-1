# bateia/gdrive_downloader.py
import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Se modificar esses escopos, delete o arquivo token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# ID da pasta do Google Drive que você compartilhou
GDRIVE_FOLDER_ID = '1P7QOAzGZ0T8jxc0mGuLIHT4j8A25kg3C'

def autenticar():
    """
    Realiza a autenticação com a API do Google Drive usando credentials.json.
    Cria ou atualiza o token.json para logins futuros.
    Retorna um objeto de serviço da API autenticado.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    try:
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        print(f"ERRO: Falha ao construir o serviço da API do Drive: {e}")
        return None

def listar_arquivos_dat(service):
    """
    Lista todos os arquivos que terminam com '_coords.dat' na pasta do GDrive.
    Retorna uma lista limpa com os nomes dos aglomerados.
    """
    nomes_aglomerados = []
    page_token = None
    try:
        while True:
            # --- CORREÇÃO AQUI ---
            # Adicionamos 'and trashed = false' para forçar a API a buscar
            # a lista mais recente e ignorar qualquer cache de arquivos na lixeira.
            query = f"'{GDRIVE_FOLDER_ID}' in parents and name contains '_coords.dat' and trashed = false"
            
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(name)',
                pageToken=page_token
            ).execute()
            
            for file in response.get('files', []):
                # Remove o sufixo '_coords.dat' para obter o nome limpo
                nome_limpo = file.get('name').replace('_coords.dat', '')
                nomes_aglomerados.append(nome_limpo)
                
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        return nomes_aglomerados
    except Exception as e:
        print(f"ERRO: Não foi possível listar os arquivos do Google Drive: {e}")
        return []

def baixar_arquivo_dat(service, nome_aglomerado, caminho_salvar):
    """
    Busca por um arquivo .dat em uma pasta específica do GDrive e o baixa.
    """
    nome_arquivo_dat = f"{nome_aglomerado}_coords.dat"
    try:
        query = f"name = '{nome_arquivo_dat}' and '{GDRIVE_FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print(f"AVISO: Arquivo '{nome_arquivo_dat}' não encontrado no Google Drive.")
            return False

        file_id = items[0].get('id')
        request = service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
        
        with io.FileIO(caminho_salvar, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.", end='\r')
        
        print(f"\nSucesso! Arquivo '{nome_arquivo_dat}' baixado para '{caminho_salvar}'")
        return True
    except Exception as e:
        print(f"ERRO: Ocorreu um problema durante o download: {e}")
        return False
    
    # Adicione esta função ao final do seu arquivo gdrive_downloader.py


def baixar_arquivo_csv_raiz(servico, nome_arquivo, caminho_destino):
    """
    Procura por um arquivo específico na pasta do Drive e o baixa para a pasta raiz do projeto.
    
    Args:
        servico: Objeto de serviço da API do Google Drive.
        nome_arquivo (str): O nome do arquivo a ser baixado (ex: 'cluster_params.csv').
        caminho_destino (str): O caminho completo onde o arquivo será salvo.
    """
    try:
        print(f"\nProcurando por '{nome_arquivo}' no Google Drive...")
        
        # Procura pelo arquivo na pasta que contém os arquivos .dat
        # (Assumindo que está na mesma pasta)
        query = f"name = '{nome_arquivo}' and trashed = false"
        results = servico.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print(f"  AVISO: Arquivo '{nome_arquivo}' não encontrado no Google Drive. Pulando o download.")
            return False

        file_id = items[0]['id']
        file_name = items[0]['name']
        
        print(f"  Encontrado. Baixando '{file_name}'...")
        request = servico.files().get_media(fileId=file_id)
        
        # Faz o download do arquivo
        fh = io.FileIO(caminho_destino, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"    Download {int(status.progress() * 100)}%.")
            
        print(f"  '{file_name}' foi baixado com sucesso para a pasta raiz.")
        return True

    except Exception as e:
        print(f"  ERRO ao baixar '{nome_arquivo}': {e}")
        return False

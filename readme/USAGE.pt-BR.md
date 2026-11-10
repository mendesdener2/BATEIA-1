[English](USAGE.md) · [Português (Brasil)](USAGE.pt-BR.md) · [Español](USAGE.es.md)

[← README](README.pt-BR.md)

# Instalação, dados e uso

[← README](README.pt-BR.md)

<a id="installation"></a>
## Instalação passo a passo

### 1. Abra a pasta do projeto

Obtenha o código completo e abra um terminal na pasta que contém `bateia_exec.py` e `environment.yml`. Para uma pasta `bateia-1` no diretório atual:

```sh
cd bateia-1
```

O pacote de documentação sozinho não inclui o código executável. Instale Conda ou Miniconda primeiro e use um terminal em que `conda` esteja disponível. É necessário acesso à internet para dependências e, possivelmente, dados e pesos pré-treinados.

### 2. Crie e ative o ambiente

```sh
conda env create -f environment.yml
conda activate bateia
```

O arquivo solicita **Python 3.12**, dependências científicas Python, Optuna, ferramentas da API Google e componentes PyTorch. A seção pip adiciona o índice **PyTorch nightly com CUDA 12.8**, mas não fixa versões nem exige explicitamente pré-lançamentos. É uma receita de ambiente, não uma instalação travada.

O uso de GPU exige driver NVIDIA e distribuição PyTorch compatíveis. O código seleciona CPU quando CUDA está indisponível, mas a execução em CPU e a compatibilidade Windows/Linux/macOS não foram testadas para esta documentação. Se a resolução falhar, examine o erro e adapte PyTorch à máquina; não há ambiente alternativo verificado aqui.

### 3. Confira o Python e a visibilidade das GPUs

```sh
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Espere Python 3.12. `CUDA available: False` significa que esse interpretador não consegue usar CUDA; não identifica a causa nem valida o restante do pipeline.

<a id="data"></a>
## Prepare dados e modelos

| Material | Identificador persistente | Finalidade |
| --- | --- | --- |
| Dataset | [10.5281/zenodo.24087976](https://doi.org/10.5281/zenodo.24087976) | Dataset de pesquisa dos experimentos BATEIA-1 |
| Modelos treinados e scalers | [10.5281/zenodo.24094057](https://doi.org/10.5281/zenodo.24094057) | Regressores salvos e scalers associados |

### Baixe e organize os recursos publicados

1. Abra o registro do dataset e baixe os arquivos necessários ao experimento. Para aplicar um modelo salvo a um aglomerado próprio, siga o [guia de dados próprios](CUSTOM_DATA.pt-BR.md), sem presumir que precisa baixar toda a base de treinamento.
2. Abra o registro dos modelos e baixe o regressor, seus scalers de entrada/saída e os metadados disponíveis do experimento. Preserve caminhos relativos e nomes.
3. Extraia arquivos compactados, se houver. Localize a pasta que contém diretamente os aglomerados e `cluster_params.csv`, quando necessário; defina `BATEIA_ROOT` para essa pasta de dados. Não aponte para a pasta que apenas contém downloads compactados.
4. Localize separadamente `Redes Neurais`, como no [guia de dados próprios](CUSTOM_DATA.pt-BR.md), e use o caminho real extraído nos exemplos. Pastas externas acrescentadas pela extração não integram os nomes dos modelos.
5. Registre os dois DOIs, arquivos/versões usados e o commit do código. Separe os artefatos baixados das saídas de novos treinamentos.

Os arquivos do Zenodo são obtidos manualmente. **Setup and Data Download** ainda autentica no Google Drive; não é um baixador do Zenodo. A estrutura abaixo define o que o código espera após a extração.

O diretório padrão é `database/`, na raiz do projeto. Opcionalmente, defina `BATEIA_ROOT` **diretamente para a pasta de dados**, antes de iniciar Python. São comandos alternativos para terminais diferentes; crie e preencha a pasta escolhida antes:

```powershell
# Windows PowerShell
$env:BATEIA_ROOT = "C:\BATEIA-data"
```

```sh
# Shell Linux/macOS
export BATEIA_ROOT="/path/to/BATEIA-data"
```

Dados de pesquisa e regressores treinados são distribuídos separadamente pelos registros Zenodo acima. Sem `BATEIA_ROOT`, prepare a estrutura esperada em `database/`.

```text
database/                         # ou BATEIA_ROOT
├── cluster_params.csv
└── <Cluster>/
    ├── Posicoes/
    │   └── <Cluster>_coords.dat
    └── Target/                   # alvos gerados da literatura
```

- **Coordenadas:** uma linha de cabeçalho e seis colunas separadas por espaços: `ID`, `RAlin(J2016)`, `DEClin(J2016)`, `Gmag`, `BPmag`, `RPmag`. O conversor ignora o cabeçalho e copia as coordenadas para `X` e `Y`. Use dados preparados com a convenção do fluxo científico; a conversão não determina o centro do aglomerado nem seleciona membros Gaia de catálogos brutos arbitrários.
- **Parâmetros da literatura:** `cluster_params.csv`, separado por vírgulas, usa `Cluster`, `f0`, `err_f0`, `Rc`, `err_Rc`, `Rt`, `err_Rt`, `fbg`, `err_fbg`. Os nomes devem corresponder aos caminhos e metadados em `bateia/config_data/clusters.py`. Preserve unidades e convenções de incerteza da referência.
- **Google Drive:** a opção **1** do menu principal usa OAuth pelo navegador com seu próprio `credentials.json` local e mantém a autorização em `token.json`. Exige acesso à pasta configurada em `bateia/tools/gdrive_downloader.py`. O acesso público e a disponibilidade dos downloads não foram verificados. Mantenha esses arquivos pessoais fora do controle de versão e de pacotes compartilhados.
- **Arquivos locais:** coloque entradas preparadas na estrutura acima. Os CSVs de `Target/` também precisam estar preparados. O comando que os gera fica no submenu de configuração, que atualmente autentica no Drive antes de abrir; um assistente totalmente offline não está implementado.
- **Pesos:** as extratoras torchvision solicitam pesos ImageNet (`IMAGENET1K_V1`), que podem ser baixados para o cache PyTorch no primeiro uso. São extratoras visuais, não regressores de parâmetros BATEIA. Use os [modelos BATEIA publicados e seus scalers](https://doi.org/10.5281/zenodo.24094057) na inferência com regressores salvos, ou treine novos pelo menu; consulte a [inferência com dados próprios](CUSTOM_DATA.pt-BR.md).

<a id="usage"></a>
## Seu primeiro uso

Execute na raiz do projeto para resolver corretamente caminhos relativos de autenticação e logs:

```sh
python bateia_exec.py
```

Os nomes abaixo são os da aplicação; os idiomas da documentação não alteram a interface.

1. Escolha **[1] Setup and Data Download** se usar Drive. Baixe/atualize parâmetros da literatura, gere parâmetros-alvo e depois baixe os arquivos `.dat`. Isso exige sua própria conta autorizada e configuração OAuth.
2. Escolha **[3] Verify Local Clusters** no menu principal. Confirme a detecção dos aglomerados esperados. A detecção verifica o caminho do arquivo de coordenadas, não a qualidade científica.
3. Escolha **[2] Run Analysis and AI Pipeline → [1] Run Data Analysis Stages**. Selecione um aglomerado preparado e confirme. Execute processamento e análise real antes das etapas dependentes; depois gere/analise sintéticos e extraia características. Escolha GAP, sem GAP ou ambos quando solicitado. Mapas de calor são uma etapa opcional de análise.
4. Com características e alvos disponíveis para uma amostra suficiente, escolha **Run AI Training** no menu de execução. Selecione um modelo e um modo de características correspondente. Use **AM Methodology (Uncontaminated)** com divisão explícita por aglomerado para avaliar casos inéditos. A divisão manual exige pelo menos 20 aglomerados (15 de treino, 4 de validação e os restantes de teste). A estratificada permite selecionar os aglomerados de teste e divide o desenvolvimento em 70/30 por perfis K-Means, sujeita a restrições de tamanho e grupos. **BM Methodology (Contaminated)** usa todos os dados e não demonstra generalização para aglomerados inéditos.
5. Inspecione as saídas; use **[5] Generate Results and Paper Reports** quando as predições necessárias existirem. Escolha **[7] Exit** para sair.

Comece com um aglomerado para inspecionar o pré-processamento, não para executar a divisão de treinamento documentada. A geração sintética tem padrão de até 1.000 amostras por aglomerado. Seleções grandes podem exigir bastante tempo, RAM, VRAM e disco.

**Atenção à numeração:** mapas de calor são a etapa 6 da análise, enquanto o treinamento também aparece como “Stage 6” em outras telas. Siga os nomes e dependências. **[4] Reset Workspace apaga dados/modelos gerados** conforme a operação escolhida.

### Onde ficam os resultados

Cada aglomerado recebe conteúdo gerado em `Posicoes/`, `Target/`, `CNN/` e `Sinteticos/`, conforme as etapas. As características sintéticas ficam em `Sinteticos/Features/<EXTRACTOR>/` (`.pt` ou `_semgap.pt`) e as tabelas de parâmetros em `Sinteticos/Resultados/`. As saídas de treinamento ficam em `Redes Neurais/`, com modelos/scalers, CSVs de predição e curvas de aprendizado em subpastas de execução/modelo. Os arquivos específicos dependem das etapas e do modelo.



<a id="docs"></a>
## Documentação e solução de problemas

| Recurso | Conteúdo |
| --- | --- |
| [Arquitetura](ARQUITETURA.md) | Módulos e estrutura do projeto |
| [Guia de modelos](MODELS_GUIDE.pt-BR.md) | GAP, regressores e otimização |
| [Paralelismo em GPUs](PARALELISMO-GPUS.md) | Caminhos existentes para múltiplas GPUs |
| [Ambiente](../environment.yml) | Dependências declaradas |
| [Artigo](https://doi.org/10.1016/j.ascom.2026.101119) · [Dissertação](http://urlib.net/8JMKD2USNRW34T/4FFHT62) | Metodologia científica e bibliografia |

Os guias técnicos estão disponíveis em inglês, português brasileiro e espanhol. Use o seletor no topo de cada guia. Consulte o código quando divergir do menu documentado.

| Sintoma | Primeira verificação |
| --- | --- |
| `conda` não encontrado | Abra um terminal com Conda disponível após instalar. |
| Erro de importação/dependência | Ative `bateia`; confira a instalação e o interpretador selecionado. |
| Nenhum aglomerado encontrado | Confira `BATEIA_ROOT` e `<Cluster>/Posicoes/<Cluster>_coords.dat`. |
| Falha no Drive | Confira OAuth, permissões da conta e pasta configurada. |
| Características/alvos ausentes | Conclua geração/análise sintética e extração; confira modo GAP, extratora, nomes e parâmetros-alvo. |
| Memória insuficiente | Reduza o trabalho selecionado e revise configurações de geração/treinamento. |

### Notebooks e Kaggle

Uma biblioteca conveniente para notebooks é uma direção futura. Não há notebook Kaggle pronto, interface estável de alto nível `fit`/`predict` ou instalação suportada para notebooks nesta versão. Módulos internos exigem adaptar caminhos, dependências, autenticação e comportamento interativo/multiprocessamento. O ponto de entrada documentado continua sendo o menu no terminal.

<a id="references"></a>
## Referências científicas e citação

As entradas foram conferidas no artigo e na dissertação fornecidos. Cite o artigo para a metodologia BATEIA-1; consulte a dissertação para o estudo ampliado de modelos.

1. Mendes, D. C.; Santiago Júnior, V. A.; Angelo, M. S. (2026). **BATEIA-1: An Approach for determining the structural parameters of open star clusters via machine and deep learning**. *Astronomy and Computing*, 56, 101119. [DOI](https://doi.org/10.1016/j.ascom.2026.101119).
2. Mendes, Dener César (2026). **BATEIA-1: An approach for determining the structural parameters of star clusters via machine and deep learning**. Master's dissertation, Applied Computing, Instituto Nacional de Pesquisas Espaciais (INPE). [INPE](http://urlib.net/8JMKD2USNRW34T/4FFHT62).
3. King, I. (1962). **The structure of star clusters. I. An empirical density law**. *The Astronomical Journal*, 67, 471.
4. Angelo, M. S.; Santos, J. F. C.; Corradi, W. J. B.; Maia, F. F. S. (2025). **Exploring the dynamical state of Galactic open clusters using Gaia DR3 and observational parameters**. *Monthly Notices of the Royal Astronomical Society*, 539(3), 2513–2536. [DOI](https://doi.org/10.1093/mnras/staf584).
5. Hunt, E. L.; Reffert, S. (2024). **Improving the open cluster census. III. Using cluster masses, radii, and dynamics to create a cleaned open cluster catalogue**. *Astronomy & Astrophysics*, 686, A42.

```bibtex
@article{mendes2026bateia,
  author  = {Mendes, Dener Cesar and Santiago Júnior, Valdivino Alexandre and Angelo, Mateus de Souza},
  title   = {{BATEIA-1}: An Approach for determining the structural parameters of open star clusters via machine and deep learning},
  journal = {Astronomy and Computing},
  volume  = {56},
  pages   = {101119},
  year    = {2026},
  doi     = {10.1016/j.ascom.2026.101119}
}
```


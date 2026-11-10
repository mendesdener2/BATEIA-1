<p align="center"><img src="../logo-bateia.png" alt="BATEIA-1 logo" width="330"></p>

<p align="center"><a href="../README.md">English</a> · <a href="../readme/README.pt-BR.md">Português (Brasil)</a> · <a href="../readme/README.es.md">Español</a></p>

# BATEIA-1

**Balanced ArTificial-intelligence Method for EstImating parameters of stAr clusters**

Estime parâmetros estruturais de aglomerados estelares abertos a partir de mapas estelares, usando dados sintéticos, extratoras visuais pré-treinadas e regressão.

[Dataset](https://doi.org/10.5281/zenodo.24087976) · [Modelos treinados e scalers](https://doi.org/10.5281/zenodo.24094057)

<a id="overview"></a>
## Resumo científico

Aglomerados estelares abertos são laboratórios naturais para estudar a evolução estelar e a estrutura da Galáxia. O BATEIA-1 estima seus parâmetros estruturais combinando o perfil analítico de King com aprendizado de máquina e aprendizado profundo aplicados a mapas estelares derivados de dados astrométricos e fotométricos do Gaia DR3. A abordagem estima densidade estelar central, raio do núcleo, raio de maré e densidade estelar de fundo, buscando automatizar a caracterização da estrutura dos aglomerados.

O estudo associado utiliza nove redes neurais profundas pré-treinadas para extração de características visuais, seguidas por modelos Multilayer Perceptron (MLP) e Support Vector Regression (SVR). Para M67, NGC 188 e NGC 6811, o artigo relata diferenças em relação à literatura de **0,7–12,8% no raio de maré, 1,1–5,5% no raio do núcleo e 1,1–8,7% na densidade central**. Score-CAM e Ablation-CAM oferecem diagnósticos complementares das extratoras visuais. Esses resultados correspondem aos experimentos do estudo; o método preserva as hipóteses dos rótulos de referência baseados no perfil de King. Consulte o [artigo](https://doi.org/10.1016/j.ascom.2026.101119) para a avaliação e o escopo.

**Palavras-chave:** aglomerados estelares abertos; astrofísica; estrutura galáctica; Gaia DR3; parâmetros estruturais; perfil de King; função de King; densidade estelar; aprendizado de máquina; aprendizado profundo; extração de características; regressão; inteligência artificial explicável (XAI); Score-CAM; Ablation-CAM.

[Resumo](#overview) · [Pipeline](#pipeline) · [Exemplos](#examples) · [Início rápido](#start) · [Documentação](#docs) · [Citação](#citation)

<a id="pipeline"></a>
## Pipeline

O BATEIA-1 combina perfis de King e aprendizado de máquina para estimar **densidade central (`f0`), raio do núcleo (`Rc`), raio de maré (`Rt`) e densidade de fundo (`fbg`)**.

<p align="center"><img src="../bateia-diagram.png" alt="BATEIA-1 pipeline" width="1100"></p>

Das coordenadas estelares preparadas às imagens, aumento de dados sintéticos, características visuais e regressão de parâmetros. A IA Explicável oferece um diagnóstico paralelo das extratoras.

**Extratoras:** VGG13/16/19, ResNet50/101/152 e DenseNet121/161/201.  
**Regressores:** MLP, SVR, Random Forest, LSTM, GRU e 1D-CNN; características com GAP e sem GAP.

<a id="examples"></a>
## Exemplo de aplicação: M67

As figuras a seguir ilustram o fluxo do BATEIA-1 usando M67 como exemplo de aplicação, dos mapas estelares e dados sintéticos aos perfis de densidade radial e diagnósticos de explicabilidade. Elas apresentam resultados representativos do estudo associado; as instruções para analisar seu próprio aglomerado estão no [guia de dados próprios](CUSTOM_DATA.pt-BR.md).

### Mapas estelares e divisão radial

<table>
<tr><th align="center">Mapa estelar</th><th align="center">Visão dos sintéticos</th><th align="center">Divisão espacial: 16 anéis</th></tr>
<tr><td align="center"><a href="../readme/assets/M67_stars.jpg"><img src="../readme/assets/M67_stars.jpg" alt="Mapa estelar" width="230"></a></td><td align="center"><a href="../readme/assets/M67_synthetics_overview.jpg"><img src="../readme/assets/M67_synthetics_overview.jpg" alt="Visão dos sintéticos" width="230"></a></td><td align="center"><a href="../readme/assets/M67_spatial_dist_best.jpg"><img src="../readme/assets/M67_spatial_dist_best.jpg" alt="Divisão espacial: 16 anéis" width="230"></a></td></tr>
<tr><td valign="top">Posições estelares preparadas para M67, centradas no aglomerado. Os eixos mostram deslocamentos angulares em minutos de arco; a concentração central e o campo circundante definem a distribuição usada para construir a imagem de entrada.</td><td valign="top">Sobreposição das distribuições espaciais sintéticas (magenta) e posições reais (preto). A figura mostra a variação espacial introduzida pelo aumento de dados; não é uma nova observação nem um mapa de atribuição de características.</td><td valign="top">O mesmo campo dividido em 16 anéis radiais concêntricos, delimitados por linhas azuis tracejadas. As contagens por área anular constroem o perfil de densidade superficial; os círculos são limites dos bins, não diferentes raios ajustados do aglomerado.</td></tr>
</table>

### Perfis de densidade radial

<table>
<tr><th align="center">Perfil de King: BATEIA-1 e literatura</th><th align="center">Perfis de densidade observado e sintéticos</th></tr>
<tr><td align="center"><a href="../readme/assets/king_uncertainty_M67_AM.jpg"><img src="../readme/assets/king_uncertainty_M67_AM.jpg" alt="Perfil de King: BATEIA-1 e literatura" width="340"></a></td><td align="center"><a href="../readme/assets/M67_real_and_synthetic_rdp.jpg"><img src="../readme/assets/M67_real_and_synthetic_rdp.jpg" alt="Perfis de densidade observado e sintéticos" width="340"></a></td></tr>
<tr><td valign="top">Densidade superficial de M67 em função do raio: pontos dos dados (preto), curva de King do BATEIA-1 (azul tracejado) e faixa de incerteza da literatura de Angelo et al. (cinza). É um exemplo individual; a faixa não estabelece intervalo preditivo calibrado para novos aglomerados arbitrários.</td><td valign="top">Densidades radiais observadas (vermelho) sobrepostas às dos perfis sintéticos (cinza), com 16 divisões radiais. A dispersão mostra a variação da amostra sintética em torno do perfil observado a cada raio; não é intervalo de confiança das predições dos regressores.</td></tr>
</table>

### IA Explicável: famílias de extratoras

<table>
<tr><th align="center">DenseNet: Score-CAM e Ablation-CAM</th><th align="center">ResNet: Score-CAM e Ablation-CAM</th><th align="center">VGG: Score-CAM e Ablation-CAM</th></tr>
<tr><td align="center"><a href="../readme/assets/heatmap_densenet_m67.jpg"><img src="../readme/assets/heatmap_densenet_m67.jpg" alt="DenseNet: Score-CAM e Ablation-CAM" width="230"></a></td><td align="center"><a href="../readme/assets/heatmap_resnet_M67.jpg"><img src="../readme/assets/heatmap_resnet_M67.jpg" alt="ResNet: Score-CAM e Ablation-CAM" width="230"></a></td><td align="center"><a href="../readme/assets/heatmap_vgg_M67.jpg"><img src="../readme/assets/heatmap_vgg_M67.jpg" alt="VGG: Score-CAM e Ablation-CAM" width="230"></a></td></tr>
<tr><td valign="top">Painéis de entrada, Score-CAM e Ablation-CAM para DenseNet121, DenseNet161 e DenseNet201. Os mapas mostram onde cada extratora responde em M67 e como os padrões espaciais variam com a profundidade da rede e o método de visualização.</td><td valign="top">Painéis de entrada, Score-CAM e Ablation-CAM para ResNet50, ResNet101 e ResNet152. Compare localização e extensão espacial das respostas entre profundidades das redes residuais e métodos, mantendo o mesmo aglomerado como referência visual.</td><td valign="top">Painéis de entrada, Score-CAM e Ablation-CAM para VGG13, VGG16 e VGG19. Os mapas ilustram respostas que podem se estender a regiões fora do centro, permitindo inspecionar diferenças em relação às famílias DenseNet e ResNet.</td></tr>
</table>

Em cada painel de família, as colunas são entrada, Score-CAM e Ablation-CAM; as linhas identificam a variante da extratora. As cores representam respostas relativas em cada visualização. São diagnósticos das extratoras visuais, não intervalos de confiança, causalidade física ou explicações diretas de parâmetros dos regressores posteriores; não trate as cores como uma escala quantitativa comum entre painéis.

A abordagem se destina a aglomerados abertos não imersos descritos por perfis de King de componente única e preserva as hipóteses dos rótulos de referência. O artigo avalia MLP e SVR; a dissertação e o código abrangem regressores adicionais.

<a id="start"></a>
## Início rápido

### 1. Crie o ambiente

Instale Conda/Miniconda e abra um terminal na raiz do projeto:

```sh
conda env create -f environment.yml
conda activate bateia
```

O ambiente solicita Python 3.12 e adiciona o índice PyTorch nightly com CUDA 12.8. As versões não estão travadas; GPU exige driver/distribuição compatíveis. Instalação e compatibilidade de plataformas não foram testadas para esta documentação.

### 2. Prepare os dados

| Material | Identificador persistente | Finalidade |
| --- | --- | --- |
| Dataset | [10.5281/zenodo.24087976](https://doi.org/10.5281/zenodo.24087976) | Dataset de pesquisa dos experimentos BATEIA-1 |
| Modelos treinados e scalers | [10.5281/zenodo.24094057](https://doi.org/10.5281/zenodo.24094057) | Regressores salvos e scalers associados |

Baixe o [dataset](https://doi.org/10.5281/zenodo.24087976) para trabalhar com a amostra publicada, ou prepare coordenadas próprias seguindo o [guia de dados próprios](CUSTOM_DATA.pt-BR.md). Obtenha os [modelos treinados e scalers](https://doi.org/10.5281/zenodo.24094057) para aplicar regressores salvos; retreinar é um fluxo separado.

Coloque as pastas dos aglomerados em `database/`, ou defina `BATEIA_ROOT` para a pasta de dados que as contém diretamente. Mantenha o acervo de modelos separado e preserve os nomes de modelos/scalers. Consulte a [estrutura e configuração dos dados](USAGE.pt-BR.md#data).

O download do Zenodo é manual: **Setup and Data Download** continua sendo a via Google Drive e não baixa esses registros DOI. Essa via exige seu próprio OAuth e acesso à pasta configurada. Os pesos ImageNet das extratoras são separados e podem ser baixados no primeiro uso.

### 3. Abra o menu interativo

```sh
python bateia_exec.py
```

1. **Setup and Data Download**: prepare parâmetros da literatura e arquivos dos aglomerados, se usar Drive.
2. **Verify Local Clusters**: confira a detecção dos aglomerados preparados.
3. **Run Analysis and AI Pipeline**: processe dados reais, gere/analise sintéticos e extraia características antes de treinar.
4. **Run AI Training**: escolha modelo, modo GAP correspondente e divisão por aglomerado. Use AM para avaliar casos inéditos; BM utiliza todos os dados.
5. **Generate Results and Paper Reports**: inspecione predições e relatórios.

Comece com um aglomerado para inspecionar o pré-processamento. A divisão manual de treinamento documentada exige pelo menos 20 aglomerados. **Reset Workspace apaga dados/modelos gerados.** Consulte o [guia completo](USAGE.pt-BR.md) para dependências, divisões e saídas.

**Interface atual:** menu no terminal. Uma API simplificada para notebooks/Kaggle e instalação como pacote ainda não estão implementadas.

<a id="docs"></a>
## Documentação

- [Análise de dados próprios de aglomerados estelares](../readme/CUSTOM_DATA.pt-BR.md)
- [Instalação, dados e uso](USAGE.pt-BR.md)
- [Arquitetura](../readme/ARQUITETURA.md)
- [Modelos e otimização](../readme/MODELS_GUIDE.pt-BR.md)
- [Paralelismo em GPUs](../readme/PARALELISMO-GPUS.md)
- [Dependências](../readme/DEPENDENCIES.pt-BR.md)

<a id="citation"></a>
## Citação

Para a metodologia e os resultados experimentais, cite:

Mendes, D. C.; Santiago Júnior, V. A.; Angelo, M. S. (2026). **BATEIA-1: An Approach for determining the structural parameters of open star clusters via machine and deep learning.** *Astronomy and Computing*, 56, 101119. [DOI](https://doi.org/10.1016/j.ascom.2026.101119).

Ao usar dados de pesquisa ou artefatos treinados, cite também o [registro do dataset](https://doi.org/10.5281/zenodo.24087976) e/ou o [registro dos modelos](https://doi.org/10.5281/zenodo.24094057), identificando arquivos e versões utilizados.

### Licença

Consulte [LICENSE](../LICENSE) para o texto da GNU General Public License v3 distribuído neste repositório. Publicações e dados de terceiros têm licenças próprias.

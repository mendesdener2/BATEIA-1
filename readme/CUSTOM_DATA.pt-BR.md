[English](CUSTOM_DATA.md) · [Português (Brasil)](CUSTOM_DATA.pt-BR.md) · [Español](CUSTOM_DATA.es.md)

[← README](README.pt-BR.md)

# Análise de dados próprios de aglomerados estelares

Este guia descreve um fluxo avançado baseado no código atual e no acervo inspecionado `modelos-ia-bateia-mestrado`. Inclui um exemplo concreto de inferência SVR e orientações para redes profundas e treinamento com dados próprios. Os blocos Python usam funções existentes e **não constituem uma nova API de alto nível**. Foram conferidos contra o código e os nomes dos arquivos, mas não executados com os modelos neste ambiente. Essas verificações não comprovam compatibilidade de execução nem acurácia científica.

## 1. Escolha sua tarefa científica

| Tarefa | Entradas necessárias | Precisa de parâmetros-alvo conhecidos? |
| --- | --- | --- |
| Aplicar modelo salvo a novo aglomerado | Coordenadas preparadas; extratora, regressor e dois scalers correspondentes | Não, na inferência direta abaixo |
| Treinar novos regressores | Amostra rotulada, características/alvos sintéticos, divisão e configurações | Sim; os rótulos precisam de origem científica definida |
| Reproduzir a publicação | Dados, divisões, configurações, versão do código e artefatos originais | Rótulos e referências do experimento original |

Use aglomerados abertos não imersos compatíveis com o domínio de treinamento e perfis de King de componente única. Uma predição numericamente válida não demonstra generalização para outras populações, seleções de membros ou contaminação de campo.

## 2. Instale e localize os arquivos

**Recursos publicados:** [Dataset BATEIA-1](https://doi.org/10.5281/zenodo.24087976) · [Modelos treinados e scalers](https://doi.org/10.5281/zenodo.24094057). Para o exemplo de inferência direta, baixe os artefatos de modelo/scalers e prepare suas coordenadas; não é necessário baixar toda a base de treinamento para predizer um novo aglomerado. Use o dataset ao reproduzir ou retreinar os experimentos.

Baixe e extraia manualmente os arquivos necessários, preservando nomes e estrutura interna. A árvore abaixo descreve a coleção de pesquisa inspecionada; localize a pasta real `Redes Neurais` se a extração acrescentar outra pasta externa. Não deduza nomes de download pelo DOI nem renomeie artefatos históricos para imitar saídas atuais.

Siga a [instalação](USAGE.pt-BR.md#installation), ative `bateia` e execute na raiz do repositório. Use a mesma sessão Python ou kernel nas etapas 4–7. Defina `BATEIA_ROOT` antes de importar `bateia`; reinicie se já importou com outra raiz. O uso em notebook exige essa preparação manual e não representa uma API empacotada.

Separe o acervo dos seus dados observacionais. O caminho usado nos exemplos é a **subpasta `Redes Neurais`**, não a pasta externa do download. Arquivos `.joblib`/`.pkl` antigos exigem versões compatíveis de scikit-learn/joblib e podem executar código serializado; use artefatos confiáveis do autor. O ambiente Conda atual não comprova compatibilidade com o acervo histórico.

## 3. Entenda a arquitetura do acervo

```text
modelos-ia-bateia-mestrado/
└── Redes Neurais/
    ├── 1DCNN/          AM1DCNN/          BM1DCNN/
    ├── GRU/            AMGRU/            BMGRU/
    ├── LSTM/           AMLSTM/           BMLSTM/
    ├── MLPerceptron/   AMMLP/            BMMLP/
    ├── RandomForest/   AMRandomForest/   BMRandomForest/
    └── SVR/            AMSVR/            BMSVR/

# Within an experiment branch, for example SVR/AMSVR/:
Modelos/<EXTRACTOR>/    # .joblib or .pth; may also contain plots
Scaler/<EXTRACTOR>/     # scaler_x_*.pkl and scaler_y_*.pkl
Previsto/<EXTRACTOR>/   # historical prediction CSVs
Logs/                  # training records; nesting varies by family
```

Os nomes das pastas são literais. `MLPerceptron` contém experimentos MLP; as ramificações distinguem AM e BM. Há extratoras `DENSENET121`, `DENSENET161`, `DENSENET201`, `RESNET50/101/152` e `VGG13/16/19`.

Exemplo real de três arquivos correspondentes:

```text
SVR/AMSVR/Modelos/DENSENET121/f0_densenet121_svr_M67_gap_svr_model.joblib
SVR/AMSVR/Scaler/DENSENET121/scaler_x_f0_densenet121_svr_M67_gap.pkl
SVR/AMSVR/Scaler/DENSENET121/scaler_y_f0_densenet121_svr_M67_gap.pkl
```

`f0` identifica o parâmetro, `densenet121` a extratora, `svr` o regressor, `M67` o experimento histórico e `gap` a representação. **Não troque `M67` pelo nome do aglomerado novo nos arquivos existentes.** Confira treino/validação/teste nos registros correspondentes de `Logs/`: o nome isolado não define a divisão. AM se destina à avaliação de aglomerados separados; BM usa todos os dados.

Há variações históricas, inclusive pesos sem `_gap` explícito e bancos Optuna antigos. Ausência de sufixo não comprova GAP ou ausência dele: confira logs, dimensões e configuração original. O código atual salva outros padrões, como `_split_...`; não renomeie arquivos antigos para simular uma execução atual. `Previsto/` e JPGs de curvas de aprendizado são resultados, não modelos.

## 4. Prepare a amostra e o arquivo de coordenadas

Selecione estrelas e determine o centro cientificamente. Registre versão Gaia, consulta/filtros, seleção de membros/campo, cortes de magnitude, tamanho do campo e época. Não forneça RA/Dec absolutos como coordenadas centradas: o conversor copia as duas colunas diretamente, e a análise radial assume centro `(0, 0)` e multiplica distâncias por 60.

Use deslocamentos angulares centrados **em graus**: em campos pequenos, aproximadamente `X = ΔRA × cos(Dec0)` e `Y = ΔDec`. Para campos amplos, regiões polares ou outra projeção, defina e valide uma transformação de plano tangente compatível com o treinamento. Preserve cobertura para avaliar fundo e truncamento; não altere silenciosamente a seleção/contaminação.

```text
BATEIA-data/
└── MyCluster/
    ├── Posicoes/
    │   └── MyCluster_coords.dat
    ├── CNN/                      # generated
    └── Features/                 # generated
```

O `.dat` tem uma linha de cabeçalho e seis colunas separadas por espaços: `ID`, `RAlin(J2016)`, `DEClin(J2016)`, `Gmag`, `BPmag`, `RPmag`. Apesar dos nomes, as coordenadas precisam conter os deslocamentos preparados. Não use minutos de arco, que introduziriam outro fator 60. Mantenha o nome exato do aglomerado, inclusive maiúsculas/minúsculas. O exemplo de campo pequeno espera seu CSV e centro:

```python
# Run from the BATEIA-1 repository root, in the activated environment.
from pathlib import Path
import os
import numpy as np
import pandas as pd

root = Path("BATEIA-data").resolve()
cluster = "MyCluster"
os.environ["BATEIA_ROOT"] = str(root)  # BEFORE importing bateia

# Your selected stellar sample, in degrees and Gaia magnitudes:
df = pd.read_csv("my_selected_stars.csv")
# Required columns: source_id, ra, dec, phot_g_mean_mag,
#                   phot_bp_mean_mag, phot_rp_mean_mag
# Supply your scientifically determined cluster centre (degrees).
ra0 = float(input("Cluster centre RA [deg]: "))
dec0 = float(input("Cluster centre Dec [deg]: "))
delta_ra = (df["ra"].to_numpy() - ra0 + 180.0) % 360.0 - 180.0
x = delta_ra * np.cos(np.deg2rad(dec0))
y = df["dec"].to_numpy() - dec0
prepared = pd.DataFrame({
    "ID": df["source_id"].astype(str),
    "RAlin(J2016)": x,
    "DEClin(J2016)": y,
    "Gmag": df["phot_g_mean_mag"],
    "BPmag": df["phot_bp_mean_mag"],
    "RPmag": df["phot_rp_mean_mag"],
})
assert len(prepared) > 0
assert np.isfinite(prepared.iloc[:, 1:].to_numpy(dtype=float)).all()
folder = root / cluster / "Posicoes"
folder.mkdir(parents=True, exist_ok=True)
prepared.to_csv(folder / f"{cluster}_coords.dat", sep=" ", index=False)
from bateia.tools.conversor_dados import processar_um_aglomerado
assert processar_um_aglomerado(cluster), "Coordinate/image conversion failed"
assert (root / cluster / "CNN" / f"{cluster}_1.jpg").is_file()
```

Se já possui deslocamentos centrados em graus, use-os sem recentrar. O exemplo exige fotometria finita como verificação; documente filtros para valores ausentes. Ele não consulta Gaia nem seleciona membros. A conversão gera `Posicoes/MyCluster_coords_processed.csv` com `X,Y` e separador `;`, `CNN/MyCluster_1.jpg` (entrada limpa) e `CNN/MyCluster_stars.jpg` (diagnóstico com eixos). Confira centro, escala, orientação e concentração. Não use mapas de calor, figuras com rótulos ou visões dos sintéticos como entrada do modelo.

## 5. Extraia características na representação correta

Preserve o pré-processamento da execução original. A extração real atual usa normalização ImageNet e redimensionamento 224 × 224; os pesos podem exigir download inicial. A função abaixo processa **todas as extratoras registradas**, consumindo tempo/memória mesmo usando apenas DenseNet121 depois. Essa função direta não exige alvos, geração sintética, autenticação Google ou configuração de ajuste por aglomerado.

```python
# Continue in the SAME session as the preparation example.
from bateia.heatmaps_features import executar_analise_reais

executar_analise_reais([cluster], suffix=".pt", apply_gap=True)
feature_path = root / cluster / "Features" / "DENSENET121" / f"features_real_{cluster}.pt"
assert feature_path.is_file(), "Inspect extraction errors before continuing"
```

Para um experimento confirmado sem GAP, use `executar_analise_reais([cluster], suffix="_semgap.pt", apply_gap=False)` e o arquivo real terminado em `_semgap.pt`. DenseNet121 com GAP tem 1.024 componentes; a representação espacial tem 1.024 × 7 × 7 antes de achatar. Dimensões iguais são necessárias, mas também precisam corresponder pré-processamento, camada, versão dos pesos e ordem das características. Valide primeiro com um aglomerado de referência do acervo.

## 6. Escolha um conjunto coerente de modelo e scalers

O exemplo usa SVR + DenseNet121 + GAP de um experimento AM, sem alegar que é o melhor modelo. Os nomes reais usam `M67` e abrangem os quatro parâmetros. Só altere família/extratora/experimento depois de verificar conjuntos completos. Não reajuste scalers baixados em um aglomerado novo, não misture parâmetros e não troque GAP silenciosamente.

## 7. Faça predições e salve uma tabela separada

```python
# Continue in the SAME session; select the trusted downloaded archive.
import joblib
import torch

models_root = Path(input("Path to the extracted Redes Neurais directory: ")).expanduser().resolve()
branch = models_root / "SVR" / "AMSVR"
extractor = "DENSENET121"
experiment = "M67"  # historical experiment identifier, NOT your cluster name

features = torch.load(feature_path, map_location="cpu", weights_only=True)
X = features.detach().cpu().numpy().reshape(1, -1)
assert np.isfinite(X).all()

rows = []
for parameter in ["f0", "rc", "rt", "fbg"]:
    stem = f"{parameter}_densenet121_svr_{experiment}_gap"
    model_path = branch / "Modelos" / extractor / f"{stem}_svr_model.joblib"
    sx_path = branch / "Scaler" / extractor / f"scaler_x_{stem}.pkl"
    sy_path = branch / "Scaler" / extractor / f"scaler_y_{stem}.pkl"
    for path in [model_path, sx_path, sy_path]:
        assert path.is_file(), f"Missing artifact: {path}"
    model = joblib.load(model_path)
    sx = joblib.load(sx_path)
    sy = joblib.load(sy_path)
    assert X.shape[1] == sx.n_features_in_, "Extractor/GAP/scaler mismatch"
    assert model.n_features_in_ == X.shape[1]
    outside = None
    if hasattr(sx, "data_min_") and hasattr(sx, "data_max_"):
        outside = bool(((X[0] < sx.data_min_) | (X[0] > sx.data_max_)).any())
    X_scaled = sx.transform(X)  # never fit the downloaded scalers again
    raw = float(sy.inverse_transform(model.predict(X_scaled).reshape(-1, 1))[0, 0])
    assert np.isfinite(raw)
    rows.append({
        "Cluster": cluster, "Parameter": parameter,
        "Raw_prediction": raw,
        "BATEIA_abs_prediction": abs(raw),  # matches current inference convention
        "Feature_outside_scaler_training_range": outside,
        "Model_file": str(model_path), "Scaler_x": str(sx_path), "Scaler_y": str(sy_path),
    })
result_dir = root / cluster / "Predictions"
result_dir.mkdir(parents=True, exist_ok=True)
output = result_dir / "svr_densenet121_M67_gap.csv"
pd.DataFrame(rows).to_csv(output, index=False)
print(pd.DataFrame(rows)[["Parameter", "Raw_prediction", "BATEIA_abs_prediction"]])
print(output)
```

O exemplo grava `MyCluster/Predictions/svr_densenet121_M67_gap.csv`. Esse é seu próprio caminho de saída, não um novo comando do menu nem o `Previsto/` histórico. A inferência do código aplica valor absoluto após a transformação inversa; o exemplo preserva **o valor bruto e essa convenção**. Predições brutas negativas exigem investigação. O aviso de características fora da faixa do scaler sinaliza extrapolação; sua ausência não valida a predição. Um campo vazio indica que o scaler não expõe limites por característica e essa verificação não estava disponível.

## 8. Valide a interpretação científica

Confirme as unidades nos rótulos originais: neste fluxo, `Rc`/`Rt` em minutos de arco e `f0`/`fbg` em estrelas por minuto de arco quadrado. Os tokens `rc`/`rt` dos nomes representam `Rc`/`Rt`. Verifique valores finitos, positividade, ordenação dos raios, sensibilidade à seleção/centro, consistência com referência e perfil radial independente. São verificações, não garantias de validade física.

Alvos desconhecidos não impedem a inferência, mas impedem medir sua acurácia ou erro percentual contra verdade desconhecida. Não reporte RMSE nessas condições. Fatores `inct_*` e faixas da literatura não são incertezas preditivas calibradas para novos aglomerados. Score-CAM/Ablation-CAM mostram respostas das extratoras, não intervalos de confiança dos parâmetros.

## 9. Adapte para regressores de deep learning

MLP fica em `MLPerceptron/AMMLP/` ou `BMMLP/`; LSTM, GRU e 1D-CNN têm suas ramificações. O acervo contém, por exemplo, `f0_densenet121_mlp_M67_gap_mlp_model.pth`. Localize os dois scalers correspondentes; não reaproveite os de SVR.

O salvador atual grava pesos `state_dict`, não uma rede autocontida. Reconstrua a rede original com `bateia/models/architectures.py` e a configuração histórica; as fábricas atuais estão nos módulos `*_model.py`. Logs podem omitir argumentos da arquitetura: recupere-os da execução/código original ou de formas compatíveis dos pesos, sem presumir valores atuais do espaço de busca. O bloco seguinte é um **padrão de adaptação**, não um exemplo completo executável:

```python
# Pattern only: first reconstruct the EXACT original architecture.
# model = the matching torch.nn.Module with its original constructor arguments
# X_scaled = matching_scaler_x.transform(X)
# shape = the exact training layout; None for flat MLP input
state = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
tensor = torch.as_tensor(X_scaled, dtype=torch.float32)
if shape is not None:
    tensor = tensor.reshape(1, *shape)
with torch.no_grad():
    prediction_scaled = model(tensor).cpu().numpy().reshape(-1, 1)
prediction = matching_scaler_y.inverse_transform(prediction_scaled)
```

MLP recebe entrada achatada. Os sequenciais atuais usam `(batch, 1, C)` com GAP ou `(batch, 49, C)` sem GAP, depois de transformar os dados achatados com o scaler original. Confira a ordem de remodelagem no treinador original: a mesma dimensão com outra ordem muda a entrada. Use `strict=True`; resolva incompatibilidades com a arquitetura original, sem escondê-las com `strict=False`. `weights_only=True` restringe desserialização, mas não reconstrói arquitetura nem garante compatibilidade.

### Exemplo concreto MLP (DenseNet121, GAP, f0)

O checkpoint inspecionado contém `model.0.weight`, `model.3.weight`, …, `model.18.weight`: sete camadas lineares, incluindo a saída. A profundidade completa não é estabelecida pelo log impresso. O exemplo deriva larguras dos pesos, verifica a indexação linear/ReLU/dropout e carrega estritamente no `MLPRegressor` atual. Não foi executado aqui. Se falhar uma asserção ou carregamento estrito, recupere a arquitetura/versão original, sem relaxar verificações. Dropout é desativado apenas para avaliação; isso não reproduz a configuração de treinamento. Para outro parâmetro, use seu conjunto completo de modelo/scalers.

```python
# Continue after the SVR example: X, models_root, extractor are already defined.
# This specifically reconstructs the inspected flat MLP layout, not LSTM/GRU/CNN.
from bateia.models.architectures import MLPRegressor

mlp_branch = models_root / "MLPerceptron" / "AMMLP"
stem = "f0_densenet121_mlp_M67_gap"
model_path = mlp_branch / "Modelos" / extractor / f"{stem}_mlp_model.pth"
sx = joblib.load(mlp_branch / "Scaler" / extractor / f"scaler_x_{stem}.pkl")
sy = joblib.load(mlp_branch / "Scaler" / extractor / f"scaler_y_{stem}.pkl")
state = torch.load(model_path, map_location="cpu", weights_only=True)
keys = sorted(
    [k for k in state if k.startswith("model.") and k.endswith(".weight")],
    key=lambda k: int(k.split(".")[1]),
)
assert len(keys) >= 2, "Not the inspected MLP state_dict layout"
assert [int(k.split(".")[1]) for k in keys] == list(range(0, len(keys) * 3, 3))
assert all(state[k].ndim == 2 for k in keys)
input_size = int(state[keys[0]].shape[1])
hidden_sizes = [int(state[k].shape[0]) for k in keys[:-1]]
assert int(state[keys[-1]].shape[0]) == 1
assert input_size == X.shape[1] == sx.n_features_in_
# Dropout has no effect in eval mode; original training dropout is not recovered.
network = MLPRegressor(input_size, hidden_sizes, output_size=1, dropout=0.0)
network.load_state_dict(state, strict=True)
network.eval()
with torch.no_grad():
    y_scaled = network(torch.as_tensor(sx.transform(X), dtype=torch.float32)).numpy()
raw_f0 = float(sy.inverse_transform(y_scaled.reshape(-1, 1))[0, 0])
assert np.isfinite(raw_f0)
print({"Parameter": "f0", "Raw_prediction": raw_f0, "BATEIA_abs_prediction": abs(raw_f0)})
```

Para LSTM/GRU/1D-CNN, recupere camadas, larguras, profundidade recorrente e ordem de remodelagem originais; a reconstrução MLP achatada não se aplica. Carregadores completos dessas famílias ainda exigem adaptação por caso.

## 10. Opcional: ajuste perfis, gere alvos e retreine

Este caminho difere da aplicação de um modelo salvo. Adicione o aglomerado em `PARAMETROS_AJUSTE`, em `bateia/config_data/clusters.py`: a análise real atual ignora aglomerados sem entrada. Revise `AjusteCentral`, `AjusteRestante`, `AjusteFundoBG`, `t` e `t_ideal`, que controlam correções empíricas e bins, não coordenadas ou pesos aprendidos. Derive/valide valores para seus dados; copiar M67 não calibra cientificamente. Fatores `inct_*` afetam comparações com a literatura e não são posteriores de novos alvos.

1. Prepare coordenadas/imagens e adicione metadados justificados antes de iniciar outro interpretador.
2. Para comparação/estratificação, prepare `cluster_params.csv` com `Cluster,f0,err_f0,Rc,err_Rc,Rt,err_Rt,fbg,err_fbg`, nas unidades e convenções corretas. A função `salvar_parametros_literatura()` do conversor preenche `Target/` localmente: chame-a diretamente após definir `BATEIA_ROOT`, sem abrir o submenu Drive. Não invente referências desconhecidas. Algumas comparações legadas diferem em maiúsculas/prefixos: confira nomes gerados como `mat_Rc...` e os esperados na plataforma.
3. Execute a análise real em **Run Analysis and AI Pipeline → Run Data Analysis Stages**. Confira perfis, resíduos e `Target/best_division_info_<Cluster>.csv`. A análise sintética usa `num_bins = t_ideal + 4`; defina `t_ideal` com base no número de bins validado, sem confundi-lo com índice de topologia.
4. Gere/analise sintéticos e extraia características correspondentes. Revise imagens e `Sinteticos/Resultados/<Cluster>_synthetic_parameters.csv` (separador `;`, vínculo por `Synthetic_ID`). O padrão atual é de até 1.000 sintéticos por aglomerado.
5. Monte uma amostra rotulada. Separe aglomerados reais e todos os seus sintéticos entre treino, validação e teste. O AM manual exige pelo menos 20 aglomerados (15 treino, 4 validação, restantes teste); o estratificado exige metadados completos e grupos suficientes. BM não avalia aglomerados inéditos.
6. Selecione modelos/GAP em **Run AI Training**. Revise `n_trials` em `bateia_exec.py` (atualmente 1), épocas/early stopping em `config.py` e espaços em `config_data/hyperparams.py`. Os padrões atuais não reproduzem automaticamente as configurações publicadas.
7. Salve modelos, scalers, divisões e configurações numa execução separada. Não treine por cima do acervo nem use **Reset Workspace** na raiz que contém artefatos a preservar.

## 11. Falhas frequentes

| Problema | Verifique |
| --- | --- |
| Aglomerado ignorado na análise real | Nome exato e entrada `PARAMETROS_AJUSTE` |
| Escala/densidade incorreta | Graus centrados, cobertura do campo e unidades dos rótulos |
| Características reais ausentes | Imagem `_1.jpg` limpa, erros de extração, pasta e sufixo |
| Dimensões incompatíveis | Parâmetro, experimento, GAP, camada e arquitetura |
| Modelo antigo não carrega | Dependências de treinamento e arquitetura original; não altere artefatos silenciosamente |
| Número plausível sem validade científica | Mudança de domínio, seleção, perfil independente e referência |

## 12. Registre a procedência

Registre o [DOI do dataset](https://doi.org/10.5281/zenodo.24087976) e o [DOI dos modelos](https://doi.org/10.5281/zenodo.24094057), arquivos/versões baixados, commit, caminhos/checksums, ambiente, centro, convenção de coordenadas, filtros, características, divisão e predições brutas/finais. Os exemplos continuam sendo fluxos conferidos no código, não resultados de novos aglomerados validados independentemente. Consulte [uso](USAGE.pt-BR.md), [modelos](MODELS_GUIDE.pt-BR.md), [arquitetura](ARQUITETURA.md) e [dependências](DEPENDENCIES.pt-BR.md).

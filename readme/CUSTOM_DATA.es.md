[English](CUSTOM_DATA.md) · [Português (Brasil)](CUSTOM_DATA.pt-BR.md) · [Español](CUSTOM_DATA.es.md)

[← README](README.es.md)

# Análisis de datos propios de cúmulos estelares

Esta guía presenta un flujo avanzado basado en el código actual y el archivo inspeccionado `modelos-ia-bateia-mestrado`. Incluye inferencia SVR y orientaciones para redes profundas y entrenamiento propio. Los bloques Python usan funciones existentes y **no constituyen una nueva API de alto nivel**. Se comprobaron contra el código y los nombres de archivos, pero no se ejecutaron con los modelos en este entorno. No se ha validado compatibilidad de ejecución ni precisión científica.

## 1. Elija su tarea científica

| Tarea | Entradas necesarias | ¿Necesita parámetros objetivo conocidos? |
| --- | --- | --- |
| Aplicar un modelo guardado | Coordenadas preparadas; extractor, regresor y dos scalers correspondientes | No, en la inferencia directa siguiente |
| Entrenar regresores nuevos | Muestra etiquetada, características/etiquetas sintéticas, división y configuración | Sí, con origen científico definido |
| Reproducir la publicación | Datos, divisiones, configuración, versión del código y artefactos originales | Etiquetas y referencias originales |

Use cúmulos abiertos no embebidos compatibles con el dominio de entrenamiento y perfiles de King de un componente. Una predicción numéricamente válida no demuestra generalización a otras poblaciones, selecciones de miembros o contaminación de campo.

## 2. Instale y localice los archivos

**Recursos publicados:** [Dataset BATEIA-1](https://doi.org/10.5281/zenodo.24087976) · [Modelos entrenados y scalers](https://doi.org/10.5281/zenodo.24094057). Para el ejemplo de inferencia directa, descargue modelos/scalers y prepare sus coordenadas; no necesita todo el dataset de entrenamiento para predecir un nuevo cúmulo. Use el dataset al reproducir o reentrenar los experimentos.

Descargue y extraiga manualmente los archivos necesarios, conservando nombres y estructura interna. El árbol siguiente describe la colección de investigación inspeccionada; localice `Redes Neurais` si la extracción añade otra carpeta externa. No deduzca nombres de descarga a partir del DOI ni renombre artefactos históricos para imitar salidas actuales.

Siga la [instalación](USAGE.es.md#installation), active `bateia` y ejecute desde la raíz del repositorio. Use la misma sesión Python o kernel en los pasos 4–7. Defina `BATEIA_ROOT` antes de importar `bateia`; reinicie si ya lo importó con otra raíz. El uso en notebook exige esta preparación manual y no implica una API empaquetada.

Separe los modelos de sus datos observacionales. Los ejemplos requieren la **subcarpeta `Redes Neurais`**, no la carpeta externa de descarga. Los `.joblib`/`.pkl` antiguos necesitan versiones compatibles de scikit-learn/joblib y pueden ejecutar código serializado; use los artefactos confiables del autor. La receta Conda actual no demuestra compatibilidad con el archivo histórico.

## 3. Entienda la estructura de los modelos

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

Los nombres de carpetas son literales. `MLPerceptron` contiene experimentos MLP; las ramas distinguen AM y BM. Los extractores incluyen `DENSENET121`, `DENSENET161`, `DENSENET201`, `RESNET50/101/152` y `VGG13/16/19`.

Ejemplo real de tres archivos correspondientes:

```text
SVR/AMSVR/Modelos/DENSENET121/f0_densenet121_svr_M67_gap_svr_model.joblib
SVR/AMSVR/Scaler/DENSENET121/scaler_x_f0_densenet121_svr_M67_gap.pkl
SVR/AMSVR/Scaler/DENSENET121/scaler_y_f0_densenet121_svr_M67_gap.pkl
```

`f0` es el parámetro, `densenet121` el extractor, `svr` el regresor, `M67` el experimento histórico y `gap` la representación. **No sustituya `M67` por el nombre del nuevo cúmulo en los archivos existentes.** Compruebe entrenamiento/validación/prueba en los registros de `Logs/`: el nombre no establece por sí solo la división. AM se orienta a evaluar cúmulos separados; BM usa todos los datos.

Existen variaciones históricas, incluidos pesos sin `_gap` explícito y bases Optuna antiguas. La ausencia de sufijo no demuestra GAP ni ausencia de él: compruebe registros, dimensiones y configuración original. El código actual guarda otros patrones, como `_split_...`; no renombre archivos antiguos para simular una ejecución actual. `Previsto/` y JPG de curvas son resultados, no modelos.

## 4. Prepare la muestra y el archivo de coordenadas

Seleccione estrellas y determine el centro científicamente. Registre versión Gaia, consulta/filtros, selección de miembros/campo, cortes de magnitud, tamaño del campo y época. No proporcione RA/Dec absolutas como coordenadas centradas: el conversor copia ambas columnas y el análisis radial supone centro `(0, 0)` y multiplica distancias por 60.

Use desplazamientos centrados **en grados**: para campos pequeños, aproximadamente `X = ΔRA × cos(Dec0)` e `Y = ΔDec`. En campos amplios, regiones polares u otra proyección, defina y valide una transformación de plano tangente compatible con el entrenamiento. Mantenga cobertura para fondo y truncamiento; no cambie silenciosamente selección/contaminación.

```text
BATEIA-data/
└── MyCluster/
    ├── Posicoes/
    │   └── MyCluster_coords.dat
    ├── CNN/                      # generated
    └── Features/                 # generated
```

El `.dat` tiene una cabecera y seis columnas separadas por espacios: `ID`, `RAlin(J2016)`, `DEClin(J2016)`, `Gmag`, `BPmag`, `RPmag`. Pese a los nombres, las coordenadas deben ser desplazamientos preparados. No use minutos de arco, que introducirían otro factor 60. Conserve el nombre exacto del cúmulo y sus mayúsculas. Este ejemplo de campo pequeño espera su CSV y centro:

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

Si ya dispone de desplazamientos centrados en grados, úselos sin volver a centrar. El ejemplo exige fotometría finita como comprobación; documente filtros de valores ausentes. No consulta Gaia ni selecciona miembros. La conversión genera `Posicoes/MyCluster_coords_processed.csv` (`X,Y`, separador `;`), `CNN/MyCluster_1.jpg` (entrada limpia) y `CNN/MyCluster_stars.jpg` (diagnóstico con ejes). Compruebe centro, escala, orientación y concentración. No use mapas de calor, figuras rotuladas ni vistas de sintéticos como entrada del modelo.

## 5. Extraiga características con la representación correcta

Mantenga el preprocesamiento de la ejecución original. La extracción real actual usa normalización ImageNet y tamaño 224 × 224; los pesos pueden requerir una descarga inicial. La función siguiente procesa **todos los extractores registrados**, consumiendo tiempo/memoria aunque después use solo DenseNet121. Esta función directa no exige etiquetas, generación sintética, autenticación Google ni ajustes por cúmulo.

```python
# Continue in the SAME session as the preparation example.
from bateia.heatmaps_features import executar_analise_reais

executar_analise_reais([cluster], suffix=".pt", apply_gap=True)
feature_path = root / cluster / "Features" / "DENSENET121" / f"features_real_{cluster}.pt"
assert feature_path.is_file(), "Inspect extraction errors before continuing"
```

Para un experimento confirmado sin GAP, use `executar_analise_reais([cluster], suffix="_semgap.pt", apply_gap=False)` y el archivo real terminado en `_semgap.pt`. DenseNet121 GAP tiene 1.024 componentes; la representación espacial tiene 1.024 × 7 × 7 antes de aplanar. Dimensiones iguales son necesarias, pero también deben coincidir preprocesamiento, capa, versión de pesos y orden de características. Valide primero con un cúmulo de referencia del archivo.

## 6. Seleccione modelo y scalers coherentes

El ejemplo usa SVR + DenseNet121 + GAP de un experimento AM, sin afirmar que sea el mejor modelo. Los nombres reales usan `M67` y abarcan cuatro parámetros. Cambie familia/extractor/experimento solo después de verificar conjuntos completos. No reajuste scalers descargados a un cúmulo nuevo, no mezcle parámetros ni cambie GAP silenciosamente.

## 7. Prediga y guarde una tabla separada

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

El ejemplo guarda `MyCluster/Predictions/svr_densenet121_M67_gap.csv`: es su propia salida, no un comando nuevo ni el `Previsto/` histórico. La inferencia del código aplica valor absoluto tras la transformación inversa; el ejemplo conserva **el valor bruto y esa convención**. Los valores brutos negativos exigen investigación. El aviso de características fuera del intervalo del scaler señala extrapolación; su ausencia no valida la predicción. Un campo vacío indica que el scaler no expone límites por característica y esta comprobación no estaba disponible.

## 8. Valide la interpretación científica

Confirme unidades en las etiquetas originales: `Rc`/`Rt` en minutos de arco y `f0`/`fbg` en estrellas por minuto de arco cuadrado. Los tokens `rc`/`rt` representan `Rc`/`Rt`. Compruebe valores finitos, positividad, orden de radios, sensibilidad a selección/centro, referencia y perfil radial independiente. Son comprobaciones, no garantías físicas.

No necesita conocer los parámetros para inferir, pero sí para medir precisión o errores porcentuales. No reporte RMSE respecto a una verdad desconocida. Los factores `inct_*` y bandas bibliográficas no son incertidumbres predictivas calibradas para nuevos cúmulos. Score-CAM/Ablation-CAM muestran respuestas del extractor, no intervalos de confianza de parámetros.

## 9. Adapte a regresores de deep learning

MLP está en `MLPerceptron/AMMLP/` o `BMMLP/`; LSTM, GRU y 1D-CNN tienen sus ramas. Existe, por ejemplo, `f0_densenet121_mlp_M67_gap_mlp_model.pth`. Localice sus dos scalers; no reutilice los de SVR.

El guardado actual escribe pesos `state_dict`, no una red autocontenida. Reconstruya la arquitectura original con `bateia/models/architectures.py` y la configuración histórica; las fábricas actuales están en `*_model.py`. Los registros pueden omitir argumentos: recupérelos de la ejecución/código original o formas compatibles de pesos, sin suponer valores actuales del espacio de búsqueda. El bloque es un **patrón de adaptación**, no un ejemplo ejecutable completo:

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

MLP usa entradas aplanadas. Los modelos secuenciales actuales usan `(batch, 1, C)` con GAP o `(batch, 49, C)` sin GAP, después de transformar las características aplanadas con el scaler original. Compruebe el orden de remodelado en el entrenador original: otra ordenación altera la entrada aunque coincidan dimensiones. Use `strict=True`; resuelva incompatibilidades con la arquitectura original, sin ocultarlas con `strict=False`. `weights_only=True` restringe deserialización, pero no reconstruye arquitectura ni garantiza compatibilidad.

### Ejemplo concreto MLP (DenseNet121, GAP, f0)

El checkpoint inspeccionado contiene `model.0.weight`, `model.3.weight`, …, `model.18.weight`: siete capas lineales, incluida la salida. El registro impreso no establece su profundidad completa. El ejemplo deriva anchuras de los pesos, comprueba la indexación lineal/ReLU/dropout y carga estrictamente el `MLPRegressor` actual. No se ejecutó aquí. Si falla una aserción o carga estricta, recupere arquitectura/versión original sin relajar comprobaciones. Dropout se desactiva solo para evaluación; esto no reproduce la configuración de entrenamiento. Para otro parámetro, use su conjunto completo de modelo/scalers.

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

Para LSTM/GRU/1D-CNN, recupere capas, anchuras, profundidad recurrente y orden de remodelado originales; la reconstrucción MLP aplanada no se aplica. Los cargadores completos de estas familias aún requieren adaptación específica.

## 10. Opcional: ajuste perfiles, genere etiquetas y reentrene

Esta ruta difiere de aplicar modelos guardados. Añada el cúmulo a `PARAMETROS_AJUSTE` en `bateia/config_data/clusters.py`: el análisis real omite cúmulos sin entrada. Revise `AjusteCentral`, `AjusteRestante`, `AjusteFundoBG`, `t` y `t_ideal`, que controlan correcciones empíricas/bins, no coordenadas ni pesos aprendidos. Derive/valide valores propios; copiar M67 no es calibración científica. Los factores `inct_*` afectan comparaciones, no son posteriores de nuevos objetivos.

1. Prepare coordenadas/imágenes y añada metadatos justificados antes de iniciar otro intérprete.
2. Para comparación/estratificación, prepare `cluster_params.csv` con `Cluster,f0,err_f0,Rc,err_Rc,Rt,err_Rt,fbg,err_fbg`, unidades y convenciones correctas. La función del conversor `salvar_parametros_literatura()` rellena `Target/` localmente; invóquela directamente tras definir `BATEIA_ROOT`, sin abrir el submenú Drive. No invente referencias desconocidas. Algunas comparaciones antiguas difieren en mayúsculas/prefijos: compruebe nombres como `mat_Rc...` y los esperados en su plataforma.
3. Ejecute análisis real en **Run Analysis and AI Pipeline → Run Data Analysis Stages**. Inspeccione perfiles, residuos y `Target/best_division_info_<Cluster>.csv`. El análisis sintético usa `num_bins = t_ideal + 4`; defina `t_ideal` según el número de bins validado, sin confundirlo con índice de topología.
4. Genere/analice sintéticos y extraiga características correspondientes. Revise imágenes y `Sinteticos/Resultados/<Cluster>_synthetic_parameters.csv` (separador `;`, enlace por `Synthetic_ID`). El valor predeterminado actual es hasta 1.000 por cúmulo.
5. Construya una muestra etiquetada. Separe cúmulos reales y todos sus sintéticos entre entrenamiento, validación y prueba. AM manual exige al menos 20 cúmulos (15 entrenamiento, 4 validación, restantes prueba); el estratificado exige metadatos completos y grupos suficientes. BM no evalúa cúmulos no vistos.
6. Seleccione modelos/GAP en **Run AI Training**. Revise `n_trials` en `bateia_exec.py` (actualmente 1), épocas/early stopping en `config.py` y espacios en `config_data/hyperparams.py`. Los valores actuales no reproducen automáticamente los experimentos publicados.
7. Guarde modelos, scalers, divisiones y configuración en una ejecución separada. No entrene sobre el archivo descargado ni use **Reset Workspace** en una raíz con artefactos que deba preservar.

## 11. Fallos frecuentes

| Problema | Compruebe |
| --- | --- |
| Cúmulo omitido en análisis real | Nombre exacto y entrada `PARAMETROS_AJUSTE` |
| Escala/densidad incorrecta | Grados centrados, cobertura y unidades originales |
| Faltan características reales | Imagen `_1.jpg` limpia, errores, carpeta y sufijo |
| Dimensiones incompatibles | Parámetro, experimento, GAP, capa y arquitectura |
| Modelo antiguo no carga | Dependencias de entrenamiento y arquitectura original; no altere artefactos silenciosamente |
| Número plausible sin validez científica | Cambio de dominio, selección, perfil independiente y referencia |

## 12. Registre la procedencia

Registre el [DOI del dataset](https://doi.org/10.5281/zenodo.24087976) y el [DOI de modelos](https://doi.org/10.5281/zenodo.24094057), archivos/versiones descargados, commit, rutas/checksums, entorno, centro, convención de coordenadas, filtros, características, división y predicciones brutas/finales. Los ejemplos siguen siendo flujos comprobados contra el código, no resultados de nuevos cúmulos validados independientemente. Consulte [uso](USAGE.es.md), [modelos](MODELS_GUIDE.es.md), [arquitectura](ARCHITECTURE.es.md) y [dependencias](DEPENDENCIES.es.md).

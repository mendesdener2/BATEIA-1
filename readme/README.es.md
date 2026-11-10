<p align="center"><img src="../logo-bateia.png" alt="BATEIA-1 logo" width="330"></p>

<p align="center"><a href="../README.md">English</a> · <a href="../readme/README.pt-BR.md">Português (Brasil)</a> · <a href="../readme/README.es.md">Español</a></p>

# BATEIA-1

**Balanced ArTificial-intelligence Method for EstImating parameters of stAr clusters**

Estime parámetros estructurales de cúmulos estelares abiertos a partir de mapas estelares, con datos sintéticos, extractores visuales preentrenados y regresión.

[Dataset](https://doi.org/10.5281/zenodo.24087976) · [Modelos entrenados y scalers](https://doi.org/10.5281/zenodo.24094057)

<a id="overview"></a>
## Resumen científico

Los cúmulos estelares abiertos son laboratorios naturales para estudiar la evolución estelar y la estructura galáctica. BATEIA-1 estima sus parámetros estructurales combinando el perfil analítico de King con aprendizaje automático y aprendizaje profundo aplicados a mapas estelares derivados de datos astrométricos y fotométricos de Gaia DR3. La metodología estima densidad estelar central, radio del núcleo, radio de marea y densidad estelar de fondo, con el objetivo de automatizar la caracterización estructural de los cúmulos.

El estudio asociado utiliza nueve redes neuronales profundas preentrenadas para extraer características visuales, seguidas de modelos Multilayer Perceptron (MLP) y Support Vector Regression (SVR). Para M67, NGC 188 y NGC 6811, el artículo presenta diferencias respecto a la literatura de **0,7–12,8% en el radio de marea, 1,1–5,5% en el radio del núcleo y 1,1–8,7% en la densidad central**. Score-CAM y Ablation-CAM ofrecen diagnósticos complementarios de los extractores visuales. Estos resultados corresponden a los experimentos del estudio; el método conserva las hipótesis de las etiquetas de referencia basadas en perfiles de King. Consulte el [artículo](https://doi.org/10.1016/j.ascom.2026.101119) para la evaluación y el alcance.

**Palabras clave:** cúmulos estelares abiertos; astrofísica; estructura galáctica; Gaia DR3; parámetros estructurales; perfil de King; función de King; densidad estelar; aprendizaje automático; aprendizaje profundo; extracción de características; regresión; inteligencia artificial explicable (XAI); Score-CAM; Ablation-CAM.

[Resumen](#overview) · [Flujo](#pipeline) · [Ejemplos](#examples) · [Inicio rápido](#start) · [Documentación](#docs) · [Cita](#citation)

<a id="pipeline"></a>
## Pipeline

BATEIA-1 combina perfiles de King y aprendizaje automático para estimar **densidad central (`f0`), radio del núcleo (`Rc`), radio de marea (`Rt`) y densidad de fondo (`fbg`)**.

<p align="center"><img src="../bateia-diagram.png" alt="BATEIA-1 pipeline" width="1100"></p>

De las coordenadas estelares preparadas a imágenes, aumento de datos sintéticos, características visuales y regresión de parámetros. La IA Explicable proporciona un diagnóstico paralelo de los extractores.

**Extractores:** VGG13/16/19, ResNet50/101/152 y DenseNet121/161/201.  
**Regresores:** MLP, SVR, Random Forest, LSTM, GRU y 1D-CNN; características con GAP y sin GAP.

<a id="examples"></a>
## M67: datos, resultados y explicabilidad

### Mapas estelares y división radial

<table>
<tr><th align="center">Mapa estelar</th><th align="center">Vista de sintéticos</th><th align="center">División espacial: 16 anillos</th></tr>
<tr><td align="center"><a href="../readme/assets/M67_stars.jpg"><img src="../readme/assets/M67_stars.jpg" alt="Mapa estelar" width="230"></a></td><td align="center"><a href="../readme/assets/M67_synthetics_overview.jpg"><img src="../readme/assets/M67_synthetics_overview.jpg" alt="Vista de sintéticos" width="230"></a></td><td align="center"><a href="../readme/assets/M67_spatial_dist_best.jpg"><img src="../readme/assets/M67_spatial_dist_best.jpg" alt="División espacial: 16 anillos" width="230"></a></td></tr>
<tr><td valign="top">Posiciones estelares preparadas de M67, centradas en el cúmulo. Los ejes muestran desplazamientos angulares en minutos de arco; la concentración central y el campo circundante definen la distribución usada para construir la imagen de entrada.</td><td valign="top">Superposición de distribuciones espaciales sintéticas (magenta) y posiciones reales (negro). Muestra la variación espacial introducida por el aumento de datos; no es una observación adicional ni un mapa de atribución de características.</td><td valign="top">El mismo campo dividido en 16 anillos radiales concéntricos, delimitados por líneas azules discontinuas. Los recuentos por área anular construyen el perfil de densidad superficial; los círculos son límites de bins, no distintos radios ajustados del cúmulo.</td></tr>
</table>

### Perfiles de densidad radial

<table>
<tr><th align="center">Perfil de King: BATEIA-1 y literatura</th><th align="center">Perfiles de densidad observado y sintéticos</th></tr>
<tr><td align="center"><a href="../readme/assets/king_uncertainty_M67_AM.jpg"><img src="../readme/assets/king_uncertainty_M67_AM.jpg" alt="Perfil de King: BATEIA-1 y literatura" width="340"></a></td><td align="center"><a href="../readme/assets/M67_real_and_synthetic_rdp.jpg"><img src="../readme/assets/M67_real_and_synthetic_rdp.jpg" alt="Perfiles de densidad observado y sintéticos" width="340"></a></td></tr>
<tr><td valign="top">Densidad superficial de M67 frente al radio: puntos de datos (negro), curva de King de BATEIA-1 (azul discontinuo) y banda de incertidumbre bibliográfica de Angelo et al. (gris). Es un ejemplo individual; la banda no establece un intervalo predictivo calibrado para nuevos cúmulos arbitrarios.</td><td valign="top">Densidades radiales observadas (rojo) superpuestas a las de los perfiles sintéticos (gris), con 16 divisiones radiales. La dispersión muestra la variación de la muestra sintética alrededor del perfil observado en cada radio; no es un intervalo de confianza de predicciones de regresores.</td></tr>
</table>

### IA Explicable: familias de extractores

<table>
<tr><th align="center">DenseNet: Score-CAM y Ablation-CAM</th><th align="center">ResNet: Score-CAM y Ablation-CAM</th><th align="center">VGG: Score-CAM y Ablation-CAM</th></tr>
<tr><td align="center"><a href="../readme/assets/heatmap_densenet_m67.jpg"><img src="../readme/assets/heatmap_densenet_m67.jpg" alt="DenseNet: Score-CAM y Ablation-CAM" width="230"></a></td><td align="center"><a href="../readme/assets/heatmap_resnet_M67.jpg"><img src="../readme/assets/heatmap_resnet_M67.jpg" alt="ResNet: Score-CAM y Ablation-CAM" width="230"></a></td><td align="center"><a href="../readme/assets/heatmap_vgg_M67.jpg"><img src="../readme/assets/heatmap_vgg_M67.jpg" alt="VGG: Score-CAM y Ablation-CAM" width="230"></a></td></tr>
<tr><td valign="top">Paneles de entrada, Score-CAM y Ablation-CAM para DenseNet121, DenseNet161 y DenseNet201. Los mapas muestran dónde responde cada extractor en M67 y cómo varían los patrones con la profundidad y el método de visualización.</td><td valign="top">Paneles de entrada, Score-CAM y Ablation-CAM para ResNet50, ResNet101 y ResNet152. Compare ubicación y extensión espacial de respuestas entre profundidades de redes residuales y métodos, usando el mismo cúmulo como referencia visual.</td><td valign="top">Paneles de entrada, Score-CAM y Ablation-CAM para VGG13, VGG16 y VGG19. Los mapas ilustran respuestas que pueden extenderse fuera del centro y permiten inspeccionar diferencias con las familias DenseNet y ResNet.</td></tr>
</table>

En cada panel de familia, las columnas son entrada, Score-CAM y Ablation-CAM; las filas identifican la variante del extractor. Los colores representan respuestas relativas dentro de cada visualización. Son diagnósticos de extractores visuales, no intervalos de confianza, causalidad física ni explicaciones directas de parámetros de los regresores posteriores; no interprete colores como una escala cuantitativa común entre paneles.

La metodología se dirige a cúmulos abiertos no embebidos descritos por perfiles de King de un componente y conserva las hipótesis de las etiquetas de referencia. El artículo evalúa MLP y SVR; la tesis y el código abarcan regresores adicionales.

<a id="start"></a>
## Inicio rápido

### 1. Cree el entorno

Instale Conda/Miniconda y abra una terminal en la raíz del proyecto:

```sh
conda env create -f environment.yml
conda activate bateia
```

El entorno solicita Python 3.12 y añade el índice PyTorch nightly con CUDA 12.8. Las versiones no están bloqueadas; GPU exige controlador/distribución compatibles. La instalación y compatibilidad de plataformas no se probaron para esta documentación.

### 2. Prepare los datos

| Material | Identificador persistente | Finalidad |
| --- | --- | --- |
| Dataset | [10.5281/zenodo.24087976](https://doi.org/10.5281/zenodo.24087976) | Dataset de investigación de los experimentos BATEIA-1 |
| Modelos entrenados y scalers | [10.5281/zenodo.24094057](https://doi.org/10.5281/zenodo.24094057) | Regresores guardados y scalers asociados |

Descargue el [dataset](https://doi.org/10.5281/zenodo.24087976) para trabajar con la muestra publicada, o prepare coordenadas propias siguiendo la [guía de datos propios](CUSTOM_DATA.es.md). Obtenga los [modelos entrenados y scalers](https://doi.org/10.5281/zenodo.24094057) para aplicar regresores guardados; reentrenar es un flujo separado.

Coloque las carpetas de cúmulos bajo `database/`, o defina `BATEIA_ROOT` como la carpeta de datos que las contiene directamente. Mantenga los modelos aparte y conserve sus nombres y los de los scalers. Consulte [estructura y configuración de datos](USAGE.es.md#data).

La descarga de Zenodo es manual: **Setup and Data Download** sigue siendo la vía Google Drive y no descarga estos registros DOI. Esa vía exige su propio OAuth y acceso a la carpeta configurada. Los pesos ImageNet de los extractores son independientes y pueden descargarse durante el primer uso.

### 3. Abra el menú interactivo

```sh
python bateia_exec.py
```

1. **Setup and Data Download**: prepare parámetros bibliográficos y archivos de cúmulos, si utiliza Drive.
2. **Verify Local Clusters**: compruebe la detección de cúmulos preparados.
3. **Run Analysis and AI Pipeline**: procese datos reales, genere/analice sintéticos y extraiga características antes de entrenar.
4. **Run AI Training**: elija modelo, modo GAP correspondiente y división por cúmulo. Use AM para evaluar casos no vistos; BM utiliza todos los datos.
5. **Generate Results and Paper Reports**: inspeccione predicciones e informes.

Empiece con un cúmulo para inspeccionar el preprocesamiento. La división manual documentada exige al menos 20 cúmulos. **Reset Workspace elimina datos/modelos generados.** Consulte la [guía completa](USAGE.es.md) para dependencias, divisiones y salidas.

**Interfaz actual:** menú de terminal. Una API simplificada para notebooks/Kaggle y la instalación como paquete aún no están implementadas.

<a id="docs"></a>
## Documentación

- [Análisis de datos propios de cúmulos estelares](../readme/CUSTOM_DATA.es.md)
- [Instalación, datos y uso](USAGE.es.md)
- [Arquitectura](../readme/ARCHITECTURE.es.md)
- [Modelos y optimización](../readme/MODELS_GUIDE.es.md)
- [Paralelismo en GPUs](../readme/GPU_PARALLELISM.es.md)
- [Dependencias](../readme/DEPENDENCIES.es.md)

<a id="citation"></a>
## Cita

Para la metodología y los resultados experimentales, cite:

Mendes, D. C.; Santiago Júnior, V. A.; Angelo, M. S. (2026). **BATEIA-1: An Approach for determining the structural parameters of open star clusters via machine and deep learning.** *Astronomy and Computing*, 56, 101119. [DOI](https://doi.org/10.1016/j.ascom.2026.101119).

Al usar datos de investigación o artefactos entrenados, cite también el [registro del dataset](https://doi.org/10.5281/zenodo.24087976) y/o el [registro de modelos](https://doi.org/10.5281/zenodo.24094057), identificando archivos y versiones utilizados.

### Licencia

Consulte [LICENSE](../LICENSE) para el texto de GNU General Public License v3 distribuido con este repositorio. Las publicaciones y los datos de terceros tienen licencias propias.

<p align="center">
  <img src="../logo-bateia.png" alt="BATEIA logo" width="280">
</p>

<p align="center">
  <a href="../README.md">English</a> ·
  <a href="../readme/README.pt-BR.md">Português (Brasil)</a> ·
  <a href="../readme/README.es.md">Español</a>
</p>

# BATEIA

**Balanced ArTificial-intelligence Method for EstImating parameters of stAr clusters**

Software de investigación para estimar parámetros estructurales de cúmulos estelares abiertos a partir de mapas estelares, combinando perfiles de King, datos sintéticos, extractores visuales preentrenados y regresión.

[Descripción](#overview) · [Instalación](#installation) · [Datos y modelos](#data) · [Primer uso](#usage) · [Documentación](#docs) · [Referencias](#references)

> **Interfaz actual:** programa Python local e interactivo iniciado con `python bateia_exec.py`. Existen módulos internos, pero una API simplificada para notebooks/Kaggle y un flujo de instalación como paquete todavía no están implementados en esta versión. Aquí no se establece un procedimiento publicado de `pip install bateia`.

<a id="overview"></a>
## De los mapas estelares a los parámetros estructurales

BATEIA se dirige a cúmulos abiertos no embebidos cuya distribución espacial puede describirse mediante un perfil de King de un solo componente. Las predicciones emulan ajustes de referencia y conservan las hipótesis y limitaciones de esas etiquetas.

| Parámetro en el código | Magnitud física |
| --- | --- |
| `f0` | Densidad estelar central |
| `Rc` | Radio del núcleo |
| `Rt` | Radio de marea |
| `fbg` | Densidad estelar de fondo |

El flujo convierte coordenadas preparadas en imágenes, ajusta perfiles de densidad reales, genera y analiza cúmulos sintéticos, extrae características, entrena regresores con búsqueda de hiperparámetros mediante Optuna y exporta predicciones e informes científicos. Score-CAM y Ablation-CAM proporcionan visualizaciones de diagnóstico.

**En el código inspeccionado:** extractores VGG13/16/19, ResNet50/101/152 y DenseNet121/161/201; regresores MLP, SVR, Random Forest, LSTM, GRU y 1D-CNN. Las características admiten promedio global (GAP) y un modo sin GAP (`_semgap.pt`). El artículo estudia nueve extractores con MLP y SVR; la tesis y el código abarcan regresores adicionales. Los experimentos publicados no validan todas las combinaciones del código actual. No se ejecutaron entrenamientos ni benchmarks para preparar este README.

> **Imagen futura · Vista del flujo**  
> Espacio reservado para un esquema: mapa estelar → muestra sintética → características → parámetros. Sustituya este bloque por imagen y leyenda cuando estén disponibles.

<a id="installation"></a>
## Instalación paso a paso

### 1. Abra la carpeta del proyecto

Obtenga el código completo y abra una terminal en la carpeta que contiene `bateia_exec.py` y `environment.yml`. Para una carpeta `bateia-1` en el directorio actual:

```sh
cd bateia-1
```

El paquete de documentación por sí solo no incluye código ejecutable. Instale Conda o Miniconda primero y use una terminal donde `conda` esté disponible. Se necesita internet para las dependencias y, posiblemente, datos y pesos preentrenados.

### 2. Cree y active el entorno

```sh
conda env create -f environment.yml
conda activate bateia
```

El archivo solicita **Python 3.12**, dependencias científicas de Python, Optuna, herramientas de la API de Google y componentes PyTorch. La sección pip añade el índice **PyTorch nightly con CUDA 12.8**, pero no fija versiones ni exige explícitamente versiones preliminares. Es una receta de entorno, no una instalación bloqueada.

Usar GPU exige un controlador NVIDIA y una distribución PyTorch compatibles. El código selecciona CPU cuando CUDA no está disponible, pero la ejecución en CPU y la compatibilidad Windows/Linux/macOS no se probaron para esta documentación. Si falla la resolución, examine el error y adapte PyTorch al equipo; aquí no se ha verificado un entorno alternativo.

### 3. Compruebe Python y las GPUs visibles

```sh
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Debe aparecer Python 3.12. `CUDA available: False` significa que este intérprete no puede usar CUDA; no identifica la causa ni valida el resto del flujo.

<a id="data"></a>
## Prepare datos y modelos

El directorio predeterminado es `database/` en la raíz del proyecto. Opcionalmente, defina `BATEIA_ROOT` **como la propia carpeta de datos**, antes de iniciar Python. Son comandos alternativos para distintas terminales; cree y rellene la carpeta elegida antes:

```powershell
# Windows PowerShell
$env:BATEIA_ROOT = "C:\BATEIA-data"
```

```sh
# Shell Linux/macOS
export BATEIA_ROOT="/path/to/BATEIA-data"
```

Sin la variable, cree y rellene `database/`. La carpeta inspeccionada no incluye ese conjunto de datos ni checkpoints de regresión entrenados.

```text
database/                         # o BATEIA_ROOT
├── cluster_params.csv
└── <Cluster>/
    ├── Posicoes/
    │   └── <Cluster>_coords.dat
    └── Target/                   # etiquetas bibliográficas generadas
```

- **Coordenadas:** una fila de cabecera y seis columnas separadas por espacios: `ID`, `RAlin(J2016)`, `DEClin(J2016)`, `Gmag`, `BPmag`, `RPmag`. El conversor omite la cabecera y copia las coordenadas a `X` e `Y`. Use datos preparados con la convención del flujo científico; la conversión no determina el centro del cúmulo ni selecciona miembros Gaia de catálogos brutos arbitrarios.
- **Parámetros bibliográficos:** `cluster_params.csv`, separado por comas, usa `Cluster`, `f0`, `err_f0`, `Rc`, `err_Rc`, `Rt`, `err_Rt`, `fbg`, `err_fbg`. Los nombres deben coincidir con rutas y metadatos en `bateia/config_data/clusters.py`. Conserve unidades y convenciones de incertidumbre de referencia.
- **Google Drive:** la opción **1** del menú principal usa OAuth en el navegador con su propio `credentials.json` local y guarda la autorización en `token.json`. Exige acceso a la carpeta configurada en `bateia/tools/gdrive_downloader.py`. No se verificaron el acceso público ni la disponibilidad de descargas. Excluya estos archivos personales del control de versiones y paquetes compartidos.
- **Archivos locales:** coloque las entradas preparadas en la estructura anterior. Los CSV de `Target/` también deben estar preparados. Su comando de generación está en el submenú de configuración, que actualmente autentica con Drive antes de abrirse; no hay asistente completamente offline implementado.
- **Pesos:** los extractores torchvision solicitan pesos ImageNet (`IMAGENET1K_V1`), que pueden descargarse a la caché PyTorch durante el primer uso. Son extractores visuales, no regresores de parámetros BATEIA. Entrene regresores mediante el menú o aporte artefactos locales compatibles; no se promete ningún checkpoint BATEIA alojado públicamente.

<a id="usage"></a>
## Su primer uso

Ejecute desde la raíz del proyecto para resolver correctamente rutas relativas de autenticación y registros:

```sh
python bateia_exec.py
```

Los nombres siguientes son los de la aplicación; los idiomas de la documentación no cambian la interfaz.

1. Elija **[1] Setup and Data Download** si utiliza Drive. Descargue/actualice parámetros bibliográficos, genere etiquetas y después descargue los archivos `.dat`. Esto exige su propia cuenta autorizada y configuración OAuth.
2. Elija **[3] Verify Local Clusters** en el menú principal. Confirme la detección de los cúmulos esperados. Se comprueba la ruta del archivo de coordenadas, no la calidad científica.
3. Elija **[2] Run Analysis and AI Pipeline → [1] Run Data Analysis Stages**. Seleccione un cúmulo preparado y confirme. Ejecute procesamiento y análisis real antes de las etapas dependientes; después genere/analice sintéticos y extraiga características. Elija GAP, sin GAP o ambos cuando se solicite. Los mapas de calor son una etapa opcional de análisis.
4. Con características y etiquetas disponibles para suficientes cúmulos, elija **Run AI Training** en el menú de ejecución. Seleccione un modelo y un modo de características correspondiente. Use **AM Methodology (Uncontaminated)** con división explícita por cúmulo para evaluar casos no vistos. La división manual exige al menos 20 cúmulos (15 de entrenamiento, 4 de validación y los restantes de prueba). La estratificada permite elegir cúmulos de prueba y divide el desarrollo en 70/30 mediante perfiles K-Means, con restricciones de tamaño y grupos. **BM Methodology (Contaminated)** usa todos los datos y no demuestra generalización a cúmulos no vistos.
5. Inspeccione las salidas; use **[5] Generate Results and Paper Reports** cuando existan las predicciones necesarias. Elija **[7] Exit** para salir.

Empiece con un cúmulo para inspeccionar el preprocesamiento, no para ejecutar la división de entrenamiento documentada. La generación sintética tiene un valor predeterminado de hasta 1.000 muestras por cúmulo. Las selecciones grandes pueden exigir bastante tiempo, RAM, VRAM y disco.

**Nota de numeración:** los mapas de calor son la etapa 6 del análisis, mientras que el entrenamiento también aparece como “Stage 6” en otras pantallas. Siga nombres y dependencias. **[4] Reset Workspace elimina datos/modelos generados** según la operación elegida.

### Dónde se guardan los resultados

Cada cúmulo recibe contenido generado en `Posicoes/`, `Target/`, `CNN/` y `Sinteticos/`, según corresponda. Las características sintéticas están en `Sinteticos/Features/<EXTRACTOR>/` (`.pt` o `_semgap.pt`) y las tablas de parámetros en `Sinteticos/Resultados/`. Las salidas de entrenamiento están bajo `Redes Neurais/`, con modelos/scalers, CSV de predicción y curvas de aprendizaje en subcarpetas de ejecución/modelo. Los archivos concretos dependen de las etapas y el modelo.

> **Imagen futura · Un ejemplo real**  
> Espacio para mapa del cúmulo, perfil de King ajustado y parámetros predichos. Leyenda: cúmulo, unidades, división de datos, modelo y procedencia de los datos.

> **Imagen futura · Explicabilidad**  
> Espacio para paneles comparables de Score-CAM y Ablation-CAM. Identifique extractor e imagen de entrada; los mapas son diagnósticos, no validación física independiente.

<a id="docs"></a>
## Documentación y solución de problemas

| Recurso | Contenido |
| --- | --- |
| [Arquitectura](../readme/ARQUITETURA.md) | Módulos y estructura del proyecto |
| [Guía de modelos](../readme/MODELS_GUIDE.md) | GAP, regresores y optimización |
| [Paralelismo en GPUs](../readme/PARALELISMO-GPUS.md) | Rutas existentes para varias GPUs |
| [Entorno](../environment.yml) | Dependencias declaradas |
| [Artículo](../artigo-bateia.pdf) · [Tesis de maestría](../dissertacao-bateia.pdf) | Metodología científica y bibliografía |

Las guías técnicas existentes están en portugués. Consulte el código cuando difieran del menú actual.

| Síntoma | Primera comprobación |
| --- | --- |
| No se encuentra `conda` | Abra una terminal con Conda disponible tras instalarlo. |
| Error de importación/dependencia | Active `bateia`; revise la instalación y el intérprete seleccionado. |
| No se encuentran cúmulos | Compruebe `BATEIA_ROOT` y `<Cluster>/Posicoes/<Cluster>_coords.dat`. |
| Falla Drive | Compruebe OAuth, permisos de la cuenta y carpeta configurada. |
| Faltan características/etiquetas | Complete generación/análisis sintético y extracción; revise modo GAP, extractor, nombres y parámetros objetivo. |
| Memoria insuficiente | Reduzca el trabajo seleccionado y revise ajustes de generación/entrenamiento. |

### Notebooks y Kaggle

Una biblioteca cómoda para notebooks es una dirección futura. No hay notebook Kaggle listo, interfaz estable de alto nivel `fit`/`predict` ni instalación soportada para notebooks en esta versión. Los módulos internos exigen adaptar rutas, dependencias, autenticación y comportamiento interactivo/multiproceso. El punto de entrada documentado sigue siendo el menú de terminal.

<a id="references"></a>
## Referencias científicas y cita

Las entradas se comprobaron en el artículo y la tesis suministrados. Cite el artículo para la metodología BATEIA-1; consulte la tesis para el estudio ampliado de modelos.

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

### Licencia y contribuciones

No se encontró archivo de licencia de software en el proyecto inspeccionado. Los avisos CC BY del artículo y CC BY-NC de la tesis se aplican a esas publicaciones y no establecen una licencia para el código. Aclare las condiciones de reutilización con los responsables. Las contribuciones deben describir procedencia de datos, configuración, entorno y validación; excluya credenciales y tokens privados.

[English](USAGE.md) · [Português (Brasil)](USAGE.pt-BR.md) · [Español](USAGE.es.md)

[← README](README.es.md)

# Instalación, datos y uso

[← README](README.es.md)

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

| Material | Identificador persistente | Finalidad |
| --- | --- | --- |
| Dataset | [10.5281/zenodo.24087976](https://doi.org/10.5281/zenodo.24087976) | Dataset de investigación de los experimentos BATEIA-1 |
| Modelos entrenados y scalers | [10.5281/zenodo.24094057](https://doi.org/10.5281/zenodo.24094057) | Regresores guardados y scalers asociados |

### Descargue y organice los recursos publicados

1. Abra el registro del dataset y descargue los archivos necesarios para su experimento. Para aplicar un modelo guardado a un cúmulo propio, siga la [guía de datos propios](CUSTOM_DATA.es.md), sin suponer que necesita todo el conjunto de entrenamiento.
2. Abra el registro de modelos y descargue el regresor, sus scalers de entrada/salida y los metadatos disponibles del experimento. Conserve rutas relativas y nombres.
3. Extraiga archivos comprimidos, si existen. Localice la carpeta que contiene directamente los cúmulos y `cluster_params.csv` cuando corresponda; defina `BATEIA_ROOT` como esa carpeta de datos. No apunte a una carpeta que solo contiene descargas comprimidas.
4. Localice `Redes Neurais` por separado, como en la [guía de datos propios](CUSTOM_DATA.es.md), y use su ruta extraída real en los ejemplos. Las carpetas externas añadidas al extraer no forman parte de los nombres de modelos.
5. Registre ambos DOI, archivos/versiones y commit del código. Separe artefactos descargados de salidas de nuevos entrenamientos.

Los archivos de Zenodo se obtienen manualmente. **Setup and Data Download** todavía autentica con Google Drive; no es un descargador de Zenodo. La estructura siguiente define lo que espera el código después de extraer.

El directorio predeterminado es `database/` en la raíz del proyecto. Opcionalmente, defina `BATEIA_ROOT` **como la propia carpeta de datos**, antes de iniciar Python. Son comandos alternativos para distintas terminales; cree y rellene la carpeta elegida antes:

```powershell
# Windows PowerShell
$env:BATEIA_ROOT = "C:\BATEIA-data"
```

```sh
# Shell Linux/macOS
export BATEIA_ROOT="/path/to/BATEIA-data"
```

Los datos de investigación y regresores entrenados se distribuyen por separado en los registros Zenodo anteriores. Sin `BATEIA_ROOT`, prepare la estructura esperada bajo `database/`.

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
- **Pesos:** los extractores torchvision solicitan pesos ImageNet (`IMAGENET1K_V1`), que pueden descargarse a la caché PyTorch durante el primer uso. Son extractores visuales, no regresores de parámetros BATEIA. Use los [modelos BATEIA publicados y sus scalers](https://doi.org/10.5281/zenodo.24094057) para inferencia con regresores guardados, o entrene nuevos desde el menú; consulte [inferencia con datos propios](CUSTOM_DATA.es.md).

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



<a id="docs"></a>
## Documentación y solución de problemas

| Recurso | Contenido |
| --- | --- |
| [Arquitectura](ARCHITECTURE.es.md) | Módulos y estructura del proyecto |
| [Guía de modelos](MODELS_GUIDE.es.md) | GAP, regresores y optimización |
| [Paralelismo en GPUs](GPU_PARALLELISM.es.md) | Rutas existentes para varias GPUs |
| [Entorno](../environment.yml) | Dependencias declaradas |
| [Artículo](https://doi.org/10.1016/j.ascom.2026.101119) · [Tesis de maestría](http://urlib.net/8JMKD2USNRW34T/4FFHT62) | Metodología científica y bibliografía |

Las guías técnicas están disponibles en inglés, portugués brasileño y español. Use el selector al inicio de cada guía. Consulte el código cuando difiera del menú documentado.

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


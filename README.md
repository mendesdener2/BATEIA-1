<p align="center">
  <img src="logo-bateia.png" alt="BATEIA logo" width="280">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="readme/README.pt-BR.md">Português (Brasil)</a> ·
  <a href="readme/README.es.md">Español</a>
</p>

# BATEIA

**Balanced ArTificial-intelligence Method for EstImating parameters of stAr clusters**

Research software for estimating the structural parameters of open star clusters from stellar maps, combining King profiles, synthetic data, pretrained visual feature extractors and regression.

[Overview](#overview) · [Installation](#installation) · [Data and models](#data) · [First run](#usage) · [Documentation](#docs) · [References](#references)

> **Current interface:** a local, interactive Python program launched with `python bateia_exec.py`. Internal modules exist, but a simplified notebook/Kaggle API and a packaged installation workflow are not implemented in this snapshot. A published `pip install bateia` workflow is not established here.

<a id="overview"></a>
## From stellar maps to structural parameters

BATEIA targets non-embedded open clusters whose spatial distributions can be described by a single-component King profile. Predictions emulate reference profile fits and retain the assumptions and limitations of those labels.

| Code parameter | Physical quantity |
| --- | --- |
| `f0` | Central stellar density |
| `Rc` | Core radius |
| `Rt` | Tidal radius |
| `fbg` | Background stellar density |

The workflow converts prepared coordinates into images, fits real density profiles, generates and analyses synthetic clusters, extracts image features, trains regressors with Optuna hyperparameter search, and exports predictions and scientific reports. Score-CAM and Ablation-CAM provide diagnostic visualizations.

**In the inspected source:** VGG13/16/19, ResNet50/101/152 and DenseNet121/161/201 extractors; MLP, SVR, Random Forest, LSTM, GRU and 1D-CNN regressors. Features support global average pooling (GAP) and a no-GAP mode (`_semgap.pt`). The article studies nine extractors with MLP and SVR; the dissertation and code cover additional regressors. Published experiments do not validate every combination in the current source. No training or benchmark was run to prepare this README.

> **Future image · Pipeline overview**  
> Reserved for an illustrated workflow: stellar map → synthetic sample → features → parameters. Replace this block with an image and caption when available.

<a id="installation"></a>
## Installation, step by step

### 1. Open the project folder

Obtain the full source and open a terminal in the folder containing `bateia_exec.py` and `environment.yml`. For a `bateia-1` folder in your current directory:

```sh
cd bateia-1
```

The documentation package alone does not include executable code. Install Conda or Miniconda first and use a terminal where `conda` is available. Internet is needed for dependencies and may be needed for data and pretrained weights.

### 2. Create and activate the environment

```sh
conda env create -f environment.yml
conda activate bateia
```

The file requests **Python 3.12**, scientific Python dependencies, Optuna, Google API tools and PyTorch components. Its pip section adds the **PyTorch nightly CUDA 12.8** index, but does not pin versions or explicitly require prereleases. This is an environment recipe, not a locked installation.

GPU use requires a compatible NVIDIA driver and PyTorch build. The code selects CPU when CUDA is unavailable, but CPU execution and Windows/Linux/macOS compatibility were not tested for this documentation. If resolution fails, inspect the error and adapt PyTorch to the target machine; no alternative environment has been verified here.

### 3. Check Python and GPU visibility

```sh
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Expect Python 3.12. `CUDA available: False` means this interpreter cannot use CUDA; it does not identify the cause or validate the remaining pipeline.

<a id="data"></a>
## Prepare data and models

Data defaults to `database/` under the project root. Optionally set `BATEIA_ROOT` **to the data directory itself**, before starting Python. These are alternative shell commands; create and populate the chosen directory first:

```powershell
# Windows PowerShell
$env:BATEIA_ROOT = "C:\BATEIA-data"
```

```sh
# Linux/macOS shell
export BATEIA_ROOT="/path/to/BATEIA-data"
```

Without the variable, create/populate `database/`. The inspected folder does not include that dataset or trained regression checkpoints.

```text
database/                         # or BATEIA_ROOT
├── cluster_params.csv
└── <Cluster>/
    ├── Posicoes/
    │   └── <Cluster>_coords.dat
    └── Target/                   # generated literature targets
```

- **Coordinates:** one header row, then six whitespace-separated columns: `ID`, `RAlin(J2016)`, `DEClin(J2016)`, `Gmag`, `BPmag`, `RPmag`. The converter skips the header and copies the two coordinates into `X` and `Y`. Use prepared data with the scientific workflow's coordinate convention; conversion does not establish the cluster centre or select Gaia members from arbitrary raw catalogues.
- **Literature parameters:** comma-separated `cluster_params.csv` uses `Cluster`, `f0`, `err_f0`, `Rc`, `err_Rc`, `Rt`, `err_Rt`, `fbg`, `err_fbg`. Names must match paths and metadata in `bateia/config_data/clusters.py`. Preserve reference units and uncertainty conventions.
- **Google Drive:** main-menu option **1** uses browser OAuth with your own local `credentials.json` and caches authorization in `token.json`. Access to the folder configured in `bateia/tools/gdrive_downloader.py` is required. Public access and download availability were not verified. Keep these personal files out of version control and shared packages.
- **Local files:** prepared inputs can be placed directly in the structure above. Target CSVs must also be prepared. Their population command is in the setup submenu, which currently authenticates with Drive before opening; a fully offline setup wizard is not implemented.
- **Weights:** torchvision extractors request ImageNet weights (`IMAGENET1K_V1`), which may download to the PyTorch cache on first use. These are visual extractors, not BATEIA parameter regressors. Train regressors through the menu or supply compatible local artifacts; no hosted BATEIA checkpoint is promised.

<a id="usage"></a>
## Your first run

Run from the project root so relative authentication and log paths resolve correctly:

```sh
python bateia_exec.py
```

Menu labels below remain those of the application; documentation languages do not change the interface.

1. Choose **[1] Setup and Data Download** if using Drive. Download/update literature parameters, populate target parameters, then download cluster `.dat` files. This requires your own authorized account and OAuth configuration.
2. Choose **[3] Verify Local Clusters** in the main menu. Confirm the expected clusters are detected. Detection checks the coordinate file path, not scientific quality.
3. Choose **[2] Run Analysis and AI Pipeline → [1] Run Data Analysis Stages**. Select one prepared cluster and confirm. Run data processing and real-cluster analysis before dependent stages; then generate/analyse synthetics and extract features. Choose GAP, no-GAP or both when prompted. Heatmaps are an optional analysis stage.
4. With features and targets available for enough clusters, choose **Run AI Training** in the execution menu. Select a model and matching feature mode. Use **AM Methodology (Uncontaminated)** with explicit cluster splits for unseen-cluster evaluation. Manual splitting requires at least 20 clusters (15 train, 4 validation, remainder test). Stratified splitting lets you select test clusters and divides development data 70/30 using K-Means profiles, subject to sample/group constraints. **BM Methodology (Contaminated)** uses all data and does not demonstrate unseen-cluster generalization.
5. Inspect outputs; use **[5] Generate Results and Paper Reports** once the needed predictions exist. Choose **[7] Exit** to leave.

Start with one cluster to inspect preprocessing, not to run the documented training split. Synthetic generation defaults to up to 1,000 samples per cluster. Large selections can require substantial time, RAM, VRAM and disk space.

**Numbering note:** heatmaps are analysis stage 6, while training is also labelled “Stage 6” elsewhere. Follow menu names and dependencies. **[4] Reset Workspace deletes generated data/models** according to the chosen reset operation.

### Output locations

Each cluster receives generated content in `Posicoes/`, `Target/`, `CNN/` and `Sinteticos/` as applicable. Synthetic features are in `Sinteticos/Features/<EXTRACTOR>/` (`.pt` or `_semgap.pt`) and parameter tables in `Sinteticos/Resultados/`. Training outputs are under `Redes Neurais/`, with model/scaler artifacts, prediction CSVs and learning curves in execution/model subdirectories. Exact files depend on the selected stages and model.

> **Future image · A real example**  
> Reserve this space for a cluster map, fitted King profile and predicted parameters. Caption: cluster, units, split, model and data provenance.

> **Future image · Explainability**  
> Reserve this space for comparable Score-CAM and Ablation-CAM panels. Identify extractor and input image; heatmaps are diagnostics, not independent physical validation.

<a id="docs"></a>
## Documentation and troubleshooting

| Resource | Content |
| --- | --- |
| [Architecture](readme/ARQUITETURA.md) | Modules and project structure |
| [Model guide](readme/MODELS_GUIDE.md) | GAP, regressors and optimization |
| [GPU parallelism](readme/PARALELISMO-GPUS.md) | Existing multi-GPU execution paths |
| [Environment](environment.yml) | Declared dependencies |
| [Article](artigo-bateia.pdf) · [Dissertation](dissertacao-bateia.pdf) | Scientific methodology and bibliography |

The existing technical guides are in Portuguese. Consult the source when they differ from the current menu.

| Symptom | First check |
| --- | --- |
| `conda` not found | Open a Conda-enabled terminal after installation. |
| Import/dependency error | Activate `bateia`; inspect installation output and selected interpreter. |
| No clusters found | Check `BATEIA_ROOT` and `<Cluster>/Posicoes/<Cluster>_coords.dat`. |
| Drive failure | Check your OAuth configuration, account permissions and configured folder. |
| Missing features/targets | Complete synthetic generation/analysis and extraction; match GAP mode, extractor, names and target parameters. |
| Insufficient memory | Reduce the selected workload and review generation/training settings. |

### Notebooks and Kaggle

A convenient notebook-first library is a future direction. There is no ready Kaggle notebook, stable high-level `fit`/`predict` facade or supported notebook installation procedure in this snapshot. Internal modules require adapting paths, dependencies, authentication and interactive/multiprocessing behavior. The entry point documented here remains the terminal menu.

<a id="references"></a>
## Scientific references and citation

Entries were checked against the supplied article and dissertation. Cite the article for BATEIA-1 methodology; consult the dissertation for the broader model study.

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

### License and contributions

No software license file was found in the inspected project. The article's CC BY and dissertation's CC BY-NC notices apply to those publications and do not establish a source-code license. Clarify reuse terms with the maintainers. Contributions should describe data provenance, configuration, environment and validation; exclude credentials and private tokens.

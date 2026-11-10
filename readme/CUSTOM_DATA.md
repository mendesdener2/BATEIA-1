[English](CUSTOM_DATA.md) · [Português (Brasil)](CUSTOM_DATA.pt-BR.md) · [Español](CUSTOM_DATA.es.md)

[← README](../README.md)

# Analyse your own stellar cluster data

This guide describes an advanced workflow using the current source and the inspected `modelos-ia-bateia-mestrado` archive. It provides a concrete SVR inference example and explains how to extend the workflow to deep learning or new training data. The Python blocks are examples built from existing functions, **not a new high-level API**. They were checked against the source and archive filenames, but have not been executed with the trained models in this environment. No runtime compatibility or scientific accuracy is claimed by these checks.

## 1. Choose your scientific task

| Task | Required inputs | Need known target parameters? |
| --- | --- | --- |
| Apply a saved model to a new cluster | Prepared stellar coordinates; matching extractor, regressor and both scalers | No, for the direct inference route below |
| Train new regressors | A labelled cluster sample, synthetic features/targets, a split and training settings | Yes; labels must have a defined scientific origin |
| Reproduce the publication | Original data, experiment splits, settings, code version and artifacts | The original experiment's labels and references |

Use non-embedded open clusters compatible with the training domain and single-component King profiles. A numerically valid prediction is not evidence of generalization to different stellar populations, membership selections or field contamination.

## 2. Install and locate the files

**Published resources:** [BATEIA-1 dataset](https://doi.org/10.5281/zenodo.24087976) · [Trained models and scalers](https://doi.org/10.5281/zenodo.24094057). For the direct inference example, download the model/scaler artifacts and prepare your own coordinates; the full training dataset is not required just to predict a new cluster. Use the dataset when reproducing or retraining the research experiments.

Download and extract the required files manually, preserving filenames and internal directories. The archive tree below describes the inspected research collection; locate the actual `Redes Neurais` directory if extraction adds an outer folder. Do not infer a downloadable filename from its DOI, or rename historical artifacts to match current output patterns.

Follow [installation](USAGE.md#installation), activate `bateia` and run the examples from the repository root. Use one Python session or notebook kernel for steps 4–7. Set `BATEIA_ROOT` before the first `bateia` import; restart the interpreter if it was already imported with another root. Notebook use here requires this manual setup and does not imply a packaged notebook API.

Keep the downloaded model archive separate from your observational data. Do not assume the top-level archive folder is the model root: the examples require its **`Redes Neurais` subdirectory**. Loading legacy `.joblib`/`.pkl` files requires compatible scikit-learn/joblib versions and may execute serialized code; use the author's trusted artifacts. The current Conda recipe is not proof that legacy artifacts are compatible with it.

## 3. Understand the model archive

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

Directory names are literal. `MLPerceptron` contains MLP experiments; branch names distinguish AM and BM. Extractor directories include `DENSENET121`, `DENSENET161`, `DENSENET201`, `RESNET50/101/152` and `VGG13/16/19`.

An actual matching triplet in the inspected archive is:

```text
SVR/AMSVR/Modelos/DENSENET121/f0_densenet121_svr_M67_gap_svr_model.joblib
SVR/AMSVR/Scaler/DENSENET121/scaler_x_f0_densenet121_svr_M67_gap.pkl
SVR/AMSVR/Scaler/DENSENET121/scaler_y_f0_densenet121_svr_M67_gap.pkl
```

`f0` is the output parameter, `densenet121` the extractor, `svr` the regressor, `M67` the historical experiment label and `gap` the representation. **Do not replace `M67` in an existing filename with your new cluster's name.** Inspect the corresponding `Logs/` records to establish the training/validation/test membership; the label alone does not establish a split. AM targets held-out-cluster evaluation; BM uses all data.

The archive contains legacy naming variations, including neural-network files without an explicit `_gap` component and older Optuna databases. Absence of a suffix does not prove GAP or no-GAP: establish the representation from logs, feature dimensions and original configuration. The current source writes other naming patterns such as `_split_...`; do not rename archived files to pretend they were generated by the current version. `Previsto/` and learning-curve JPGs are results, not models.

## 4. Prepare an observational sample and coordinate file

Select stars and determine the cluster centre scientifically before conversion. Record Gaia release, query/quality filters, membership/field selection, magnitude cuts, field size and epoch. Do not treat arbitrary absolute RA/Dec as centred coordinates: the converter copies the two coordinate columns directly, and the radial analysis assumes a centre at `(0, 0)` and multiplies distances by 60.

Use centred angular offsets **in degrees**: the expected small-field convention is approximately `X = ΔRA × cos(Dec0)` and `Y = ΔDec`. For wide fields, polar regions or a different projection, define and validate an appropriate tangent-plane transformation and its compatibility with training. Keep enough field coverage to assess background and truncation; do not alter membership or contamination conventions silently.

```text
BATEIA-data/
└── MyCluster/
    ├── Posicoes/
    │   └── MyCluster_coords.dat
    ├── CNN/                      # generated
    └── Features/                 # generated
```

The `.dat` format has one header line and six whitespace-separated columns: `ID`, `RAlin(J2016)`, `DEClin(J2016)`, `Gmag`, `BPmag`, `RPmag`. Despite their names, the coordinate columns must contain the prepared offsets for this pipeline. Do not use arcminutes here, which would introduce an additional factor of 60. Match cluster names exactly, including case. This small-field example expects your own CSV and centre:

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

If your data already has centred offsets in degrees, use those instead of centring again. The example requires finite photometry as a preparation check; document any filtering of missing values. It does not retrieve Gaia data or perform membership selection. Conversion creates a semicolon-separated `Posicoes/MyCluster_coords_processed.csv` (`X`, `Y`), `CNN/MyCluster_1.jpg` (clean model input) and `CNN/MyCluster_stars.jpg` (labelled diagnostic). Inspect centring, field size, orientation and crowding before continuing. Do not substitute a heatmap, labelled figure or synthetic-overview image as the model input.

## 5. Extract features with the matching representation

Use the same source/extractor preprocessing as the archived training run. The current real-image extraction uses ImageNet normalization and 224 × 224 resizing. ImageNet weights may need a first download. The existing function below processes **all registered extractors**, so it can consume time and memory even though the example later uses only DenseNet121. No target CSV, synthetic generation, Google authentication or cluster-specific fit settings are required by this direct feature-extraction function.

```python
# Continue in the SAME session as the preparation example.
from bateia.heatmaps_features import executar_analise_reais

executar_analise_reais([cluster], suffix=".pt", apply_gap=True)
feature_path = root / cluster / "Features" / "DENSENET121" / f"features_real_{cluster}.pt"
assert feature_path.is_file(), "Inspect extraction errors before continuing"
```

For a verified no-GAP experiment, the corresponding call is `executar_analise_reais([cluster], suffix="_semgap.pt", apply_gap=False)` and the real feature file ends in `_semgap.pt`. DenseNet121 GAP is a 1,024-component vector; its spatial representation has 1,024 × 7 × 7 components before flattening. Matching dimensions is necessary but not sufficient: preprocessing, layer, weight version and feature ordering must also match. Validate with a reference cluster from the archive before interpreting a new target.

## 6. Select one coherent model/scaler set

Start with SVR + DenseNet121 + GAP and one AM experiment, solely as a concrete example, not as a claim that it is the best model. The code below uses the verified `M67` experiment filenames and predicts all four parameters. Change family/extractor/experiment only after locating and verifying the complete artifact sets. Never refit downloaded scalers on one new cluster, combine scalers across parameters or silently switch GAP modes.

## 7. Predict and save a separate result table

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

The custom example writes `MyCluster/Predictions/svr_densenet121_M67_gap.csv`; this is its own output location, not a newly added menu command or the historical `Previsto/` layout. The source's inference functions apply an absolute value after inverse scaling, so the example retains **both** the raw value and that convention. Negative raw predictions require investigation, not automatic physical acceptance. A feature-range warning flags extrapolation relative to the saved input scaler; absence of that warning does not validate the prediction. A blank flag means the scaler does not expose stored feature bounds and this check was not available.

## 8. Validate the scientific interpretation

Confirm target units from the training labels: this workflow uses `Rc`/`Rt` in arcminutes and `f0`/`fbg` in stars per square arcminute. The lowercase filename tokens `rc`/`rt` refer to `Rc`/`Rt`, not new quantities. Check finite values, positivity, radius ordering, sensitivity to centring/selection, reference-cluster consistency and an independently constructed radial profile. These are checks, not guarantees of physical validity.

Unknown target parameters are not required for inference, but reference truth is required to compute accuracy or percent errors. Do not report RMSE against an unknown target. The archive's `inct_*` metadata and literature uncertainty bands are not calibrated predictive uncertainties for your new cluster. Score-CAM/Ablation-CAM characterize extractor responses, not confidence intervals on the regressed physical parameters.

## 9. Extend to deep-learning regressors

MLP artifacts are under `MLPerceptron/AMMLP/` or `BMMLP/`; LSTM, GRU and 1D-CNN have their own branches. For example, the archive contains `f0_densenet121_mlp_M67_gap_mlp_model.pth`. Locate its exact `scaler_x_...pkl` and `scaler_y_...pkl` partners rather than assuming the SVR scalers apply.

The current saver writes PyTorch `state_dict` weights, not a self-contained callable network. Reconstruct the original network using `bateia/models/architectures.py` and the archived architecture/settings. Current factories are in the corresponding `*_model.py` modules. Historical logs may not include every constructor argument; missing architecture metadata must be recovered from the original run/source or compatible weight shapes, not guessed from a new search-space default. The following is an **adaptation pattern**, not a complete runnable example:

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

MLP uses flat inputs. Current sequential models use `(batch, 1, C)` with GAP or `(batch, 49, C)` without GAP after the training scaler's flattened transformation. Verify the exact reshape order against the original trainer; changing the order produces different inputs even with the same dimensions. Use `strict=True`; resolve incompatible keys/shapes with the original architecture rather than hiding them with `strict=False`. `weights_only=True` restricts deserialization but does not reconstruct architecture or guarantee compatibility.

### Concrete MLP example (DenseNet121, GAP, f0)

The inspected MLP checkpoint has `model.0.weight`, `model.3.weight`, …, `model.18.weight`: seven linear layers, including the output. Its complete depth is not established by the printed optimization log. This example derives hidden widths from the weights, checks the inspected linear/ReLU/dropout indexing and uses strict loading against the current `MLPRegressor`. It has not been executed here. If any assertion or strict-loading check fails, stop and recover the original architecture/version; do not relax the checks. Dropout is disabled for evaluation only, so this does not reproduce the original training configuration. For another parameter, use that parameter's full model/scaler triplet.

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

For LSTM/GRU/1D-CNN, recover the original layers, hidden widths, recurrent depth and reshape order; the flat MLP reconstruction is not applicable. Their complete executable loaders still require case-specific adaptation.

## 10. Optional: fit profiles, generate labels and retrain

This route differs from applying a saved model. Add the cluster to `PARAMETROS_AJUSTE` in `bateia/config_data/clusters.py`; the current real-analysis function skips unregistered clusters. Review `AjusteCentral`, `AjusteRestante`, `AjusteFundoBG`, `t` and `t_ideal`. They control empirical corrections/binning, not coordinates or learned network weights. Derive/validate them for your data; copying M67 values is not a scientific calibration. `inct_*` factors affect literature comparisons and are not new-target posterior uncertainties.

1. Prepare the coordinates/images as above. Add a justified metadata entry before starting a fresh interpreter.
2. For literature comparisons/stratification, prepare `cluster_params.csv` with `Cluster,f0,err_f0,Rc,err_Rc,Rt,err_Rt,fbg,err_fbg`. Use the established units and uncertainty meaning. The converter's `salvar_parametros_literatura()` can populate `Target/` locally without opening the Drive setup submenu; invoke it directly after setting `BATEIA_ROOT`. Do not fabricate unknown reference values. Some legacy comparisons differ in filename case/prefix: inspect generated `mat_Rc...` versus expected names on your platform.
3. Run real-cluster analysis through **Run Analysis and AI Pipeline → Run Data Analysis Stages**. Inspect radial profiles, residual tables and `Target/best_division_info_<Cluster>.csv`. The synthetic analyser uses `num_bins = t_ideal + 4`; set `t_ideal` consistently with a validated chosen bin count rather than copying a topology index.
4. Generate and analyse synthetics, then extract the matching features. Review synthetic images and `Sinteticos/Resultados/<Cluster>_synthetic_parameters.csv` (semicolon-separated, `Synthetic_ID` links labels to feature filenames). Current generation defaults to up to 1,000 samples per cluster.
5. Assemble a labelled cluster sample. Separate real clusters and all their derived synthetics between training, validation and test; AM's manual route requires at least 20 clusters (15 training, 4 validation, remainder test). Its stratified route needs complete literature metadata and adequate group sizes. BM does not provide an unseen-cluster evaluation.
6. Select models and GAP mode in **Run AI Training**. Review `n_trials` in `bateia_exec.py` (currently 1), epoch/early-stopping settings in `config.py` and search spaces in `config_data/hyperparams.py`. Current defaults are not automatically the publication's experiment configuration.
7. Save new models, scalers, splits and settings in a separate run. Do not train over the downloaded archive or use **Reset Workspace** on a data root whose artifacts must be preserved.

## 11. Common failure modes

| Problem | What to check |
| --- | --- |
| Cluster skipped by real analysis | Exact name and `PARAMETROS_AJUSTE` entry |
| Wrong angular scale or density | Centred degrees in the input, field coverage and original label units |
| Missing real features | Clean `_1.jpg`, extraction errors, extractor folder and suffix |
| Scaler/network dimension mismatch | Parameter, experiment, GAP mode, layer and architecture |
| Legacy model cannot load | Training dependency versions and original architecture; do not silently alter the artifact |
| Plausible number but unreliable science | Domain shift, sample selection, independent profile and reference-cluster checks |

## 12. Record what you used

Record the [dataset DOI](https://doi.org/10.5281/zenodo.24087976) and [model DOI](https://doi.org/10.5281/zenodo.24094057), the exact downloaded files/versions, code commit, artifact paths/checksums, environment versions, centre, coordinate convention, filters, feature settings, split and raw/output predictions. The examples remain source-inspected workflows, not independently validated new-cluster results. See [usage](USAGE.md), [models](MODELS_GUIDE.md), [architecture](ARCHITECTURE.md) and [dependencies](DEPENDENCIES.md).

[English](GPU_PARALLELISM.md) · [Português (Brasil)](PARALELISMO-GPUS.md) · [Español](GPU_PARALLELISM.es.md)

[← README](../README.md)

# GPU and CPU parallelism

## Start with a single process

```sh
python bateia_exec.py
```

The menu distributes data-analysis stages 1–4 across approximately 85% of available CPU cores. Feature extraction can divide clusters among visible GPUs when more than one GPU and cluster are available. Heatmap generation follows a separate sequential path. Training behavior depends on the selected regressor and its configuration; do not assume every model uses all CPU cores or every GPU automatically.

## Select a GPU

Set visibility before starting Python. In each example, GPU 0 is the physical device to expose; use 1 to expose GPU 1 instead.

```powershell
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"
python bateia_exec.py
```

```sh
# Linux/macOS shell, where a compatible CUDA setup is available
CUDA_VISIBLE_DEVICES=0 python bateia_exec.py
```

This restricts visibility; it does not itself configure distributed training. Several terminals on one GPU share its VRAM and may exhaust memory.

## Optuna journals and resuming

Both trainer backends use `JournalStorage` with `JournalFileBackend` and `load_if_exists=True`. Reusing an optimization study requires the same journal path and study name, which incorporate execution/job identifiers. Check output paths and identifiers before assuming two terminals share a study.

Journal storage records trial progress, but it is not an automatic checkpoint of every in-progress training epoch. Concurrent runs can also target the same final models/scalers/reports. Starting identical menus in several terminals is therefore not a verified end-to-end parallel training recipe, and concurrent speedups are not guaranteed.

Use `nvidia-smi`, when available, to inspect GPU memory and processes. Review workload, `n_trials` and CPU settings before scaling. Keep journal files when resuming; removing a journal discards its optimization history. No multi-GPU benchmark was run for this guide.

See [models](MODELS_GUIDE.md) and [usage](USAGE.md).

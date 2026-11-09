[English](GPU_PARALLELISM.md) · [Português (Brasil)](PARALELISMO-GPUS.md) · [Español](GPU_PARALLELISM.es.md)

[← README](README.es.md)

# Paralelismo en GPU y CPU

## Empiece con un proceso

```sh
python bateia_exec.py
```

El menú distribuye las etapas 1–4 del análisis entre aproximadamente el 85% de los núcleos CPU disponibles. La extracción puede repartir cúmulos entre GPUs visibles cuando hay más de una GPU y cúmulo. Los mapas de calor siguen una ruta secuencial separada. El entrenamiento depende del regressor y su configuración; no todos los modelos usan automáticamente todos los núcleos o GPUs.

## Seleccione una GPU

Defina la visibilidad antes de iniciar Python. En los ejemplos, GPU 0 es el dispositivo físico expuesto; use 1 para exponer GPU 1.

```powershell
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"
python bateia_exec.py
```

```sh
# Shell Linux/macOS, si existe una configuración CUDA compatible
CUDA_VISIBLE_DEVICES=0 python bateia_exec.py
```

Esto restringe la visibilidad; no configura por sí solo entrenamiento distribuido. Varios terminales en una GPU comparten VRAM y pueden agotar la memoria.

## Journals Optuna y reanudación

Ambos backends usan `JournalStorage` con `JournalFileBackend` y `load_if_exists=True`. Reutilizar un estudio exige la misma ruta de journal y nombre de estudio, que incorporan identificadores de ejecución/tarea. Compruebe rutas e identificadores antes de suponer que dos terminales comparten un estudio.

El journal registra progreso de los trials, pero no es un checkpoint automático de cada época en curso. Las ejecuciones concurrentes también pueden escribir los mismos modelos/scalers/informes finales. Iniciar menús idénticos en varios terminales no constituye una receta verificada de entrenamiento paralelo completo ni garantiza aceleración.

Use `nvidia-smi`, cuando esté disponible, para inspeccionar memoria y procesos. Revise carga, `n_trials` y configuración CPU antes de escalar. Conserve journals para reanudar; eliminar uno descarta su historial de optimización. No se ejecutaron benchmarks multi-GPU para esta guía.

Consulte [modelos](MODELS_GUIDE.es.md) y [uso](USAGE.es.md).

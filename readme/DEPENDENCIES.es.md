[English](DEPENDENCIES.md) · [Português (Brasil)](DEPENDENCIES.pt-BR.md) · [Español](DEPENDENCIES.es.md)

[← README](README.es.md)

# Dependencias

La receta de referencia es [environment.yml](../environment.yml). Solicita Python 3.12 e incluye:

- Datos científicos: NumPy, pandas, SciPy, scikit-learn, Astropy y astroquery.
- Imágenes/gráficos: Matplotlib, imageio, OpenCV, Pillow y corner.
- Entrenamiento: PyTorch, torchvision, torchaudio y Optuna.
- Explicabilidad: grad-cam.
- Utilidades: joblib, tqdm y clientes Google API/OAuth.

La sección pip añade el índice PyTorch nightly CUDA 12.8. Las versiones no están bloqueadas y no se exigen explícitamente versiones preliminares. GPU requiere controlador/distribución compatibles. Esta documentación no certifica compatibilidad de plataformas.

Siga la [instalación](USAGE.es.md#installation) antes de ejecutar el menú. Los pesos ImageNet y el dataset son independientes de las dependencias del entorno.

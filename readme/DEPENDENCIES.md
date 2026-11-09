[English](DEPENDENCIES.md) · [Português (Brasil)](DEPENDENCIES.pt-BR.md) · [Español](DEPENDENCIES.es.md)

[← README](../README.md)

# Dependencies

The authoritative environment recipe is [environment.yml](../environment.yml). It requests Python 3.12 and includes:

- Scientific data: NumPy, pandas, SciPy, scikit-learn, Astropy and astroquery.
- Images/plots: Matplotlib, imageio, OpenCV, Pillow and corner.
- Training: PyTorch, torchvision, torchaudio and Optuna.
- Explainability: grad-cam.
- Utilities: joblib, tqdm and Google API/OAuth clients.

The pip section adds the PyTorch nightly CUDA 12.8 index. Versions are not locked and prereleases are not explicitly required. GPU use needs a compatible driver/build. This documentation does not certify platform compatibility.

Follow [installation](USAGE.md#installation) before running the menu. ImageNet weights and the project dataset are separate from environment dependencies.

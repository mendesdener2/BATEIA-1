[English](DEPENDENCIES.md) · [Português (Brasil)](DEPENDENCIES.pt-BR.md) · [Español](DEPENDENCIES.es.md)

[← README](README.pt-BR.md)

# Dependências

A receita de referência é [environment.yml](../environment.yml). Solicita Python 3.12 e inclui:

- Dados científicos: NumPy, pandas, SciPy, scikit-learn, Astropy e astroquery.
- Imagens/gráficos: Matplotlib, imageio, OpenCV, Pillow e corner.
- Treinamento: PyTorch, torchvision, torchaudio e Optuna.
- Explicabilidade: grad-cam.
- Utilitários: joblib, tqdm e clientes Google API/OAuth.

A seção pip adiciona o índice PyTorch nightly CUDA 12.8. As versões não estão travadas e pré-lançamentos não são exigidos explicitamente. GPU requer driver/distribuição compatíveis. Esta documentação não certifica compatibilidade de plataformas.

Siga a [instalação](USAGE.pt-BR.md#installation) antes de executar o menu. Pesos ImageNet e o dataset são separados das dependências do ambiente.

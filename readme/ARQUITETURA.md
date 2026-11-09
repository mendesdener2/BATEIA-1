# Arquitetura do Sistema BATEIA 🚀

Este documento descreve a organização e a responsabilidade de cada módulo dentro do projeto BATEIA. A arquitetura segue uma estrutura modular e profissional, separando a lógica de orquestração, processamento de dados e construção de modelos de IA.

---

## 1. Orquestração Principal (Raiz)
- **`bateia_exec.py`**: O **"Coração"** do sistema. Ponto de entrada interativo que orquestra todo o pipeline: download de dados, split estratificado, treinamento multi-GPU e geração de relatórios.
- **`environment.yml`**: Definição do ambiente Conda para reprodutibilidade científica.

---

## 2. Módulo Central (`bateia/`)
Contém o código de domínio da física/astrofísica e as configurações globais.
- **`config.py`**: Central de configurações. Define constantes do Optuna, caminhos do disco (`BASE_DIR`), topologias e hiperparâmetros.
- **`config_data/`**: Metadados de aglomerados (`clusters.py`) e espaços de busca (`hyperparams.py`).
- **`config_data/paper_data.py`**: (Novo) Centraliza dados de referência de Angelo e Hunt para o artigo.

---

## 3. Inteligência Artificial (`bateia/models/`)
Camada de Deep Learning e Modelos Estatísticos.
- **`architectures.py`**: Definição das classes `torch.nn.Module` (MLP, LSTM, GRU, 1D-CNN).
- **`core_trainer.py`**: Motor de treinamento PyTorch integrado ao Optuna.
- **Arquivos `*_model.py`**: Fábricas de modelos e funções de objetivo para otimização bayesiana.

---

## 4. Ferramentas e Relatórios (`bateia/tools/`)
Scripts utilitários e geradores de resultados acadêmicos.
- **`paper_utils.py`**: (Novo) Consolida a geração de tabelas e exportação LaTeX para publicações.
- **`correction.py`**: (Novo) Ferramenta de manutenção para recalcular métricas e corrigir CSVs de predição.
- **`verify_split.py`**: (Novo) Validador da estratégia de amostragem e estratificação de dados.
- **`results_bateia.py`**: Compilador de métricas históricas de desempenho.

---

## 5. Documentação (`readme/`)
- **`ARQUITETURA.md`**: Este documento.
- **`MODELS_GUIDE.md`**: Detalhes técnicos sobre GAP e arquiteturas sequenciais.
- **`PARALELISMO-GPUS.md`**: Guia para treinamento distribuído.

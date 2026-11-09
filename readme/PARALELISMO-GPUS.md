[English](GPU_PARALLELISM.md) · [Português (Brasil)](PARALELISMO-GPUS.md) · [Español](GPU_PARALLELISM.es.md)

[← README](README.pt-BR.md)

# Paralelismo em GPU e CPU

## Comece com um processo

```sh
python bateia_exec.py
```

O menu distribui as etapas 1–4 de análise por aproximadamente 85% dos núcleos CPU disponíveis. A extração pode dividir aglomerados entre GPUs visíveis quando há mais de uma GPU e aglomerado. Mapas de calor seguem um caminho sequencial separado. O treinamento depende do regressor e da configuração; nem todo modelo usa automaticamente todos os núcleos ou GPUs.

## Selecione uma GPU

Defina a visibilidade antes de iniciar Python. Nos exemplos, GPU 0 é o dispositivo físico exposto; use 1 para expor a GPU 1.

```powershell
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"
python bateia_exec.py
```

```sh
# Shell Linux/macOS, quando houver configuração CUDA compatível
CUDA_VISIBLE_DEVICES=0 python bateia_exec.py
```

Isso restringe visibilidade; não configura treinamento distribuído por si só. Vários terminais na mesma GPU compartilham VRAM e podem esgotar a memória.

## Journals Optuna e retomada

Os dois backends usam `JournalStorage` com `JournalFileBackend` e `load_if_exists=True`. Reutilizar um estudo exige o mesmo caminho do journal e nome do estudo, que incorporam identificadores de execução/tarefa. Confira caminhos e identificadores antes de supor que dois terminais compartilham um estudo.

O journal registra progresso dos trials, mas não é um checkpoint automático de cada época em andamento. Execuções concorrentes também podem gravar nos mesmos modelos/scalers/relatórios finais. Iniciar menus iguais em vários terminais não constitui uma receita verificada de treinamento paralelo completo, nem garante aceleração.

Use `nvidia-smi`, quando disponível, para inspecionar memória e processos. Revise carga, `n_trials` e configuração CPU antes de ampliar a execução. Preserve journals para retomar; remover um journal descarta o histórico de otimização. Nenhum benchmark multi-GPU foi executado para este guia.

Consulte [modelos](MODELS_GUIDE.pt-BR.md) e [uso](USAGE.pt-BR.md).

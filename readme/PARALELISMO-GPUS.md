# Guia de Paralelismo: GPU e CPU (BATEIA)

Este documento explica como utilizar a nova arquitetura de paralelismo do projeto BATEIA, baseada em **Optuna JournalStorage**. Esta implementação permite que múltiplos processos (terminais) colaborem na mesma otimização de hiperparâmetros sem a necessidade de bancos de dados complexos como PostgreSQL.

---

## 1. Como funciona o novo sistema?
Substituímos o banco de dados SQLite (`.db`) pelo **JournalStorage** (`.log`). 
- **Vantagem:** Permite que vários processos escrevam e leiam o progresso simultaneamente sem erros de "Database locked".
- **Resiliência:** Se uma execução for interrompida, basta rodar novamente; o sistema retoma do ponto onde parou.

---

## 2. Cenário: 1 GPU (Padrão e Seguro)
Se você tem apenas uma GPU e deseja rodar o treinamento da forma mais estável possível:

**Comando:**
```bash
python bateia_exec.py
```
**O que acontece:**
- As Redes Neurais (CNN, LSTM, GRU, MLP) rodarão uma por uma sequencialmente na GPU.
- Os modelos de Machine Learning (SVR, RF) usarão **todos os núcleos da sua CPU** automaticamente em paralelo.

---

## 3. Cenário: 2 ou mais GPUs (Alta Performance)
Se você tem múltiplas GPUs físicas, pode distribuir o trabalho entre elas para terminar a otimização muito mais rápido.

**Terminal 1 (GPU 0):**
```bash
# No Windows (PowerShell):
$env:CUDA_VISIBLE_DEVICES="0"; python bateia_exec.py

# No Linux:
CUDA_VISIBLE_DEVICES=0 python bateia_exec.py
```

**Terminal 2 (GPU 1):**
```bash
# No Windows (PowerShell):
$env:CUDA_VISIBLE_DEVICES="1"; python bateia_exec.py

# No Linux:
CUDA_VISIBLE_DEVICES=1 python bateia_exec.py
```

**Por que funciona?**
Como ambos os terminais apontam para o mesmo arquivo de log do Optuna, o **Terminal 1** verá que a "Trial 1" já começou e pulará automaticamente para a "Trial 2", enquanto o **Terminal 2** assume a "Trial 3", e assim por diante.

---

## 4. Cenário: 1 GPU com Múltiplos Terminais
Se os seus modelos forem leves (como MLP) e você quiser tentar rodar dois ao mesmo tempo na **mesma GPU** para aproveitar o tempo de espera de I/O:

1. Abra dois terminais.
2. Rode `python bateia_exec.py` em ambos.

**Atenção:** Isso dividirá a memória da sua GPU (VRAM). Se o modelo for grande, ambos os scripts podem falhar com erro de "Out of Memory" (OOM). Use com cautela!

---

## 5. Resumo de Comandos Úteis
- **Limpar Progresso:** Para começar uma otimização do zero, exclua o arquivo `*_optuna_journal.log` na pasta de resultados do modelo.
- **Verificar GPUs:** Use o comando `nvidia-smi` para monitorar o uso de memória e processos em cada placa.

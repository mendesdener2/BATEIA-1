# bateia/readme.py
import os
import sys

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def exibir_sobre():
    """Exibe informações conceituais sobre a abordagem BATEIA-1."""
    while True:
        limpar_tela()
        print("="*80)
        print(f"{'PROJETO BATEIA-1 - CONCEITOS E ARQUITETURAS':^80}")
        print("="*80)
        
        print("\n[1] Estratégia de Processamento (GAP vs Sem GAP)")
        print("[2] Metodologias de Validação (AM vs BM)")
        print("[3] Espaço de Busca de Hiperparâmetros (Optuna)")
        print("[4] Explainable AI (XAI / Mapas de Calor)")
        print("[5] Voltar ao Menu Principal")
        
        escolha = input("\nEscolha uma opção: ").strip()
        
        if escolha == '1':
            _exibir_gap_vs_nogap()
        elif escolha == '2':
            _exibir_am_vs_bm()
        elif escolha == '3':
            _exibir_intervalos_optuna()
        elif escolha == '4':
            _exibir_xai()
        elif escolha == '5':
            break

def _exibir_gap_vs_nogap():
    limpar_tela()
    print("--- GAP vs NO-GAP (Estratégia de Entrada) ---\n")
    print("O BATEIA-1 processa a saída das extratoras (ex: VGG16: 512x7x7) de duas formas:")
    print("\n1. COM GAP (Global Average Pooling):")
    print("   - Compacta a dimensão espacial numa média global.")
    print("   - Entrada para o modelo: Vetor 1D (ex: 512 features).")
    print("   - Foco: Robustez estatística e velocidade.")
    print("   - Modelos: MLP, SVR, RandomForest.")
    print("\n2. SEM GAP (Spatial/Sequential):")
    print("   - Preserva a topologia 7x7 do aglomerado.")
    print("   - Entrada para o modelo: Sequência de 49 passos temporais/espaciais.")
    print("   - Foco: Detecção de estruturas locais (núcleo vs cauda).")
    print("   - Modelos: LSTM, GRU, 1D-CNN.")
    input("\nPressione Enter para voltar...")

def _exibir_am_vs_bm():
    limpar_tela()
    print("--- Metodologias AM vs BM (Validação) ---\n")
    print("1. AM (Approximated Method / Não-Contaminado):")
    print("   - Simulação 'Cega': O aglomerado de teste é removido da base.")
    print("   - Nenhum dado sintético gerado a partir do alvo é usado no treino.")
    print("   - Objetivo: Testar generalização pura para objetos desconhecidos.")
    print("\n2. BM (Benchmark Method / Contaminado):")
    print("   - 'Best Case Scenario': O modelo treina com sintéticos de TODOS os aglomerados.")
    print("   - Objetivo: Testar a capacidade máxima de ajuste (fitting) da arquitetura.")
    input("\nPressione Enter para voltar...")

def _exibir_intervalos_optuna():
    limpar_tela()
    print("--- Espaço de Busca de Hiperparâmetros (Optuna) ---\n")
    print("A arquitetura não é fixa. O BATEIA-1 busca a melhor configuração dentro destes intervalos:")
    
    print("\n[MLP] Multi-Layer Perceptron")
    print("   - Perfil Adaptativo: Parâmetros f0, Rc e Rt usam dropout baixo (0.05-0.2).")
    print("   - O parâmetro fbg (fundo) é mais ruidoso, exigindo dropout agressivo (0.4-0.6) e LR maior.")
    print("   - Batch Size: [8, 16, 32]")

    print("\n[LSTM & GRU] Redes Recorrentes")
    print("   - Hidden Size: [64, 128, 256] unidades.")
    print("   - Camadas Recorrentes: 1 a 3 (Empilhadas).")
    print("   - Camadas Densas Finais: 1 a 3.")
    print("   - Neurônios Densos: [64, 128, 256].")
    
    print("\n[1D-CNN] Convolutional Neural Network")
    print("   - Camadas Convolucionais: 1 a 4.")
    print("   - Canais de Saída: [32, 64, 128].")
    print("   - Tamanho do Kernel: [3, 5, 7] (Janela de leitura da sequência).")

    print("\n[SVR] Support Vector Regressor (Cercado)")
    print("   - C (Regularização): 0.1 a 100.0 (Log scale).")
    print("   - Gamma: 1e-4 a 0.1.")
    print("   - Kernels: RBF ou Polinomial (Graus 2-5).")

    print("\n[RandomForest] (Cercado)")
    print("   - Estimadores (Árvores): 50 a 300.")
    print("   - Profundidade Máxima: 5 a 30.")
    
    input("\nPressione Enter para voltar...")

def _exibir_xai():
    limpar_tela()
    print("--- Explainable AI (XAI) - Mapas de Calor ---\n")
    print("O BATEIA-1 utiliza algoritmos de Grad-CAM para visualizar ONDE a rede 'olha' no aglomerado.")
    print("Isso garante que a IA está aprendendo astrofísica (densidade estelar) e não ruído de fundo.")
    
    print("\n1. Baseados em Gradiente (Gradient-based):")
    print("   - Grad-CAM: O clássico. Usa o gradiente da classe alvo na última camada convolucional.")
    print("   - Grad-CAM++: Melhor para localizar múltiplos objetos ou estruturas complexas.")
    print("   - Score-CAM: Não depende de gradientes, usa o peso da ativação no score final.")
    
    print("\n2. Livres de Gradiente (Gradient-free):")
    print("   - Eigen-CAM: Usa os componentes principais (PCA) das ativações. Ótimo para localizar o objeto principal.")
    print("   - Ablation-CAM: Remove partes da ativação para ver o quanto o score cai.")
    print("   - Layer-CAM: Captura detalhes de granulação mais fina em camadas anteriores.")
    
    print("\n")
    input("\nPressione Enter para voltar...")

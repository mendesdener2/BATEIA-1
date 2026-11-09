# bateia/models/core_trainer.py
import sys, os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.preprocessing import MinMaxScaler
from bateia.config import BASE_DIR, TRAINING_CONSTANTS, RESHAPE_CONFIG, RESHAPE_CONFIG_SEQ, PARAMS_CONFIG
from bateia.tools.dl_utils import (
    Logger, load_data, train_model_pytorch, 
    gerar_relatorio_consistencia, test_model_pytorch, salvar_resultados_finais, predict_on_real_clusters
)
from bateia.tools.verificador import listar_aglomerados_locais
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

def executar_treino_pytorch_generico(
    modelo_nome: str, 
    objective_func: callable, 
    final_model_factory: callable,
    metodologia: str, 
    parametro: str, 
    extratora: str, 
    data_split_config: list, 
    config_geral: dict, 
    output_dir: str, 
    gap_label: str,
    suffix: str = ".pt"
) -> None:
    print(f"\n{'='*25} INICIANDO TREINO: {modelo_nome} ({metodologia}) | {parametro} | {extratora} | {suffix} {'='*25}")

    log_dir = os.path.join(output_dir, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    device = config_geral.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    n_trials = config_geral.get("n_trials", 30)

    progress_file = os.path.join(output_dir, f"bm_{modelo_nome.lower()}_progress.log" if metodologia == 'BM' else f"am_{modelo_nome.lower()}_progress.log")
    job_id = f"{parametro}_{extratora}"
    completed_jobs = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f: completed_jobs = set(line.strip() for line in f)
    
    if metodologia == 'BM' and job_id in completed_jobs:
        print(f"✅ Job '{job_id}' (BM-{modelo_nome}) já concluído. Pulando.")
        return

    splits_para_rodar = data_split_config if metodologia == 'AM' else [{"train": [], "val": [], "test": []}]

    for i, data_split in enumerate(splits_para_rodar):
        if metodologia == 'AM':
            c_list = data_split.get("test", [])
            run_id = c_list[0] if len(c_list) == 1 else f"Group_{len(c_list)}"
            am_job_key = f"{job_id}_{run_id}"
            if am_job_key in completed_jobs:
                print(f"✅ Job '{am_job_key}' (AM-{modelo_nome}) já concluído. Pulando.")
                continue
        else:
            run_id = "BM"

        tag_extra = suffix.replace('.pt', '')
        log_filename = f"log_{metodologia.lower()}-{modelo_nome.lower()}_{job_id}_{run_id}{tag_extra}.txt"
        log_filepath = os.path.join(log_dir, log_filename)
        original_stdout = sys.stdout
        sys.stdout = Logger(log_filepath)

        try:
            print(f"\n{'='*50}\nIniciando: {modelo_nome} ({metodologia}) - Alvo: {run_id}\n{'='*50}")

            if metodologia == 'AM':
                train_X, train_y, _ = load_data(data_split["train"], parametro, extratora, BASE_DIR, suffix=suffix)
                val_X, val_y, val_cluster_ids = load_data(data_split["val"], parametro, extratora, BASE_DIR, suffix=suffix)
                test_X, test_y, test_cluster_ids = load_data(data_split["test"], parametro, extratora, BASE_DIR, suffix=suffix)
                clusters_to_predict = data_split["test"] 
            else: 
                all_X, all_y, all_cluster_ids = load_data(listar_aglomerados_locais(BASE_DIR), parametro, extratora, BASE_DIR, suffix=suffix)
                if all_X is None: raise Exception("Dados não encontrados")
                indices = np.arange(len(all_X))
                np.random.RandomState(seed=42).shuffle(indices)
                n_train, n_val = int(0.6 * len(indices)), int(0.2 * len(indices))
                train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
                train_X, train_y = all_X[train_idx], all_y[train_idx]
                val_X, val_y, val_cluster_ids = all_X[val_idx], all_y[val_idx], all_cluster_ids[val_idx]
                test_X, test_y, test_cluster_ids = all_X[test_idx], all_y[test_idx], all_cluster_ids[test_idx]
                clusters_to_predict = listar_aglomerados_locais(BASE_DIR) 

            if train_X is None: raise Exception("Dados de treino não encontrados")

            train_X = train_X.reshape(train_X.shape[0], -1)
            val_X = val_X.reshape(val_X.shape[0], -1)
            test_X = test_X.reshape(test_X.shape[0], -1)

            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            train_X_s = torch.from_numpy(scaler_x.fit_transform(train_X.numpy() if isinstance(train_X, torch.Tensor) else train_X)).to(torch.float32)
            val_X_s = torch.from_numpy(scaler_x.transform(val_X.numpy() if isinstance(val_X, torch.Tensor) else val_X)).to(torch.float32)
            test_X_s = torch.from_numpy(scaler_x.transform(test_X.numpy() if isinstance(test_X, torch.Tensor) else test_X)).to(torch.float32)
            
            train_y_s = torch.from_numpy(scaler_y.fit_transform(train_y)).to(torch.float32)
            val_y_s = torch.from_numpy(scaler_y.transform(val_y)).to(torch.float32)

            data = {"train": (train_X_s, train_y_s), "val": (val_X_s, val_y_s)}

            study_name = f"{metodologia.lower()}-{modelo_nome.lower()}-{job_id}-{run_id}"
            optuna_log_file = os.path.join(output_dir, f"{metodologia.lower()}_{modelo_nome.lower()}_optuna_journal.log")
            storage = JournalStorage(JournalFileBackend(optuna_log_file))
            study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), storage=storage, study_name=study_name, load_if_exists=True)
            
            if len(study.trials) < n_trials:
                study.optimize(lambda trial: objective_func(trial, parametro, data, device, suffix), n_trials=(n_trials - len(study.trials)))

            best_params = study.best_params
            print(f"\nOtimização concluída! Melhor Val RMSE: {study.best_value}")
            [print(f"  - {k}: {v}") for k, v in best_params.items()]

            print("\nTreinando o modelo final com os melhores parâmetros...")
            input_size_flat = train_X.shape[1]
            
            # Determinar as dimensões (seq_len e features_per_step)
            is_mlp = (modelo_nome == "MLP")
            if is_mlp:
                seq_len, features_per_step = 1, input_size_flat
            else:
                config_dict = RESHAPE_CONFIG_SEQ if "semgap" in suffix else RESHAPE_CONFIG
                if input_size_flat not in config_dict: 
                    raise ValueError(f"Tamanho {input_size_flat} não encontrado.")
                seq_len, features_per_step = config_dict[input_size_flat]

            best_batch_size = best_params["batch_size"]
            
            if is_mlp:
                train_X_reshaped = train_X_s
                val_X_reshaped = val_X_s
            else:
                train_X_reshaped = train_X_s.view(train_X_s.size(0), seq_len, features_per_step)
                val_X_reshaped = val_X_s.view(val_X_s.size(0), seq_len, features_per_step)
                
            final_train_loader = DataLoader(TensorDataset(train_X_reshaped, train_y_s), batch_size=best_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            final_val_loader = DataLoader(TensorDataset(val_X_reshaped, val_y_s), batch_size=best_batch_size, num_workers=0, pin_memory=True)

            final_model = final_model_factory(features_per_step, best_params, input_size_flat).to(device)
            optimizer_name = best_params.get("optimizer", "AdamW")
            lr = best_params.get("lr", 1e-4)
            if optimizer_name == "AdamW":
                optimizer = optim.AdamW(final_model.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(final_model.parameters(), lr=lr)

            final_model, train_losses, val_losses = train_model_pytorch(
                final_model, optimizer, final_train_loader, final_val_loader, TRAINING_CONSTANTS['final_epochs'], device,
                PARAMS_CONFIG[parametro]["patience"], PARAMS_CONFIG[parametro]["min_delta"]
            )
            
            gerar_relatorio_consistencia(final_model, val_X, val_y, val_cluster_ids, scaler_x, scaler_y, device, reshape_dims=(seq_len, features_per_step) if not is_mlp else None)

            test_y_s = torch.from_numpy(scaler_y.transform(test_y)).to(torch.float32)
            
            if is_mlp:
                final_test_loader = DataLoader(TensorDataset(test_X_s, test_y_s), batch_size=best_batch_size)
                final_mse, preds, targets = test_model_pytorch(final_model, final_test_loader, device, scaler_y, reshape_dims=None)
            else:
                final_test_loader = DataLoader(TensorDataset(test_X_s, test_y_s), batch_size=best_batch_size)
                final_mse, preds, targets = test_model_pytorch(final_model, final_test_loader, device, scaler_y, reshape_dims=(seq_len, features_per_step))
                
            print(f"🚀 Resultado Final -> Teste MSE: {final_mse:.6f}")

            scalers = {'x': scaler_x, 'y': scaler_y, 'train_losses': train_losses, 'val_losses': val_losses}
            salvar_resultados_finais(final_model, scalers, preds, targets, output_dir, parametro, extratora, run_id, modelo_nome, cluster_ids=test_cluster_ids)

            predict_on_real_clusters(
                final_model, scalers, clusters_to_predict, extratora, parametro, output_dir, run_id, device, 
                modelo_nome.lower(), gap_label,
                reshape_dims=(seq_len, features_per_step) if not is_mlp else None, suffix=suffix
            )

            with open(progress_file, 'a') as f:
                key_to_save = am_job_key if metodologia == 'AM' else job_id
                f.write(f"{key_to_save}\n")
            print(f"Progresso salvo! Job concluído.")

        except Exception as e:
            print(f"ERRO durante o treinamento de {modelo_nome}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if isinstance(sys.stdout, Logger): sys.stdout.log_file.close()
            sys.stdout = original_stdout

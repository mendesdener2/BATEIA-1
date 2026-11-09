# bateia/models/core_trainer_sklearn.py
import sys, os
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from bateia.config import BASE_DIR, T_MAX_SINTETICOS, OPTUNA_N_JOBS_CPU
from bateia.tools.dl_utils import (
    Logger, load_data, 
    gerar_relatorio_consistencia, salvar_resultados_finais, predict_on_real_clusters_sklearn
)
from bateia.tools.verificador import listar_aglomerados_locais
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

def executar_treino_sklearn_generico(
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
    n_trials = config_geral.get("n_trials", 20)

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
                train_X, train_y, _ = load_data(data_split["train"], parametro, extratora, BASE_DIR, limit=T_MAX_SINTETICOS, as_numpy=True, suffix=suffix)
                val_X, val_y, val_cluster_ids = load_data(data_split["val"], parametro, extratora, BASE_DIR, limit=T_MAX_SINTETICOS, as_numpy=True, suffix=suffix)
                test_X, test_y, test_cluster_ids = load_data(data_split["test"], parametro, extratora, BASE_DIR, limit=T_MAX_SINTETICOS, as_numpy=True, suffix=suffix)
                clusters_to_predict = data_split["test"]
            else:
                all_X, all_y, all_cluster_ids = load_data(listar_aglomerados_locais(BASE_DIR), parametro, extratora, BASE_DIR, limit=T_MAX_SINTETICOS, as_numpy=True, suffix=suffix)
                if all_X is None: raise Exception("No data found")
                indices = np.arange(len(all_X))
                np.random.RandomState(seed=42).shuffle(indices)
                n_train, n_val = int(0.6 * len(indices)), int(0.2 * len(indices))
                train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
                train_X, train_y = all_X[train_idx], all_y[train_idx]
                val_X, val_y, val_cluster_ids = all_X[val_idx], all_y[val_idx], all_cluster_ids[val_idx]
                test_X, test_y, test_cluster_ids = all_X[test_idx], all_y[test_idx], all_cluster_ids[test_idx]
                clusters_to_predict = listar_aglomerados_locais(BASE_DIR)

            if train_X is None: raise Exception("No training data")

            train_X = train_X.reshape(train_X.shape[0], -1)
            val_X = val_X.reshape(val_X.shape[0], -1)
            test_X = test_X.reshape(test_X.shape[0], -1)

            scaler_x = MinMaxScaler()
            train_X_s = scaler_x.fit_transform(train_X)
            val_X_s = scaler_x.transform(val_X)
            test_X_s = scaler_x.transform(test_X)
            
            scaler_y = MinMaxScaler()
            train_y_s = scaler_y.fit_transform(train_y)
            val_y_s = scaler_y.transform(val_y)
            
            data = {"train": (train_X_s, train_y_s), "val": (val_X_s, val_y_s)}

            study_name = f"{metodologia.lower()}-{modelo_nome.lower()}-{job_id}-{run_id}"
            optuna_log_file = os.path.join(output_dir, f"{metodologia.lower()}_{modelo_nome.lower()}_optuna_journal.log")
            storage = JournalStorage(JournalFileBackend(optuna_log_file))
            study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), storage=storage, study_name=study_name, load_if_exists=True)
            if len(study.trials) < n_trials:
                study.optimize(lambda trial: objective_func(trial, parametro, data), n_trials=(n_trials - len(study.trials)), n_jobs=1)

            best_params = study.best_params
            print(f"\nOtimização concluída! Melhor Val RMSE: {study.best_value}")
            
            final_model = final_model_factory(best_params)
            
            # Aviso para SVR em datasets grandes (subsampling removido a pedido do usuário)
            if modelo_nome == "SVR" and len(train_X_s) > 5000:
                print(f"  [AVISO] Treinando SVR com {len(train_X_s)} amostras sem limite. Isso pode demorar significativamente.")
            
            final_model.fit(train_X_s, train_y_s.ravel())
            
            gerar_relatorio_consistencia(final_model, val_X, val_y, val_cluster_ids, scaler_x, scaler_y, "cpu")

            preds_s = final_model.predict(test_X_s)
            preds = np.abs(scaler_y.inverse_transform(preds_s.reshape(-1, 1)))
            final_rmse = np.sqrt(mean_squared_error(test_y, preds))
            print(f"🚀 Resultado Final -> Teste RMSE: {final_rmse:.6f}")

            scalers = {'x': scaler_x, 'y': scaler_y}
            salvar_resultados_finais(final_model, scalers, preds, test_y, output_dir, parametro, extratora, run_id, modelo_nome, cluster_ids=test_cluster_ids)

            predict_on_real_clusters_sklearn(
                final_model, scalers, clusters_to_predict, 
                extratora, parametro, output_dir, run_id, 
                modelo_nome.lower(), gap_label, suffix=suffix
            )
            
            with open(progress_file, 'a') as f:
                key_to_save = am_job_key if metodologia == 'AM' else job_id
                f.write(f"{key_to_save}\n")
            print(f"Progresso salvo! Job concluído.")

        finally:
            if isinstance(sys.stdout, Logger): sys.stdout.log_file.close()
            sys.stdout = original_stdout

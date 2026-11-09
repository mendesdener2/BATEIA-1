# bateia/tools/data_loader.py
import os, sys, glob, re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from bateia.config import PARAMETROS_AJUSTE

class Logger(object):
    """Duplica a saída padrão (prints) para o console e para um arquivo de log."""
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def __del__(self):
        if self.log_file: self.log_file.close()

def load_data(clusters, parametro, extratora, base_dir, as_numpy=False, limit=None, suffix=".pt"):
    features, targets, cluster_ids = [], [], []

    for cluster_name in clusters:
        features_dir = os.path.join(base_dir, cluster_name, "Sinteticos", "Features", extratora.upper())
        targets_csv_path = os.path.join(base_dir, cluster_name, "Sinteticos", "Resultados", f"{cluster_name}_synthetic_parameters.csv")

        if not os.path.exists(targets_csv_path) or not os.path.isdir(features_dir):
            continue

        try:
            df_targets = pd.read_csv(targets_csv_path, sep=';').set_index('Synthetic_ID')
            
            search_pattern = f"*{suffix}"
            feature_files = glob.glob(os.path.join(features_dir, search_pattern))
            
            if limit is not None:
                feature_files = feature_files[:limit]

            regex_pattern = rf'_(\d+){re.escape(suffix)}$'

            for f_path in feature_files:
                match = re.search(regex_pattern, f_path)
                if not match:
                    continue
                
                synthetic_id = int(match.group(1))
                
                if synthetic_id in df_targets.index:
                    target_row = df_targets.loc[synthetic_id]
                    feature_data = torch.load(f_path, map_location=torch.device('cpu'), weights_only=True)
                    
                    if as_numpy:
                        features.append(feature_data.numpy())
                    else:
                        features.append(feature_data)
                    
                    targets.append(float(target_row[parametro]))
                    cluster_ids.append(cluster_name)

        except Exception as e:
            print(f"  AVISO: Erro ao processar dados para {cluster_name}: {e}")

    if not features:
        return None, None, None

    if as_numpy:
        return np.stack(features), np.array(targets).reshape(-1, 1), np.array(cluster_ids)
    else:
        return torch.stack(features), torch.tensor(targets, dtype=torch.float32).unsqueeze(1), np.array(cluster_ids)

def salvar_resultados_finais(model, scalers, preds, targets, output_dir, parametro, extratora, split_id, modelo_nome, cluster_ids=None):
    model_dir = os.path.join(output_dir, "Modelos", extratora)
    scaler_dir = os.path.join(output_dir, "Scaler", extratora)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)

    if str(split_id).lower() == "bm":
        filename_suffix = f"{parametro.lower()}_{extratora.lower()}_bm"
    else:
        filename_suffix = f"{parametro.lower()}_{extratora.lower()}_split_{split_id}"

    if isinstance(model, nn.Module):
        model_path = os.path.join(model_dir, f"{filename_suffix}_{modelo_nome.lower()}_model.pth")
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
    else:
        model_path = os.path.join(model_dir, f"{filename_suffix}_{modelo_nome.lower()}_model.joblib")
        joblib.dump(model, model_path)

    joblib.dump(scalers['x'], os.path.join(scaler_dir, f"scaler_x_{filename_suffix}.pkl"))
    joblib.dump(scalers['y'], os.path.join(scaler_dir, f"scaler_y_{filename_suffix}.pkl"))
    print(f"✅ Modelo e scalers salvos em: {model_dir} e {scaler_dir}")

    preds_dir = os.path.join(output_dir, "Previsto", extratora)
    os.makedirs(preds_dir, exist_ok=True)
    preds_path = os.path.join(preds_dir, f"predicoes_{filename_suffix}.csv")

    if cluster_ids is not None:
        predictions_list = []
        for i in range(len(targets)):
            cluster_name = cluster_ids[i]
            real_value = targets.flatten()[i]
            pred_real_value = preds.flatten()[i]

            rmse, err_lit, err_pred, error_perc, approved = None, None, None, None, 'N/A'
            inct_key = f"inct_{parametro.lower()}"
            incerteza_fator = PARAMETROS_AJUSTE.get(cluster_name, {}).get(inct_key)

            rmse = np.abs(pred_real_value - real_value)
            if real_value != 0:
                error_perc = np.abs(((pred_real_value - real_value) / real_value) * 100)
            
            if incerteza_fator:
                err_lit = real_value * (incerteza_fator - 1)
                err_pred = pred_real_value * (incerteza_fator - 1)
                lower_bound = real_value - err_lit
                upper_bound = real_value + err_lit
                approved = 'Yes' if lower_bound <= pred_real_value <= upper_bound else 'No'
            
            predictions_list.append({
                'Cluster': cluster_name,
                f'Literature_{parametro}': real_value,
                f'err_Literature_{parametro}': err_lit,
                f'{extratora.lower()}_{parametro}': pred_real_value,
                f'err_{extratora.lower()}_{parametro}': err_pred,
                'RMSE': rmse,
                'Error_%': error_perc,
                'Approved': approved
            })
        df_preds = pd.DataFrame(predictions_list)
    else:
        df_preds = pd.DataFrame({"True": targets.flatten(), "Predicted": preds.flatten()})

    df_preds.to_csv(preds_path, index=False)
    print(f"✅ Predições detalhadas salvas em: {preds_path}")

    if 'train_losses' in scalers:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.plot(scalers['train_losses'], label="Treinamento", color='#1f77b4', linewidth=2)
        ax.plot(scalers['val_losses'], label="Validação", color='#ff7f0e', linewidth=2, linestyle='--')
        
        ax.set_title(f"Curva de Aprendizado - {parametro} ({extratora} - Split {split_id})", fontsize=18, pad=15)
        ax.set_xlabel("Épocas", fontsize=16)
        ax.set_ylabel("MSE (Loss)", fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=14, loc='upper right')
        
        fig.tight_layout()
        fig_path = os.path.join(model_dir, f"loss_curve_{filename_suffix}.jpg")
        plt.savefig(fig_path, format='jpg', bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Curva de perda salva em: {fig_path}")

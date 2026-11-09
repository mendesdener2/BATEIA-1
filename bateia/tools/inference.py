# bateia/tools/inference.py
import os
import numpy as np
import pandas as pd
import torch
from bateia.config import BASE_DIR, PARAMETROS_AJUSTE

def predict_on_real_clusters(model, scalers, clusters_to_predict, extratora, parametro, output_dir, split_id, device, modelo_nome, gap_label, reshape_dims=None, suffix=".pt"):
    print("\n--- Iniciando Predição nos Aglomerados Reais (PyTorch) ---")
    model.eval()
    real_predictions = []

    for cluster_name in clusters_to_predict:
        try:
            if suffix == ".pt":
                file_name = f"features_real_{cluster_name}.pt"
            else:
                file_name = f"features_real_{cluster_name}{suffix.replace('.pt', '')}.pt"

            feature_path = os.path.join(BASE_DIR, cluster_name, "Features", extratora.upper(), file_name)
            
            if not os.path.exists(feature_path):
                print(f"  AVISO: Feature real não encontrada ({file_name}) para {cluster_name}. Pulando.")
                continue
            
            feature_tensor = torch.load(feature_path, map_location=device, weights_only=True)

            real_value = None
            target_path = os.path.join(BASE_DIR, cluster_name, "Target", f"{parametro.lower()}_otm_{cluster_name}.csv")
            if os.path.exists(target_path):
                try:
                    read_val = pd.read_csv(target_path, header=None).iloc[1, 0]
                    real_value = float(read_val)
                except (ValueError, IndexError) as e:
                    print(f"  AVISO: Não foi possível converter o valor real para float no arquivo {target_path}. Erro: {e}")

            feature_flat = feature_tensor.view(1, -1)
            feature_numpy = feature_flat.cpu().numpy()

            expected_dim = scalers['x'].n_features_in_
            if feature_numpy.shape[1] != expected_dim:
                alt_file_name = f"features_real_{cluster_name}_semgap.pt" if suffix == ".pt" else f"features_real_{cluster_name}.pt"
                alt_feature_path = os.path.join(BASE_DIR, cluster_name, "Features", extratora.upper(), alt_file_name)
                if os.path.exists(alt_feature_path):
                    alt_tensor = torch.load(alt_feature_path, map_location=device, weights_only=True)
                    alt_flat = alt_tensor.view(1, -1)
                    if alt_flat.shape[1] == expected_dim:
                        feature_tensor = alt_tensor
                        feature_flat = alt_flat
                        feature_numpy = alt_flat.cpu().numpy()
                        print(f"  AVISO: Fallback dimensional ativado para {cluster_name}. Arquivo alternativo carregado.")
                    else:
                        print(f"  ERRO: Nenhuma feature real de {cluster_name} corresponde à dimensão esperada ({expected_dim}).")
                        continue
                else:
                    print(f"  ERRO: Dimensão incompatível ({feature_numpy.shape[1]} vs {expected_dim}) e fallback não encontrado para {cluster_name}.")
                    continue

            feature_scaled_numpy = scalers['x'].transform(feature_numpy)
            feature_scaled = torch.from_numpy(feature_scaled_numpy).to(torch.float32).to(device)

            if reshape_dims:
                feature_scaled = feature_scaled.view(feature_scaled.size(0), *reshape_dims)

            with torch.no_grad():
                pred_scaled = model(feature_scaled)
            
            pred_numpy_scaled = pred_scaled.cpu().numpy().reshape(-1, 1)
            pred_real_value = np.abs(scalers['y'].inverse_transform(pred_numpy_scaled)[0][0])
            
            rmse, err_lit, err_pred, error_perc, approved = None, None, None, None, 'N/A'
            inct_key = f"inct_{parametro.lower()}"
            incerteza_fator = PARAMETROS_AJUSTE.get(cluster_name, {}).get(inct_key)

            if real_value is not None:
                rmse = np.abs(pred_real_value - real_value)
                if real_value != 0:
                    error_perc = np.abs(((pred_real_value - real_value) / real_value) * 100)
                
                if incerteza_fator:
                    err_lit = real_value * (incerteza_fator - 1)
                    err_pred = pred_real_value * (incerteza_fator - 1)
                    lower_bound = real_value - err_lit
                    upper_bound = real_value + err_lit
                    approved = 'Yes' if lower_bound <= pred_real_value <= upper_bound else 'No'
                else:
                    print(f"  AVISO: Fator de incerteza '{inct_key}' não encontrado para {cluster_name}.")
            
            real_predictions.append({
                'Cluster': cluster_name,
                f'Literature_{parametro}': real_value,
                f'err_Literature_{parametro}': err_lit,
                f'{extratora.lower()}_{parametro}': pred_real_value,
                f'err_{extratora.lower()}_{parametro}': err_pred,
                'RMSE': rmse,
                'Error_%': error_perc,
                'Approved': approved
            })

            output_indiv_path = os.path.join(BASE_DIR, cluster_name, f"{parametro.lower()}_otm_{cluster_name}.csv")
            pd.DataFrame([pred_real_value]).to_csv(output_indiv_path, header=False, index=False)
            print(f"  ✅ Predição para '{cluster_name}': {pred_real_value:.6f}")

        except Exception as e:
            print(f"  ERRO ao processar predição real para {cluster_name}: {e}")

    if real_predictions:
        df_real_preds = pd.DataFrame(real_predictions)
        
        if str(split_id).lower() == "bm":
            filename_suffix = f"{parametro.lower()}_{extratora.lower()}_bm_{gap_label.lower()}"
        else:
            filename_suffix = f"{parametro.lower()}_{extratora.lower()}_{modelo_nome.lower()}_{split_id}_{gap_label.lower()}"

        preds_dir = os.path.join(output_dir, "Previsto", extratora)
        consolidated_path = os.path.join(preds_dir, f"predicoes_reais_{filename_suffix}.csv")
        df_real_preds.to_csv(consolidated_path, index=False)
        print(f"✅ Arquivo consolidado de predições reais salvo em: {consolidated_path}")
    print("-" * 50)
    
def predict_on_real_clusters_sklearn(model, scalers, clusters_to_predict, extratora, parametro, output_dir, split_id, modelo_nome, gap_label, suffix=".pt"):
    print("\n--- Iniciando Predição nos Aglomerados Reais (Scikit-learn) ---")
    real_predictions = []

    for cluster_name in clusters_to_predict:
        try:
            if suffix == ".pt":
                file_name = f"features_real_{cluster_name}.pt"
            else:
                file_name = f"features_real_{cluster_name}{suffix.replace('.pt', '')}.pt"

            feature_path = os.path.join(
                BASE_DIR, cluster_name, "Features", extratora.upper(), file_name
            )
            if not os.path.exists(feature_path):
                print(f"  AVISO: Feature real não encontrada ({file_name}) para {cluster_name}. Pulando.")
                continue

            feature_tensor = torch.load(feature_path, map_location='cpu', weights_only=True)
            feature_flat = feature_tensor.view(1, -1)
            feature_numpy = feature_flat.cpu().numpy()

            expected_dim = scalers['x'].n_features_in_
            if feature_numpy.shape[1] != expected_dim:
                alt_file_name = f"features_real_{cluster_name}_semgap.pt" if suffix == ".pt" else f"features_real_{cluster_name}.pt"
                alt_feature_path = os.path.join(BASE_DIR, cluster_name, "Features", extratora.upper(), alt_file_name)
                if os.path.exists(alt_feature_path):
                    alt_tensor = torch.load(alt_feature_path, map_location='cpu', weights_only=True)
                    alt_flat = alt_tensor.view(1, -1)
                    if alt_flat.shape[1] == expected_dim:
                        feature_tensor = alt_tensor
                        feature_flat = alt_flat
                        feature_numpy = alt_flat.cpu().numpy()
                        print(f"  AVISO: Fallback dimensional ativado para {cluster_name}. Arquivo alternativo carregado.")
                    else:
                        print(f"  ERRO: {cluster_name}: Nenhuma feature corresponde à dimensão esperada ({expected_dim}).")
                        continue
                else:
                    print(f"  ERRO: {cluster_name}: {feature_numpy.shape[1]} features, esperado {expected_dim} e fallback não encontrado.")
                    continue

            feature_scaled_numpy = scalers['x'].transform(feature_numpy)
            pred_scaled = model.predict(feature_scaled_numpy)

            pred_numpy_scaled = pred_scaled.reshape(-1, 1)
            pred_real_value = np.abs(scalers['y'].inverse_transform(pred_numpy_scaled)[0][0])

            target_path = os.path.join(BASE_DIR, cluster_name, "Target", f"mat_{parametro.lower()}_otm_{cluster_name}.csv")
            real_value = None
            if os.path.exists(target_path):
                try:
                    df_target = pd.read_csv(target_path, header=None)
                    first_val = df_target.iloc[0, 0]
                    if isinstance(first_val, str) and not first_val.replace('.', '', 1).isdigit():
                        first_val = df_target.iloc[1, 0]
                    real_value = float(first_val)
                except Exception as e:
                    print(f"  AVISO: Erro ao ler valor real de {target_path}: {e}")

            rmse = err_lit = err_pred = error_perc = None
            approved = 'N/A'
            inct_key = f"inct_{parametro.lower()}"
            incerteza_fator = PARAMETROS_AJUSTE.get(cluster_name, {}).get(inct_key)

            if real_value is not None:
                rmse = np.abs(pred_real_value - real_value)
                if real_value != 0:
                    error_perc = np.abs(((pred_real_value - real_value) / real_value) * 100)

                if incerteza_fator:
                    err_lit = real_value * (incerteza_fator - 1)
                    err_pred = pred_real_value * (incerteza_fator - 1)
                    lower_bound = real_value - err_lit
                    upper_bound = real_value + err_lit
                    approved = 'Yes' if lower_bound <= pred_real_value <= upper_bound else 'No'
                else:
                    print(f"  AVISO: Fator de incerteza '{inct_key}' não encontrado para {cluster_name}.")

            real_predictions.append({
                'Cluster': cluster_name,
                f'Literature_{parametro}': real_value,
                f'err_Literature_{parametro}': err_lit,
                f'{extratora.lower()}_{parametro}': pred_real_value,
                f'err_{extratora.lower()}_{parametro}': err_pred,
                'RMSE': rmse,
                'Error_%': error_perc,
                'Approved': approved
            })

            output_indiv_path = os.path.join(BASE_DIR, cluster_name, f"{parametro.lower()}_otm_{cluster_name}.csv")
            pd.DataFrame([pred_real_value]).to_csv(output_indiv_path, header=False, index=False)
            print(f"  ✅ Predição para '{cluster_name}': {pred_real_value:.6f}")

        except Exception as e:
            print(f"  ERRO ao processar predição real para {cluster_name}: {e}")

    if real_predictions:
        df_real_preds = pd.DataFrame(real_predictions)
        if str(split_id).lower() == "bm":
            filename_suffix = f"{parametro.lower()}_{extratora.lower()}_bm_{gap_label.lower()}"
        else:
            filename_suffix = f"{parametro.lower()}_{extratora.lower()}_{modelo_nome.lower()}_{split_id}_{gap_label.lower()}"
        preds_dir = os.path.join(output_dir, "Previsto", extratora)
        os.makedirs(preds_dir, exist_ok=True)
        consolidated_path = os.path.join(preds_dir, f"predicoes_reais_{filename_suffix}.csv")
        df_real_preds.to_csv(consolidated_path, index=False)
        print(f"✅ Arquivo consolidado de predições reais salvo em: {consolidated_path}")

    print("-" * 50)

# bateia/tools/trainer_backend.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import optuna
from sklearn.metrics import mean_squared_error

def train_model_pytorch(model, optimizer, train_loader, val_loader, epochs, device, patience, min_delta, trial=None):
    criterion = nn.MSELoss()
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    if trial:
        desc = f"Trial {trial.number:<3} (Optuna)"
    else:
        desc = "Treino Final"
    
    epoch_iterator = tqdm(range(epochs), desc=desc, leave=False, dynamic_ncols=True, unit="época")

    for epoch in epoch_iterator: 
        model.train()
        train_loss_epoch = 0
        
        for x, y in train_loader: 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * x.size(0)
        
        train_losses.append(train_loss_epoch / len(train_loader.dataset))

        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss_epoch += criterion(model(x).squeeze(), y.squeeze()).item() * x.size(0)
        
        val_loss = val_loss_epoch / len(val_loader.dataset)
        val_rmse = np.sqrt(val_loss)
        val_losses.append(val_loss)

        epoch_iterator.set_postfix(val_rmse=f"{val_rmse:.6f}")

        if trial:
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                epoch_iterator.close()
                raise optuna.TrialPruned()
        
        if val_rmse < (best_val_rmse - min_delta):
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if not trial:
                epoch_iterator.set_description(f"{desc} (Early Stop)")
            break
            
    epoch_iterator.close() 
    
    if trial:
        return best_val_rmse
    
    print(f"  -> Treinamento final concluído. Melhor Val RMSE: {best_val_rmse:.6f}")
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def test_model_pytorch(model, test_loader, device, scaler_y, reshape_dims=None):
    model.eval()
    model.to(device)
    all_preds_original_scale, all_targets_original_scale = [], []
    
    with torch.no_grad():
        for x, y_scaled in test_loader:
            if reshape_dims:
                x = x.view(x.size(0), *reshape_dims)
            x = x.to(device)
            
            preds_scaled = model(x).cpu()
            
            preds_reshaped = preds_scaled.detach().numpy().reshape(-1, 1)
            targets_reshaped = y_scaled.detach().numpy().reshape(-1, 1)
            
            preds_original = np.abs(scaler_y.inverse_transform(preds_reshaped))
            targets_original = scaler_y.inverse_transform(targets_reshaped)
            
            all_preds_original_scale.extend(preds_original.flatten())
            all_targets_original_scale.extend(targets_original.flatten())
            
    mse = mean_squared_error(all_targets_original_scale, all_preds_original_scale)
    rmse = np.sqrt(mse)
    return rmse, np.array(all_preds_original_scale), np.array(all_targets_original_scale)

def gerar_relatorio_consistencia(model, X_val, y_val, cluster_ids_val, scaler_x, scaler_y, device, reshape_dims=None):
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        return x

    print("\n--- Relatório de Consistência na Validação (RMSE) ---")
    if len(np.unique(cluster_ids_val)) < 2:
        print("  (Apenas um cluster no conjunto de validação, relatório de consistência pulado)")
        return

    X_val_numpy = to_numpy(X_val)
    if len(X_val_numpy.shape) > 2:
        X_val_flat = X_val_numpy.reshape(X_val_numpy.shape[0], -1)
    else:
        X_val_flat = X_val_numpy

    X_val_s = scaler_x.transform(X_val_flat)

    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)
        X_val_tensor = torch.from_numpy(X_val_s).to(torch.float32)
        if reshape_dims:
            X_val_tensor = X_val_tensor.view(X_val_tensor.size(0), *reshape_dims)
        with torch.no_grad():
            preds_s = model(X_val_tensor.to(device)).detach().cpu().numpy()
    else:
        preds_s = model.predict(X_val_s)

    preds_real = np.abs(scaler_y.inverse_transform(preds_s.reshape(-1, 1)))
    
    y_val = to_numpy(y_val)
    cluster_ids_val = to_numpy(cluster_ids_val)

    df_val = pd.DataFrame({
        'cluster': cluster_ids_val,
        'rmse': np.abs(preds_real.flatten() - y_val.flatten())
    })

    report = df_val.groupby('cluster')['rmse'].agg(['mean', 'std', 'count'])

    cluster_rmses = []
    for cluster, row in report.iterrows():
        cluster_rmses.append(row['mean'])
        print(f"- Cluster '{cluster}': RMSE Médio = {row['mean']:.6f} ({int(row['count'])} amostras)")
    
    if len(cluster_rmses) > 1:
        print(f"- Desvio Padrão dos RMSEs: {np.std(cluster_rmses):.6f} (menor é mais consistente)")
    print("-" * 50)

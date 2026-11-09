# bateia/models/mlp_model.py
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bateia.config import OPTUNA_HYPERPARAMETERS, TRAINING_CONSTANTS, PARAMS_CONFIG
from bateia.tools.dl_utils import train_model_pytorch

from bateia.models.core_trainer import executar_treino_pytorch_generico
from bateia.models.architectures import MLPRegressor

def executar_treino_mlp(metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix=".pt"):
    def factory_mlp(features_per_step, best_params, input_size_flat):
        arch = best_params.get("architecture", "Media")
        hidden_sizes = [128, 64] if arch == "Pequena" else [256, 128, 64] if arch == "Media" else [512, 256, 128]
        if "hidden_size_base" in best_params: 
             hs = best_params["hidden_size_base"]
             hidden_sizes = [hs, hs//2] if hs==128 else [hs, hs//2, hs//4]
        return MLPRegressor(input_size_flat, hidden_sizes, output_size=1, dropout=best_params["dropout"])

    executar_treino_pytorch_generico(
        "MLP", _objective_mlp, factory_mlp, 
        metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix
    )

def _objective_mlp(trial, parametro, data, device, suffix):
    """Função objective do Optuna para MLP."""
    hp_key = f"MLP_{parametro}"
    if hp_key not in OPTUNA_HYPERPARAMETERS: hp_key = "MLP_f0"
    hp = OPTUNA_HYPERPARAMETERS[hp_key]
    
    train_X_s, train_y_s = data["train"]
    val_X_s, val_y_s = data["val"]
    input_size = train_X_s.shape[1]

    lr = trial.suggest_float("lr", hp["lr"]["low"], hp["lr"]["high"], log=hp["lr"]["log"])
    dropout = trial.suggest_float("dropout", hp["dropout"]["low"], hp["dropout"]["high"])
    
    # Arquitetura via hidden_size_base (compatível com os intervalos fornecidos anteriormente)
    # Se architecture não estiver no dict, usamos um fallback lógico
    if "architecture" in hp:
        arch = trial.suggest_categorical("architecture", hp["architecture"]["choices"])
        hidden_sizes = [128, 64] if arch == "Pequena" else [256, 128, 64] if arch == "Media" else [512, 256, 128]
    else:
        # Fallback genérico para MLP se não especificado
        hs = trial.suggest_categorical("hidden_size_base", [128, 256, 512])
        hidden_sizes = [hs, hs//2] if hs==128 else [hs, hs//2, hs//4]

    batch_size = trial.suggest_categorical("batch_size", hp["batch_size"]["choices"])

    train_loader = DataLoader(TensorDataset(train_X_s, train_y_s), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_X_s, val_y_s), batch_size=batch_size, num_workers=0, pin_memory=True)

    model = MLPRegressor(input_size, hidden_sizes, output_size=1, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    val_rmse = train_model_pytorch(
        model, optimizer, train_loader, val_loader, 
        TRAINING_CONSTANTS['optuna_epochs'], device, 
        PARAMS_CONFIG[parametro]["patience"], PARAMS_CONFIG[parametro]["min_delta"], trial
    )
    return val_rmse

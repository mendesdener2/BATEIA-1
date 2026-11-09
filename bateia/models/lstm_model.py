# bateia/models/lstm_model.py
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bateia.config import (
    OPTUNA_HYPERPARAMETERS, TRAINING_CONSTANTS, PARAMS_CONFIG,
    RESHAPE_CONFIG, RESHAPE_CONFIG_SEQ
)
from bateia.tools.dl_utils import train_model_pytorch

from bateia.models.core_trainer import executar_treino_pytorch_generico
from bateia.models.architectures import LSTMRegressor

def executar_treino_lstm(metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix=".pt"):
    def factory_lstm(features_per_step, best_params, input_size_flat):
        return LSTMRegressor(features_per_step, best_params["hidden_size"], best_params["num_layers"], best_params["n_dense_layers"], best_params["dense_neurons"], best_params["dropout"])

    executar_treino_pytorch_generico(
        "LSTM", _objective_lstm, factory_lstm, 
        metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix
    )

def _objective_lstm(trial, parametro, data, device, suffix):
    hp_key = f"LSTM_{parametro}"
    if hp_key not in OPTUNA_HYPERPARAMETERS: hp_key = "LSTM_f0"
    hp = OPTUNA_HYPERPARAMETERS[hp_key]

    train_X_s, train_y_s = data["train"]
    val_X_s, val_y_s = data["val"]
    input_size_flat = train_X_s.shape[1]
    
    config_dict = RESHAPE_CONFIG_SEQ if "semgap" in suffix else RESHAPE_CONFIG
    if input_size_flat not in config_dict: 
        raise ValueError(f"Tamanho {input_size_flat} não encontrado em {'RESHAPE_CONFIG_SEQ' if 'semgap' in suffix else 'RESHAPE_CONFIG'}.")
    seq_len, features_per_step = config_dict[input_size_flat]
    
    train_X_reshaped = train_X_s.view(train_X_s.size(0), seq_len, features_per_step)
    val_X_reshaped = val_X_s.view(val_X_s.size(0), seq_len, features_per_step)

    lr = trial.suggest_float("lr", hp["lr"]["low"], hp["lr"]["high"], log=hp["lr"]["log"])
    dropout = trial.suggest_float("dropout", hp["dropout"]["low"], hp["dropout"]["high"])
    batch_size = trial.suggest_categorical("batch_size", hp["batch_size"]["choices"])
    num_layers = trial.suggest_int("num_layers", hp["num_layers"]["low"], hp["num_layers"]["high"])
    hidden_size = trial.suggest_categorical("hidden_size", hp["hidden_size"]["choices"])
    n_dense_layers = trial.suggest_int("n_dense_layers", hp["n_dense_layers"]["low"], hp["n_dense_layers"]["high"])
    dense_neurons = trial.suggest_categorical("dense_neurons", hp["dense_neurons"]["choices"])

    train_loader = DataLoader(TensorDataset(train_X_reshaped, train_y_s), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_X_reshaped, val_y_s), batch_size=batch_size, num_workers=0, pin_memory=True)

    model = LSTMRegressor(features_per_step, hidden_size, num_layers, n_dense_layers, dense_neurons, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    val_rmse = train_model_pytorch(model, optimizer, train_loader, val_loader, TRAINING_CONSTANTS['optuna_epochs'], device, PARAMS_CONFIG[parametro]["patience"], PARAMS_CONFIG[parametro]["min_delta"], trial)
    return val_rmse

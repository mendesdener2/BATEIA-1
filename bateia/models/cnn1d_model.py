# bateia/models/cnn1d_model.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bateia.config import (
    OPTUNA_HYPERPARAMETERS, TRAINING_CONSTANTS, PARAMS_CONFIG,
    RESHAPE_CONFIG, RESHAPE_CONFIG_SEQ
)
from bateia.tools.dl_utils import train_model_pytorch

from bateia.models.core_trainer import executar_treino_pytorch_generico
from bateia.models.architectures import CNN1DRegressor

def executar_treino_1dcnn(metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix=".pt"):
    def factory_1dcnn(features_per_step, best_params, input_size_flat):
        # seq_len para 1DCNN pode ser obtida do RESHAPE_CONFIG no core_trainer, mas a classe precisa.
        # A forma que fizemos o factory, ela recebe (features_per_step, best_params, input_size_flat).
        # Vamos re-derivar seq_len aqui.
        from bateia.config import RESHAPE_CONFIG, RESHAPE_CONFIG_SEQ
        config_dict = RESHAPE_CONFIG_SEQ if "semgap" in suffix else RESHAPE_CONFIG
        seq_len, _ = config_dict[input_size_flat]

        return CNN1DRegressor(
            input_channels=features_per_step, seq_length=seq_len, 
            n_conv_layers=best_params["n_conv_layers"], out_channels=best_params["out_channels"], 
            kernel_size=best_params["kernel_size"], n_dense_layers=best_params["n_dense_layers"], 
            dense_neurons=best_params["dense_neurons"], dropout=best_params["dropout"]
        )

    executar_treino_pytorch_generico(
        "1DCNN", _objective_1dcnn, factory_1dcnn, 
        metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix
    )

def _objective_1dcnn(trial, parametro, data, device, suffix):
    hp_key = f"1DCNN_{parametro}"
    if hp_key not in OPTUNA_HYPERPARAMETERS: hp_key = "1DCNN_f0"
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
    n_conv_layers = trial.suggest_int("n_conv_layers", hp["n_conv_layers"]["low"], hp["n_conv_layers"]["high"])
    out_channels = trial.suggest_categorical("out_channels", hp["out_channels"]["choices"])
    kernel_size = trial.suggest_categorical("kernel_size", hp["kernel_size"]["choices"])
    n_dense_layers = trial.suggest_int("n_dense_layers", hp["n_dense_layers"]["low"], hp["n_dense_layers"]["high"])
    dense_neurons = trial.suggest_categorical("dense_neurons", hp["dense_neurons"]["choices"])
    
    train_loader = DataLoader(TensorDataset(train_X_reshaped, train_y_s), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_X_reshaped, val_y_s), batch_size=batch_size, num_workers=0, pin_memory=True)
    
    model = CNN1DRegressor(
        input_channels=features_per_step, seq_length=seq_len, 
        n_conv_layers=n_conv_layers, out_channels=out_channels, 
        kernel_size=kernel_size, n_dense_layers=n_dense_layers, 
        dense_neurons=dense_neurons, dropout=dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    val_rmse = train_model_pytorch(model, optimizer, train_loader, val_loader, TRAINING_CONSTANTS['optuna_epochs'], device, PARAMS_CONFIG[parametro]["patience"], PARAMS_CONFIG[parametro]["min_delta"], trial)
    
    return val_rmse

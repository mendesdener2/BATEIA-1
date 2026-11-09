OPTUNA_HYPERPARAMETERS = {
    # --- PERFIS DE TUNING PARA MLP ---
    "MLP_f0": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.2, "high": 0.4, "log": False},
        "optimizer": {"type": "categorical", "choices": ["AdamW"]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "MLP_Rc": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "optimizer": {"type": "categorical", "choices": ["AdamW"]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "MLP_Rt": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "optimizer": {"type": "categorical", "choices": ["AdamW"]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "MLP_fbg": {
        "lr": {"type": "float", "low": 1e-5, "high": 5e-5, "log": True},
        "dropout": {"type": "float", "low": 0.4, "high": 0.6, "log": False},
        "optimizer": {"type": "categorical", "choices": ["AdamW"]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },

    # --- SVR (PERFIL ÚNICO "CERCADO") ---
    "SVR": {
        "C": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
        "gamma": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
        "kernel": {"type": "categorical", "choices": ["rbf", "poly"]},
        "degree": {"type": "int", "low": 2, "high": 5},
        "epsilon": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True}
    },
    
    # --- RF (PERFIL ÚNICO "CERCADO") ---
    "RandomForest": {
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "max_depth": {"type": "int", "low": 5, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
        "min_samples_leaf": {"type": "int", "low": 2, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]}
    },
    
    # --- PERFIS DE TUNING PARA LSTM ---
    "LSTM_f0": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.2, "high": 0.4, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "LSTM_Rc": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "LSTM_Rt": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "LSTM_fbg": {
        "lr": {"type": "float", "low": 1e-5, "high": 5e-5, "log": True},
        "dropout": {"type": "float", "low": 0.4, "high": 0.6, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    
    # --- PERFIS DE TUNING PARA GRU ---
    "GRU_f0": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.2, "high": 0.4, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "GRU_Rc": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "GRU_Rt": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "GRU_fbg": {
        "lr": {"type": "float", "low": 1e-5, "high": 5e-5, "log": True},
        "dropout": {"type": "float", "low": 0.4, "high": 0.6, "log": False},
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    
    # --- PERFIS DE TUNING PARA 1DCNN ---
    "1DCNN_f0": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.2, "high": 0.4, "log": False},
        "n_conv_layers": {"type": "int", "low": 1, "high": 4},
        "out_channels": {"type": "categorical", "choices": [32, 64, 128]},
        "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "1DCNN_Rc": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "n_conv_layers": {"type": "int", "low": 1, "high": 4},
        "out_channels": {"type": "categorical", "choices": [32, 64, 128]},
        "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "1DCNN_Rt": {
        "lr": {"type": "float", "low": 1e-6, "high": 5e-6, "log": True},
        "dropout": {"type": "float", "low": 0.05, "high": 0.2, "log": False},
        "n_conv_layers": {"type": "int", "low": 1, "high": 4},
        "out_channels": {"type": "categorical", "choices": [32, 64, 128]},
        "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    },
    "1DCNN_fbg": {
        "lr": {"type": "float", "low": 1e-5, "high": 5e-5, "log": True},
        "dropout": {"type": "float", "low": 0.4, "high": 0.6, "log": False},
        "n_conv_layers": {"type": "int", "low": 1, "high": 4},
        "out_channels": {"type": "categorical", "choices": [32, 64, 128]},
        "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
        "n_dense_layers": {"type": "int", "low": 1, "high": 3},
        "dense_neurons": {"type": "categorical", "choices": [64, 128, 256]},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
    }
}

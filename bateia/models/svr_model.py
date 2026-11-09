# bateia/models/svr_model.py
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from bateia.config import OPTUNA_HYPERPARAMETERS

def _objective_svr(trial, parametro, data):
    hp = OPTUNA_HYPERPARAMETERS["SVR"]
    train_X_s, train_y_s = data["train"]
    val_X_s, val_y_s = data["val"]

    c = trial.suggest_float("C", hp["C"]["low"], hp["C"]["high"], log=hp["C"]["log"])
    epsilon = trial.suggest_float("epsilon", hp["epsilon"]["low"], hp["epsilon"]["high"], log=hp["epsilon"]["log"])
    gamma = trial.suggest_float("gamma", hp["gamma"]["low"], hp["gamma"]["high"], log=hp["gamma"]["log"])
    kernel = trial.suggest_categorical("kernel", hp["kernel"]["choices"])
    degree = trial.suggest_int("degree", hp["degree"]["low"], hp["degree"]["high"]) if kernel == 'poly' else 3

    model = SVR(C=c, epsilon=epsilon, gamma=gamma, kernel=kernel, degree=degree)
    
    if len(train_X_s) > 5000:
        print(f"  [AVISO Optuna] SVR com {len(train_X_s)} amostras sem limite. Trial pode ser lento.")
        
    model.fit(train_X_s, train_y_s.ravel())
    preds = model.predict(val_X_s)
    return np.sqrt(mean_squared_error(val_y_s, preds))

from bateia.models.core_trainer_sklearn import executar_treino_sklearn_generico

def executar_treino_svr(metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix=".pt"):
    def factory_svr(best_params):
        return SVR(**best_params)

    executar_treino_sklearn_generico(
        "SVR", _objective_svr, factory_svr, 
        metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix
    )

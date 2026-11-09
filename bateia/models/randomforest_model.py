# bateia/models/randomforest_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from bateia.config import OPTUNA_HYPERPARAMETERS

def _objective_rf(trial, parametro, data):
    hp = OPTUNA_HYPERPARAMETERS["RandomForest"]
    train_X_s, train_y_s = data["train"]
    val_X_s, val_y_s = data["val"]

    n_estimators = trial.suggest_int("n_estimators", hp["n_estimators"]["low"], hp["n_estimators"]["high"])
    max_depth = trial.suggest_int("max_depth", hp["max_depth"]["low"], hp["max_depth"]["high"])
    min_samples_split = trial.suggest_int("min_samples_split", hp["min_samples_split"]["low"], hp["min_samples_split"]["high"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", hp["min_samples_leaf"]["low"], hp["min_samples_leaf"]["high"])
    max_features = trial.suggest_categorical("max_features", hp["max_features"]["choices"])

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=-1, random_state=42
    )
    idx = np.random.choice(len(train_X_s), size=min(5000, len(train_X_s)), replace=False)
    model.fit(train_X_s[idx], train_y_s[idx].ravel())
    preds = model.predict(val_X_s)
    return np.sqrt(mean_squared_error(val_y_s, preds))

from bateia.models.core_trainer_sklearn import executar_treino_sklearn_generico

def executar_treino_random_forest(metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix=".pt"):
    def factory_rf(best_params):
        return RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)

    executar_treino_sklearn_generico(
        "RandomForest", _objective_rf, factory_rf, 
        metodologia, parametro, extratora, data_split_config, config_geral, output_dir, gap_label, suffix
    )

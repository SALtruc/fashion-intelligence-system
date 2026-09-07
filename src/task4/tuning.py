import optuna

from src.task4.config import HISTORY_DIR, SEED


def suggest_parameters(trial: optuna.Trial, model_name: str):
    parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
    }

    if model_name == "triplet":
        parameters["margin"] = trial.suggest_float("margin", 0.1, 1.0, step=0.1)
    elif model_name == "supcon":
        parameters["temperature"] = trial.suggest_float(
            "temperature", 0.05, 0.20, step=0.025
        )
    elif model_name == "multi_similarity":
        parameters["alpha"] = trial.suggest_float("alpha", 1.0, 3.0)
        parameters["beta"] = trial.suggest_float("beta", 20.0, 80.0)
        parameters["base"] = trial.suggest_float("base", 0.3, 0.7)
    elif model_name == "arcface":
        parameters["margin"] = trial.suggest_float("margin", 0.1, 0.5)
        parameters["scale"] = trial.suggest_categorical("scale", [16, 32, 48, 64])
    return parameters


def create_study(study_name: str):
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{HISTORY_DIR.resolve().as_posix()}/{study_name}_optuna.db",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

import optuna
from optuna.trial import TrialState

from src.task4.config import HISTORY_DIR, SEED


def study_storage_url(study_name: str) -> str:
    """Return the SQLite URL used to persist one Task 4 Optuna study."""
    database_path = HISTORY_DIR / f"{study_name}_optuna.db"
    return f"sqlite:///{database_path.resolve().as_posix()}"


def load_best_params(study_name: str) -> tuple[dict[str, object], float]:
    """Load the best completed trial from a persisted Optuna study.

    This never invokes ``Study.optimize``. It fails early when the expected
    database is absent or contains no completed trials, preventing accidental
    final training with undefined hyperparameters.
    """
    database_path = HISTORY_DIR / f"{study_name}_optuna.db"
    if not database_path.exists():
        raise FileNotFoundError(
            f"No Optuna database found for {study_name!r}: {database_path}. "
            "Run the tuning section once to create it."
        )

    study = optuna.load_study(
        study_name=study_name,
        storage=study_storage_url(study_name),
    )
    if not any(trial.state == TrialState.COMPLETE for trial in study.trials):
        raise ValueError(f"Optuna study {study_name!r} has no completed trials")

    return study.best_params, study.best_value


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
        storage=study_storage_url(study_name),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

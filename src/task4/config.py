from pathlib import Path

SEED = 42
RESNET_INPUT_SIZE = (128, 128)
INPUT_SIZE = RESNET_INPUT_SIZE

LABEL_COLUMN = "articleType_gender"
LABEL_ID_COLUMN = "articleType_gender_id"

MIN_CLASS_SIZE = 5
VAL_FRACTION = 0.10
CLASSES_PER_BATCH = 16
IMAGES_PER_CLASS = 4
CAE_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
NUM_WORKERS = 0

N_TRIALS = 20
TUNING_EPOCHS = 15
FINAL_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "preprocessed_datasets" / "train" / "styles_train.csv"
IMAGE_DIR = PROJECT_ROOT / "preprocessed_datasets" / "train" / "images_train"
TEST_IMAGE_DIR = PROJECT_ROOT / "preprocessed_datasets" / "test" / "images_test"
SPLIT_DIR = PROJECT_ROOT / "splits" / "task4"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "task4"

CONFIG_DIR = ARTIFACT_DIR / "configs"
MODEL_DIR = ARTIFACT_DIR / "models"
OPTUNA_DB = ARTIFACT_DIR / "optuna.db"

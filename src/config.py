from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "figures"

DATA_PATH = DATA_DIR / "cardiovascular_disease.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/kinir/catboost-with-pipelines/"
    "master/cardiovascular-disease-dataset/original/cardio_train.csv"
)
DATASET_NAME = "Cardiovascular Disease (Sulanova et al., ~70k patients)"

METRICS_PATH = REPORTS_DIR / "metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
BEST_PARAMS_PATH = MODELS_DIR / "best_hyperparameters.json"

TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TUNE_SAMPLE_SIZE = 30_000
TUNE_N_ITER = 18
TUNE_CV_FOLDS = 3

FEATURE_COLUMNS = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

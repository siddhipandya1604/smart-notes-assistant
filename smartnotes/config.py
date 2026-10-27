"""Central paths and constants shared by training, the API and the tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "notes_model.joblib"

TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
OUT_OF_SCOPE_CSV = DATA_DIR / "out_of_scope_examples.csv"
STARTER_NOTES_JSON = DATA_DIR / "starter_notes.json"

LABELS = ("Study", "Money", "Travel", "Shopping")
MAX_TEXT_CHARS = 1000
RANDOM_STATE = 42

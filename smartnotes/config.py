"""Central paths and constants shared by training, the API and the tests."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "notes_model.joblib"
DB_PATH = Path(os.environ.get("SMARTNOTES_DB", DATA_DIR / "notes.db"))  # override to run a separate database

TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
OUT_OF_SCOPE_CSV = DATA_DIR / "out_of_scope_examples.csv"
STARTER_NOTES_JSON = DATA_DIR / "starter_notes.json"

FIRST_USER_NOTE_NUMBER = 13  # N001-N012 are reserved for data/starter_notes.json

LABELS = ("Study", "Money", "Travel", "Shopping")
MAX_TEXT_CHARS = 1000
RANDOM_STATE = 42

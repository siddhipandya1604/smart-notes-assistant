"""Central paths and constants shared by training, the API and the tests."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "notes_model.joblib"
DB_PATH = Path(os.environ.get("SMARTNOTES_DB", DATA_DIR / "notes.db"))  # override to run a separate database

FRONTEND_DIST = ROOT / "frontend" / "dist"  # created by `npm run build`; served at / when present

EXTENDED_MODEL_PATH = ROOT / "models" / "extended_model.joblib"
EXTRA_NOTES_CSV = DATA_DIR / "extended_notes.csv"      # extra labelled notes, incl. Health/Family/Home/Work
EXTENDED_EVAL_CSV = DATA_DIR / "extended_eval.csv"     # hand-labelled notes the extended model never trains on
EXTENDED_FINAL_CSV = DATA_DIR / "extended_eval_final.csv"  # written after the training data was frozen; read once

TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
OUT_OF_SCOPE_CSV = DATA_DIR / "out_of_scope_examples.csv"
STARTER_NOTES_JSON = DATA_DIR / "starter_notes.json"

FIRST_USER_NOTE_NUMBER = 13  # N001-N012 are reserved for data/starter_notes.json

LABELS = ("Study", "Money", "Travel", "Shopping")
MAX_TEXT_CHARS = 1000
RANDOM_STATE = 42

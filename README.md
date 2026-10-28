# Smart Notes Assistant

Save short notes, have a **model I trained** sort them into **Study / Money / Travel / Shopping**, get an honest **"Uncertain"** badge on notes that fit no category, and search them later by keyword or by a short question.

| | |
| --- | --- |
| Classifier | TF-IDF (word + character n-grams) into Multinomial Naive Bayes, trained on `data/train.csv` with scikit-learn |
| Validation | accuracy **0.920**, macro-F1 **0.924** (baseline keyword rules: 0.560 / 0.529) |
| Labels beyond the four | a second model also knows **Health, Family, Home, Work**, so "gym at 6 PM" is Health, not a bad Money guess: 22 of 27 held-out notes labelled correctly (81%) |
| Uncertainty | the four-label model's top probability below a threshold (0.8971), combined with the extended model's second opinion: 8% of validation notes flagged, 11 of 12 notes outside the four categories flagged on the final set |
| Search | hybrid of TF-IDF and the classifier's topic sense: hit rate @3 **0.86** vs **0.57** for a keyword baseline |
| Stack | Python + FastAPI + SQLite, React (Vite) UI |

Full numbers, error analysis and limitations: [RESULTS.md](RESULTS.md). How AI tools were used: [AI_WORKFLOW.md](AI_WORKFLOW.md). Original brief: [docs/assignment/TASK.md](docs/assignment/TASK.md).

## Requirements
- Python 3.12 (developed and tested on 3.12.5)
- Node.js 18+ and npm, **only** to build the web UI. Without Node the API and its interactive docs at `/docs` still work.
- No paid service, API key, GPU or network access is needed at run time.

## Quick start

```bash
# 1. install Python dependencies (pinned in requirements.txt)
pip install -r requirements.txt

# 2. build the web UI (once)
cd frontend && npm install && npm run build && cd ..

# 3. start the app: API and UI in one process
python -m uvicorn smartnotes.api.app:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete` (about 7 seconds: importing scikit-learn and loading the model). Then open **http://127.0.0.1:8000**. Requests take about 10-40 ms once it is up.

> **Port already in use?** Any port works; pass a different `--port` (for example `8765`) and use it in the URLs below. On the machine this was developed on, port 8000 was held by another program.

The trained model is committed (`models/notes_model.joblib`), so the app starts without training. Notes are stored in `data/notes.db` (SQLite, created on first run, gitignored). Set `SMARTNOTES_DB=/path/to/other.db` to use a different file.

### The interface
A Notion-style sidebar lists **All notes**, **Needs a look** and every label with its count. The page has a capture block (it shows the predicted label and confidence as you type; **Ctrl/⌘ + Enter** saves), a search bar (**Ctrl/⌘ K** or **/** jumps to it, **Esc** clears it) and three views of the same notes:
- **Cards** and **List**: label, confidence, how long ago, delete on hover.
- **Map**: an Obsidian-style graph with one hub per label and each note orbiting it. Hover a dot to read the note, click to jump to it. Dashed rings are notes the model isn't sure about.

The theme follows your system (Auto) and can be set to Light or Dark from the sidebar. The URL hash can preset a view or theme, for example `http://127.0.0.1:8000/#map` or `#list&dark`.

### Load the sample notes (`N001`-`N012`)
Either click **Load starter notes** in the UI, or run the one-liner:

```bash
python -m smartnotes.store.starter
```

They keep their original IDs. Loading twice adds nothing, and notes you add yourself start at `N013` and never reuse an ID (even after deleting).

### Development mode (hot reload)
```bash
python -m uvicorn smartnotes.api.app:app --port 8000     # terminal 1
cd frontend && npm run dev                               # terminal 2 -> http://localhost:5173
# if the API is on another port:  VITE_API_TARGET=http://127.0.0.1:8765 npm run dev
```

## Retrain the model (one command)

```bash
python train.py
```

Reads `data/train.csv`, compares the baselines and three candidates by template-grouped 5-fold CV on the training data, fits the winner, prints validation accuracy, macro-F1, per-category scores, the confusion table and the mistakes, prints the uncertainty trade-off, and saves `models/notes_model.joblib`. It then trains the **extended label model** (`train.csv` + `data/extended_notes.csv`, five candidates compared by CV) and saves `models/extended_model.joblib`. Both files are loaded by the app. It takes about 75 seconds. Validation labels are never used for fitting or choosing a model.

Related commands:
```bash
python -m smartnotes.ml.evaluate_labels                  # shown-label accuracy and uncertain rates on the held-out sets
python -m smartnotes.ml.experiments                      # the "try one fix" comparison
python -m smartnotes.ml.explain "<note>" --expected Travel   # why a prediction went wrong: features + nearest training notes
python eval_search.py                                    # search hit rate @3 for all approaches
```

## The `/classify` API

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Pay my electricity bill tomorrow."}'
```

```json
{"category":"Money","uncertain":false,"confidence":1.0,"probabilities":{"Money":1.0,"Shopping":0.0,"Study":0.0,"Travel":0.0},"label":"Money","label_confidence":0.9886}
```

- `category` is always one of `Study`, `Money`, `Travel`, `Shopping` (the four-label model's best guess, even when unsure). `uncertain` is a JSON boolean. `confidence` and `probabilities` are extras.
- `label` and `label_confidence` are extras for the app: the label also can be `Health`, `Family`, `Home`, `Work` or `Other`. For "I have to go to gym today at 6:00 p.m." the contract still returns one of the four in `category`, with `uncertain: true` because it fits none of them, while `label` is `Health`.
- Invalid input (empty or whitespace text, missing field, non-string, `null`, a list, malformed JSON, more than 1,000 characters) returns **422** with `{"message": "...", "errors": [...]}`, and the server keeps working.
- **Windows PowerShell** strips the inner quotes from `curl -d '{"text":...}'`. Use:
  `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/classify -ContentType application/json -Body '{"text":"Pay my electricity bill tomorrow."}'`

Self-test supplied with the brief, and a replay of the validation set through the live endpoint:

```bash
python examples/test_api.py --url http://127.0.0.1:8000/classify
python examples/replay_validation.py --url http://127.0.0.1:8000/classify   # reproduces the RESULTS.md validation numbers
```

Other routes (interactive docs at `/docs`): `GET/POST /api/notes` (filters `?category=` and `?uncertain=`), `GET/DELETE /api/notes/{id}`, `POST /api/notes/load-starter`, `GET /api/search?q=`, `GET /api/health`.

## Tests

```bash
python -m pytest
```

93 tests, about 7 seconds, no network or credentials. They run against a temporary database, never `data/notes.db`. They cover `/classify` (valid note, every invalid-input case, 1,000-character and `₹` text, determinism, and that the `uncertain` flag follows the model's threshold), saving and reloading notes across a restart, ID policy, idempotent starter loading, search, and the evaluation sets.

## How it works

```
smartnotes/
  ml/        preprocess, baselines, model pipeline, grouped CV, uncertainty threshold, explain, experiments, train
  text.py    shared function-word stop list
  services/  NoteClassifier: loads the saved model once, returns category / uncertain / confidence
  store/     SQLite note store, ID policy, starter loader
  search/    keyword, TF-IDF and hybrid retrievers, evaluation
  api/       FastAPI app factory, routes (classify, notes, search, health), validation and error shape
frontend/    React + Vite UI (composer with live preview, filters, search)
data/        supplied data + search_queries.json, search_no_match_queries.json
tests/       pytest suite
```

- **Two models.** The four-label model decides the contract's `category`. The extended model (also Health, Family, Home, Work, trained on 252 extra notes I wrote) decides the `label` shown in the app and gives a second opinion on `uncertain`. If `models/extended_model.joblib` is missing the app still runs on the four-label model alone.
- **Uncertain notes.** The four-label model's highest probability must reach a threshold (0.8971, stored inside the model file); the extended model can overrule it when it confidently agrees (not uncertain) or confidently names a label outside the four (uncertain). The UI shows the amber badge only for notes the model cannot place or doubts, not for confidently recognised Health/Family/Home/Work notes.
- **Search.** `score = tf-idf cosine + 0.5 * topic agreement`. The classifier also labels the query, so "money transfer" finds the savings note with no shared words. The topic boost turns off when the classifier is not confident about the query, so unfamiliar queries return nothing rather than unrelated notes.
- **Why not an LLM?** The brief requires a trained classifier driving predictions, and the whole project runs offline on a CPU.

## Limitations (details in RESULTS.md)
- The supplied data is synthetic and template-based, so 0.92 validation accuracy means "good on notes that look like the training templates". New phrasing (for example "railway station") can be confidently wrong.
- Naive Bayes probabilities are overconfident. The uncertain flag catches notes unlike the training data, not every wrong answer; the threshold was partly tuned on validation, as the brief permits.
- Labels beyond the four rest on 252 notes I wrote and only cover Health, Family, Home and Work. On a 27-note final set 22 got the right label; unfamiliar words become `Other` or a wrong guess, and when both models agree on a wrong core label (for example a poem labelled Study) the note is not flagged. Adding a label means adding notes for it and running `python train.py`.
- Topic search relies on the classifier: a query it is unsure about gets no topic boost (for example `where am I staying?` finds nothing), and one it is wrongly sure about can return unrelated notes (`water the plants`).
- The 12 starter notes hold 3 per category, which makes hit rate @3 generous; RESULTS.md also reports hit rate @1 and MRR.
- The web UI was rendered and screenshot-checked, but not driven by automated browser tests; its endpoints are covered by the API tests.
- Single user, single process; there is no authentication.

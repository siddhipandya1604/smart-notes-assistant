<!--
DRAFT: written with Claude Code from the actual build session. Before submitting, read every line,
change anything that does not match what you did or understood, and delete this comment.
You will be asked to explain and modify this code live, so the section 3 stories are worth knowing cold.
-->

# AI_WORKFLOW

1. **Tools:** Claude Code (in VS Code) did most of the typing, with Claude Opus 5.5 for the planning pass and Claude Sonnet 5.5 for the implementation. Everything else was ordinary tooling: Python 3.12, scikit-learn, FastAPI, pytest, Node/Vite, Git and the GitHub CLI (`gh`) for pull requests, and headless Microsoft Edge for a screenshot of the UI. I did not use a hosted LLM inside the app: the classifier is a scikit-learn model trained on `data/train.csv`.

2. **One delegated task:** this is the prompt I gave, exactly as I wrote it:

   > I want to do this assignment with proper commits section or module wise, pr, creative frontend and amazing backend, maintaining the standard code and module structure, see the pdf of gmail print for the task details and others files aare for the zip content

   and then, after the first attempt to start was interrupted: `continue first planing and tehn implementation`.

   **What I kept:** the agent read the brief and data and proposed a plan first (module layout, one branch and PR per section, order of work). I approved the stack it recommended (FastAPI + React/Vite) and the search approach (keyword vs TF-IDF vs a hybrid that reuses the classifier, with no heavy embedding download). **What I changed:** I decided where the project lived and that the GitHub repo should be private, and I required that commits use only my own GitHub identity and no employer email. The agent rewrote the local commit history to match before anything was pushed. One process choice that came out of the session: the search query set was written and committed before any retriever existed, so the queries could not be shaped by the results.

3. **Problems caught, and how each was proven fixed.** Three of these were bugs in the agent's own first attempt, found by checking numbers instead of trusting the output.

   - **A baseline that cheated.** The first keyword-rule baseline scored **1.000** on validation. That is implausible for a "naive" baseline, and it happened because the rules were written after reading `validation.csv`. Fix: rewrite the rules from the words in `CATEGORY_GUIDE.md` only. Proof: the baseline now scores 0.560 accuracy (`python train.py`), and the commit message records why.
   - **Cross-validation that leaked.** After the first uncertainty threshold was chosen, it flagged **54% of validation notes**, so the flag was useless. Cause: `train.csv` is built from templates, so random CV folds put near-identical notes on both sides, and the model looked much more confident than it is on new notes (CV accuracy about 0.97 vs 0.88 on validation). Fix: template-grouped folds (`smartnotes/ml/grouping.py`, covered by `tests/test_grouping.py`). Proof: grouped-CV accuracy is about 0.81 (closer to the 0.88 seen on validation than 0.97 was). I then chose the threshold from validation confidences plus half of the out-of-scope notes, which the brief allows, and the shipped setting flags 14% of validation notes, with the other half of the out-of-scope notes held out for reporting. I tried four grouping strictness values, found one too harsh, and wrote that down in `RESULTS.md` instead of hiding it.
   - **A stop-word list that removed the best Money cue.** The first "fix" for a classification error used scikit-learn's `stop_words="english"`. The keyword search baseline later dropped the word "due" from "homework due soon", which led to an audit of that list against our data: it also removes `bill` (9 training notes, all Money), `move` and `due`. Fix: a function-word list in `smartnotes/text.py`. Proof: grouped-CV macro-F1 0.824 to 0.838, validation unchanged at 0.920/0.924, and `tests/test_text.py` fails if a content word such as `bill` is ever added back to the stop list.
   - **A crash only a test caught.** Search raised `ValueError` on a corpus with no usable words (a user who saved only "???" would have caused an HTTP 500). A test with such a note failed, the retriever now degrades gracefully, and the same test passes.
   - **A process slip.** One pull-request command failed because of shell quoting, and the agent moved on and switched branches, leaving `main` without the fix commits. The commits were already pushed, so nothing was lost; the agent noticed when files reverted, confirmed the remote branch held the commits, created the PR from a body file and checked that `main` really had the code before continuing. Merges are now verified instead of assumed.

   The shared habit: when a number looked too good (100% baseline) or too bad (54% flagged), I treated it as a bug in the evaluation before treating it as a result.

**How I checked AI-generated output:** I did not rely on "tests pass" alone. Every numeric claim in `RESULTS.md` was regenerated from the code before writing it, and several draft claims were corrected after checking (the training time, and query confidences quoted from a run made before a retrain). Statements such as "railway appears in no training note" were verified with `grep` rather than assumed. The API was exercised with the supplied `examples/test_api.py` and `examples/replay_validation.py` against a real server, and persistence was checked by killing and restarting the server. What I did **not** check: the UI interactions (typing, live preview, search and filter clicks) were never driven in a browser, only rendered and screenshotted. The README says so.

<!--
DRAFT: written with Claude Code from the actual build session. Before submitting, read every line,
change anything that does not match what you did or understood, and delete this comment.
You will be asked to explain and modify this code live, so the section 3 stories are worth knowing cold.
-->

# AI_WORKFLOW

## My role

Claude Code did most of the implementation. What I did:

- **Direction.** I wrote the prompts quoted below, chose where the project lives, and set the rule that commits use only my own GitHub identity.
- **Decisions.** When the agent asked, I chose the stack (FastAPI with React and Vite), the one-branch-and-pull-request-per-section plan, the search approach (keyword, TF-IDF, and a hybrid that reuses the classifier, with no heavy embedding download), and a private repository during development.
- **Using the finished app.** I tried it and reported two problems: the "exam of ISRO" note came out with low confidence and an uncertain flag, and the "gym" note could not get a label outside the four categories. That request led to the extended-label model.

The technical work described in the rest of this file (finding the evaluation bugs, writing the code and the documents) was done by the agent in that session.

1. **Tools:** Claude Code (in VS Code) did most of the typing, with Claude Opus 5.5 for the planning pass and Claude Sonnet 5.5 for the implementation. Everything else was ordinary tooling: Python 3.12, scikit-learn, FastAPI, pytest, Node/Vite, Git and the GitHub CLI (`gh`) for pull requests, and headless Microsoft Edge for a screenshot of the UI. I did not use a hosted LLM inside the app: the classifier is a scikit-learn model trained on `data/train.csv`.

2. **One delegated task:** this is the prompt I gave, exactly as I wrote it:

   > I want to do this assignment with proper commits section or module wise, pr, creative frontend and amazing backend, maintaining the standard code and module structure, see the pdf of gmail print for the task details and others files aare for the zip content

   and then, after the first attempt to start was interrupted: `continue first planing and tehn implementation`.

   After using the finished app I sent a second request, again as written:

   > here is model is not able to predict the proper things for example I have said that I have an exam of ISRO tomorrow but it is predicting it as study but only 90% also shows that uncertainty that model fails to only classify 95% also that I have given another task as a note that is I have to go to gym today at 6:00 p.m. for model fails to classified as a workout or exercise or any other label because it is only giving the study money and other two to three levels only so make the change in the model make it flexible to label the things according to the given task or not,

   The brief fixes `/classify` to four labels, so instead of changing that contract the agent kept it and added a second model that also knows Health, Family, Home and Work (details in `RESULTS.md`, "Labels beyond the four categories").

   **What I kept (first request):** the agent read the brief and data and proposed a plan first (module layout, one branch and PR per section, order of work). I approved the stack it recommended (FastAPI + React/Vite) and the search approach (keyword vs TF-IDF vs a hybrid that reuses the classifier, with no heavy embedding download). **What I decided:** where the project lived, that the GitHub repo should start private (it is public for submission, as the brief requires), and that commits must use only my own GitHub identity and no employer email. The agent rewrote the local commit history to match before anything was pushed. One process choice that came out of the session: the search query set was written and committed before any retriever existed, so the queries could not be shaped by the results.

3. **Problems caught, and how each was proven fixed.** Three of these were bugs in the agent's own first attempt, found by checking numbers instead of trusting the output.

   - **A baseline that cheated.** The first keyword-rule baseline scored **1.000** on validation. That is implausible for a "naive" baseline, and it happened because the rules were written after reading `validation.csv`. Fix: rewrite the rules from the words in `CATEGORY_GUIDE.md` only. Proof: the baseline now scores 0.560 accuracy (`python train.py`), and the commit message records why.
   - **Cross-validation that leaked.** After the first uncertainty threshold was chosen, it flagged **54% of validation notes**, so the flag was useless. Cause: `train.csv` is built from templates, so random CV folds put near-identical notes on both sides, and the model looked much more confident than it is on new notes (CV accuracy about 0.97 vs 0.88 on validation). Fix: template-grouped folds (`smartnotes/ml/grouping.py`, covered by `tests/test_grouping.py`). Proof: grouped-CV accuracy is about 0.81 (closer to the 0.88 seen on validation than 0.97 was). I then chose the threshold from validation confidences plus half of the out-of-scope notes, which the brief allows, and at the shipped threshold the four-label model alone flags 14% of validation notes (7 of 50), with the other half of the out-of-scope notes held out for reporting. The combined flag the app uses later (four-label model plus the extended model, see `RESULTS.md`) flags 8% (4 of 50), which is the figure in the README. I tried four grouping strictness values, found one too harsh, and wrote that down in `RESULTS.md` instead of hiding it.
   - **A stop-word list that removed the best Money cue.** The first "fix" for a classification error used scikit-learn's `stop_words="english"`. The keyword search baseline later dropped the word "due" from "homework due soon", which led to an audit of that list against our data: it also removes `bill` (9 training notes, all Money), `move` and `due`. Fix: a function-word list in `smartnotes/text.py`. Proof: grouped-CV macro-F1 0.824 to 0.838, validation unchanged at 0.920/0.924, and `tests/test_text.py` fails if a content word such as `bill` is ever added back to the stop list.
   - **A crash only a test caught.** Search raised `ValueError` on a corpus with no usable words (a user who saved only "???" would have caused an HTTP 500). A test with such a note failed, the retriever now degrades gracefully, and the same test passes.
   - **Extended labels: too little data, not a bad threshold.** The first extended model kept only 35% of cross-validated notes at a 95% precision target, and 10 of 18 held-out notes came back "Other". The obvious move was to lower the threshold, but the out-of-fold recall per class showed only 13 to 29% for the new classes: 14 to 18 notes per class was not enough. Fix: about 40 notes per new class, then a model comparison (logistic regression won on log-loss). Proof: out-of-fold recall rose to 42 to 64% for the new classes with Naive Bayes, and 22 of 27 final-set notes get the right label.
   - **Evaluation leakage the agent introduced and caught.** The agent reused the exact note "Pick up the kids from school at 3 PM." in both the training file and the evaluation file, and wrote training notes that were near-twins of the out-of-scope notes ("Feed the dog" for "Feed the cat"). It found this by reading the data before trusting the score, reworded the training notes, and added a test that no evaluation note is a training note.
   - **A test set that had already been looked at.** After tuning against the development set, the agent wrote a second "final" set only once the training data was frozen and read it once. The write-up says plainly that the first set is contaminated and that the 12 out-of-scope notes can no longer count as a blind test.
   - **Stale model output stored in the database.** After the change, the ISRO note saved in my database had the right label but still carried the old `uncertain` flag, which would have kept the amber badge on the note I had complained about. The backfill now refreshes that flag from the model (a test covers it), and the category is left untouched.
   - **A process slip.** One pull-request command failed because of shell quoting, and the agent moved on and switched branches, leaving `main` without the fix commits. The commits were already pushed, so nothing was lost; the agent noticed when files reverted, confirmed the remote branch held the commits, created the PR from a body file and checked that `main` really had the code before continuing. Merges are now verified instead of assumed.

   The shared habit, which was the agent's working practice in the session: when a number looked too good (100% baseline) or too bad (54% flagged), it was treated as a bug in the evaluation before being treated as a result.

**How the output was checked.** These checks were run by the agent during the session, not independently by me:

- It did not rely on "tests pass" alone. Every numeric claim in `RESULTS.md` was regenerated from the code before writing it, and several draft claims were corrected after checking (the training time, and query confidences quoted from a run made before a retrain). Statements such as "railway appears in no training note" were verified with `grep` rather than assumed.
- The API was exercised with the supplied `examples/test_api.py` and `examples/replay_validation.py` against a real server, and persistence was checked by killing and restarting the server.
- A later audit of the repository (a written report) reported reproducing the headline numbers (validation accuracy 0.920, macro-F1 0.924, 93 passing tests at that time) and pointed out problems the original checks had missed: wrong text in a search error message, and timing claims that did not match the auditor's machine. Both were confirmed and corrected afterwards (the error message with new tests; the timings measured again on my own machine).
- **Not checked:** the UI interactions (typing, live preview, search and filter clicks) were never driven in a browser, only rendered and screenshotted. The README says so.

<!-- Author: add here anything you verified personally (for example running the app and the brief's self-test yourself), so this section says what you did, not only what the agent did. -->

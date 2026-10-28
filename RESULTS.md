# RESULTS

Every number below is printed by the code in this repository. Re-run `python train.py`, `python -m smartnotes.ml.experiments` and `python eval_search.py`, or replay validation through a live server with `python examples/replay_validation.py`.

## Dataset and training
- Training rows: 156
- Validation rows: 50
- Command that reproduces these numbers: `python train.py` (about 75 s on a laptop CPU; it trains the four-label model described here, saves `models/notes_model.joblib`, then trains the extended label model described in "Labels beyond the four categories" and saves `models/extended_model.joblib`)
- Baselines:
  - **Majority class** (always `Study`).
  - **Keyword rules**, written from the "Includes" column of `CATEGORY_GUIDE.md` only (`smartnotes/ml/baseline.py`). I had already read `validation.csv` when I first wrote them, and that first version scored 100% on it, so it was rewritten to use only words from the guide.
- Trained model: TF-IDF features (word 1-2 grams plus character 2-5 grams, numbers replaced by a `num` token, function words removed) into **Multinomial Naive Bayes** (`alpha=0.1`).
- Library versions and seed: scikit-learn 1.5.2, numpy 2.0.2, Python 3.12, `random_state=42` everywhere.

### How the model was chosen (and two evaluation mistakes I fixed on the way)
Candidates are compared with 5-fold CV **on `train.csv` only**; `validation.csv` is never used to pick a model.

1. **Random folds were too optimistic.** `train.csv` is built from templates ("Look for a sale on X this Sunday"), so a random fold almost always holds a near-twin of each test note in its training half. Random-fold CV accuracy was about 0.97 while validation accuracy was 0.88, and the confidences it produced made my first uncertainty threshold flag **54% of validation notes**. Notes with cosine similarity >= 0.6 now stay in the same fold (`smartnotes/ml/grouping.py`). I tried similarity 0.4, 0.5, 0.6 and 0.7: 0.4 was far too harsh (CV accuracy about 0.5, because whole vocabularies disappear from the training folds), and the model ranking was the same for 0.5 to 0.7. I chose 0.6 because it groups exact-template siblings, not because of a score.
2. **scikit-learn's stop-word list removes category cues.** It contains `bill` (9 training notes, all Money), `move`, `due`, `find`, `put`, `call`, `amount`, `cost`, `interest`. I replaced it with a function-word list (`smartnotes/text.py`) and re-ran the experiment.

## Metrics (validation dataset only)

| Approach | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Majority class | 0.300 | 0.115 |
| Keyword rules | 0.560 | 0.529 |
| Trained model (Naive Bayes + fix) | **0.920** | **0.924** |

What improved: the trained model gains about 36 accuracy points over the guide-only keyword rules. On the model-selection data (grouped CV on train), the candidates scored:

| Candidate | grouped-CV accuracy | grouped-CV macro-F1 | log-loss |
| --- | ---: | ---: | ---: |
| Logistic regression, word+char | 0.756 | 0.721 | 0.826 |
| Logistic regression, word only | 0.660 | 0.606 | 0.973 |
| Naive Bayes, word+char | 0.808 | 0.791 | 0.832 |
| **Naive Bayes + fix (selected)** | **0.853** | **0.838** | **0.668** |

Per category (validation):

```
              precision    recall  f1-score   support
       Study      0.833     1.000     0.909        15
       Money      1.000     1.000     1.000        11
      Travel      0.909     0.769     0.833        13
    Shopping      1.000     0.909     0.952        11

true \ pred      Study     Money    Travel  Shopping
Study               15         0         0         0
Money                0        11         0         0
Travel               3         0        10         0
Shopping             0         0         1        10
```

## Error analysis (at least 3 distinct mistakes)

The final model makes 4 validation mistakes, which are only **2 distinct** ones. The railway-station template accounts for three of them. For the third item I use a mistake the model made before the fix, which is now only flagged uncertain (see the table).

| Note | Expected | Predicted | Possible reason |
| --- | --- | --- | --- |
| I must reach the railway station by 10 AM. (also "...by 2 PM.", "...by 8 AM.") | Travel | Study (0.91, **not** flagged) | "station" is a strong Travel cue, but character n-grams from "must" and "reach" resemble Study vocabulary ("st", "ach", "rea" ~ "statistics", "machine learning", "read"), and "railway" and "reach" appear in no training note (checked with grep) |
| Put snacks and cleaning items in the cart. | Shopping | Travel (0.70, flagged uncertain) | Almost none of its words occur in training ("cart", "snacks" and "cleaning" appear in no training note), so the weak fragments left ("in ", "ca" ~ "cab") point at Travel |
| Check the credit card balance on Sunday. (also "...on the 25th.") | Money | **Fixed**: Money (0.795, flagged uncertain). Before the fix: Travel | Training has "Arrange a cab to the airport on Sunday" and "Check the train departure time for Pune", so "Check the ... on Sunday" matched Travel patterns while "credit" appears in no training note |

**Evidence for the railway-station mistake** (`python -m smartnotes.ml.explain "I must reach the railway station by 10 AM." --expected Travel`):

```
probabilities: Study 0.911, Travel 0.077, Money 0.012, Shopping 0.000
features pushing toward Study over Travel:   'st' +0.322, 'st ' +0.262, 'ch ' +0.235, 'ach' +0.224, 'rea' +0.222
features pushing toward Travel over Study:   'station' +1.606, 'i ' +0.315, 'mu' +0.302, 'ai' +0.301, 'rai' +0.283
nearest training notes (cosine):
  0.31 [Travel] Find directions from the station to our hotel.
  0.18 [Study]  My biology exam starts at 10 AM on the 25th.
  0.16 [Study]  Complete the statistics homework by the 25th.
```

The single Travel word "station" is outweighed by many small Study-leaning character fragments, and the nearest training notes show why: only one Travel note shares a content word, while the Study notes share "10 AM" and "by the 25th". That is a lack of training coverage for "railway/reach", not noise I can fix with a threshold, and the model is confident about it.

## Search evaluation

- Query file and command: `data/search_queries.json`, `python eval_search.py`
- Number of queries / number sharing no words with their target note: **14 / 6** (ignoring common words; the count is computed by the script). A second file, `data/search_no_match_queries.json`, has 4 queries that should find nothing. The query set was committed before any retriever existed.

| Search approach | Hit rate @3 | Hit rate @3, 6 no-shared-word queries | Hit rate @1 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Keyword overlap (baseline) | 0.57 | 0.00 | 0.57 | 0.57 |
| TF-IDF cosine (word + char n-grams) | 0.57 | 0.00 | 0.50 | 0.54 |
| **Hybrid: TF-IDF + classifier topic (used in the app)** | **0.86** | **0.67** | **0.71** | **0.79** |
| Category only (ablation) | 0.86 | 0.67 | 0.43 | 0.61 |

- **Approach used in the app, and why:** the hybrid. The same trained model that labels notes also labels the query, and notes in the query's category get a boost (`score = tf-idf cosine + 0.5 * topic agreement`). That is how "money transfer" finds N006 ("Move ₹2,000 into the savings account...") with no shared words. The boost is switched off when the classifier is not confident about the query (the same threshold as Part D), so an unfamiliar query returns nothing instead of unrelated notes.
- **Read hit@3 with care.** The 12 starter notes hold exactly 3 notes per category, so guessing the right category already guarantees a top-3 hit. The category-only ablation matches the hybrid on hit@3 for that reason. The evidence that the words add something is hit@1 and MRR (0.71 / 0.79 vs 0.43 / 0.61).
- **Queries that still fail:**
  - `where am I staying?` (wanted N007) and `refund broken item` (wanted N008): the classifier is not confident about either query (Shopping 0.66 and Money 0.38), so the topic boost is off and, with no shared words, nothing is returned.
  - `water the plants` should find nothing but returns three Money notes: training contains "water bill", so the classifier is 0.93 sure it is Money. That is 1 of the 4 no-match queries, so the hybrid's no-match score is 0.75. Keyword and TF-IDF score 1.00 simply because none of those 4 queries shares a word with a saved note.
- The topic weight (0.5) and cutoff (0.2) were fixed before running. Hit@3 is 0.86 for weights 0.25, 0.5 and 1.0 (hit@1 reaches 0.79 at 1.0); I did not pick the best of those from 14 queries.

## Notes that fit no category

> **What changed after the first version of this document.** The table below measures the *four-label model alone* (threshold 0.8971). The app now also asks a second, extended model (see "Labels beyond the four categories") and the shipped `uncertain` flag combines both. Combined numbers are in the second table of this section.

- **How the four-label model decides `uncertain`:** the note is uncertain when its highest class probability is below a threshold (`smartnotes/ml/uncertainty.py`). `/classify` always still returns its best-guess category.
- **How the threshold was chosen:** the 12 out-of-scope notes are split once with a seeded shuffle into 6 for choosing and 6 held out. Among thresholds that flag at most a chosen share of **validation** notes (the brief allows validation for this), I take the one that flags the most of the 6 tuning out-of-scope notes. The shipped model uses the 20% cap.
- I did **not** train on `out_of_scope_examples.csv`. The classifier only ever sees the four categories, so the held-out six are a clean test of the flag.

| Setting (cap on validation flagged) | Threshold | Out-of-scope flagged, tuning 6 | Out-of-scope flagged, **held-out 6** | Out-of-scope flagged, all 12 | Validation notes flagged | Train grouped-OOF flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 0.6988 | 50% | 67% | 58% | 2% | 14% |
| 10% | 0.8379 | 67% | 83% | 75% | 8% | 24% |
| **Chosen: 20%** | **0.8971** | 83% | **100%** | **92%** | **14%** | 31% |
| 30% | 0.9792 | 100% | 100% | 100% | 28% | 45% |

- **Why this trade-off:** a spurious "Uncertain" badge is cheap, since the note stays visible with its best-guess category, while a confident wrong label on a note that fits no category is the failure the brief is about. 20% is the highest cap that still flags a minority of validation notes (7 of 50). Past it, flagging 28% of normal notes starts to make the badge less useful.
- **What is and is not independent evidence:** the "validation" and "tuning 6" columns were used to pick the threshold, so they are optimistic by construction. The honest checks are the **held-out 6** (100% flagged, but 6 notes means one note is 17 points) and the **grouped train out-of-fold** column (31%), which is a pessimistic bound because those folds hold back whole templates. Use ranges, not point values: roughly 80-100% of out-of-scope notes flagged, roughly 14-31% of in-scope notes flagged.
- With the four-label model alone, 1 of the 4 validation mistakes (the "cart" note) is flagged uncertain and the 3 railway-station notes are not. With the combined flag (below), 2 of the 4 are flagged: the cart note and one of the three railway notes.

### What the app ships: the combined `uncertain` flag

The extended model (more labels, more training notes) gives a second opinion. `uncertain` is **false** when it confidently names the same core label as the four-label model, **true** when it confidently names something outside the four (Health, Family, Home, Work), **true** when it confidently names a different core label, and otherwise falls back to the four-label rule. Measured on the same notes with the old rule and the new rule:

| Notes | Flagged uncertain, four-label model only | Flagged uncertain, combined (shipped) |
| --- | ---: | ---: |
| `validation.csv`, 50 in-scope notes | 7 (14%) | **4 (8%)** |
| The 12 out-of-scope notes | 11 (92%) | **12 (100%)** (but see below) |
| Final set: 12 notes outside the four categories (Health/Family/Home/Work) | 7 | **11** |
| Final set: 12 ordinary core-category notes | 2 | 2 |
| Final set: 3 notes that fit nothing ("Other") | 2 | 2 |

- **The 12 out-of-scope notes are no longer an independent test.** I wrote extra training notes after reading them, so words like "feed", "trash", "vitamins" and "neighbour" appear in training. The 12/12 is a development check. The clean evidence is the final-set rows: 11 of 12 notes outside the four categories are flagged, and 2 of the 3 "Other" notes. The final set has only 27 notes, so these are small counts.
- **A known false negative:** "Write a poem about the monsoon." is predicted Study at 0.92 by both models, so it is *not* flagged. When the two models agree on a wrong core label, the combined rule clears the flag.

## Labels beyond the four categories

**The problem, as reported from using the app.** "I have an exam of ISRO tomorrow" was labelled Study but marked uncertain (0.83, below the 0.8971 threshold), and "I have to go to gym today at 6:00 p.m." came out as **Money** (0.47), because the model only has four labels and none fits. The causes: the training data is 156 templated notes (only 4 contain the word "exam", none "ISRO" or "gym"), and a four-label model has nothing to say about a workout.

**The constraint.** The brief fixes `POST /classify` to exactly Study / Money / Travel / Shopping plus an `uncertain` flag, and reviewers re-run validation through it. So that contract is unchanged: `category` is still the four-label model's answer (validation accuracy 0.920, macro-F1 0.924, re-checked through the live endpoint). The additions sit beside it:

- A second **extended model** that also knows **Health, Family, Home and Work**. It gives `label` and `label_confidence` (extra fields in `/classify`, shown in the app) and a second opinion on `uncertain`, as described above. A label is shown only above a threshold; otherwise it is `Other`.
- It is trained on `train.csv` plus `data/extended_notes.csv`: 252 notes I wrote (Health 43, Family 39, Home 40, Work 38, and 24 / 24 / 22 / 22 more phrasings of Study / Money / Travel / Shopping). This is my own synthetic data, so it is only as good as my coverage of how people write notes.

**Model choice** (grouped 5-fold CV on the 408 training notes, chosen by log-loss because the label threshold reads the probabilities. The selection uses only training-data CV, but note that I had already seen development-set results from an earlier Naive Bayes version, so the development set is not a blind test; the final set below is):

| Candidate | grouped-CV accuracy | macro-F1 | log-loss |
| --- | ---: | ---: | ---: |
| Naive Bayes, word+char | 0.721 | 0.681 | 0.986 |
| Naive Bayes, word only | 0.711 | 0.660 | 1.051 |
| Logistic regression C=10, word+char | 0.750 | 0.715 | 0.983 |
| **Logistic regression C=100, word+char (selected)** | **0.750** | **0.714** | **0.832** |
| Logistic regression C=100, word only | 0.711 | 0.673 | 0.995 |

The label threshold is 0.5749: the lowest cut-off at which grouped out-of-fold predictions are at least 90% correct. I first aimed for 95%, saw that it kept only 35 to 46% of cross-validated notes as labelled, and lowered the target to 90%; that was a judgement call made from cross-validation numbers, not from the final set. At that cut-off 65% of cross-validated notes keep a label. Grouped folds are pessimistic here, so shown labels should be right more often than that on ordinary notes.

**What happened on the way (kept because it changed the design).** My first extended model used only about 14 to 18 hand-written notes per new class. Out-of-fold recall on the new classes was only 13 to 29%, and no threshold can fix that, because the model had not seen enough phrasings. Writing about 40 per class raised it (Naive Bayes 42 to 64%), and switching to logistic regression raised overall accuracy further. I also found training notes that were near-twins of the out-of-scope evaluation notes ("Feed the dog" / "Feed the cat", "take vitamins") and reworded them; a test now checks that no evaluation note is a training note.

**Held-out results** (`python -m smartnotes.ml.evaluate_labels`; labels are mine; all of these use the app's combined classifier):

| Set | Notes | Right label shown | Labelled Other |
| --- | ---: | ---: | ---: |
| `validation.csv` (four core labels) | 50 | 44 (88%) | 3 |
| Development set, hand-written new notes | 18 | 16 | 1 |
| Development set, the 12 out-of-scope notes (contaminated, see above) | 12 | 8 | 2 |
| **Final set, written after the training data was frozen and read once** | **27** | **22 (81%)** | 4 |

The five final-set misses, unedited:

| Note | Expected | Shown | Why it matters |
| --- | --- | --- | --- |
| Reach the bus stand by 9 AM for the Surat bus. | Travel | Other (0.31) | "bus stand" and "Surat" are new; the four-label model still says Travel but is unsure |
| Find a cheaper blender on the app. | Shopping | Other (0.38) | "blender" and "app" are unseen |
| Go to the school parents meeting. | Family | Work (0.58) | "meeting" pulls towards Work |
| Fill in the expense report for the business trip. | Work | Travel (0.92) | confidently wrong; both models say Travel, so it is not flagged |
| Write a poem about the monsoon. | Other | Study (0.92) | confidently wrong; there is no "Other" class to learn from, only low confidence |

**The two reported examples** (both were in the development set, never trained on): "I have an exam of ISRO tomorrow" is now Study at 0.98 with `uncertain: false` (was Study at 0.83, uncertain). "I have to go to gym today at 6:00 p.m." is now Health at 0.98 (was Money at 0.47); the contract's `category` is still one of the four (Money) and `uncertain` is true because it fits none of them, while the app shows Health without an amber badge.

**Limits of this evidence.** The final set is 27 notes with labels I wrote, so 81% could easily be 70% or 90% on other notes. The model only knows words it saw: unfamiliar vocabulary becomes "Other" or a wrong guess, and a confident wrong label is possible (the last two rows). Extra labels are only the four I chose (Health, Family, Home, Work); adding a label means adding notes for it and retraining. Embeddings from a pretrained language model would generalise to unseen words much better, but need a large download, which I avoided so the project runs offline.

## What I tried to fix, and whether it worked
- **Targeted mistake:** "I must reach the railway station by 10 AM." was predicted Study with 0.99 confidence. Evidence: the stop word "by" and the time "10 AM" outweighed "station".
- **Change:** replace numbers with a `num` token and drop function words from the word features. Chosen with grouped CV on train only, with the adoption rule (gain of at least 0.01 macro-F1) fixed beforehand (`python -m smartnotes.ml.experiments`):

| Variant | grouped-CV accuracy | macro-F1 | log-loss |
| --- | ---: | ---: | ---: |
| Naive Bayes, word+char (before) | 0.808 | 0.791 | 0.832 |
| + numbers to `num` | 0.801 | 0.785 | 0.817 |
| + drop function words | 0.833 | 0.817 | 0.668 |
| **+ both (adopted)** | **0.853** | **0.838** | **0.668** |

- **Result on validation (read once):** accuracy 0.880 to 0.920, macro-F1 0.883 to 0.924, mistakes 6 to 4. It fixed the two "credit card balance" notes (Money, now flagged uncertain) but **did not fix the railway-station notes**: confidence fell from 0.99 to 0.91, and they are still wrong. Numbers alone did not help (-0.006 macro-F1); the gain came from dropping function words.
- **Why the target stayed wrong:** after the fix, "station" pushes Travel more than twice as hard (+1.61 vs +0.82 before), but character n-grams from "must" and "reach" still push Study. The next experiment would tune the character n-gram range with grouped CV.

## What I would improve next
- Collect real, varied notes: the supplied data is a few dozen templates, so "railway", "cart" and "credit card" are single points of failure.
- Tune the char n-gram range (the cause of the remaining mistake) and calibrate probabilities properly (for example with `CalibratedClassifierCV`) instead of relying on raw Naive Bayes probabilities.
- Add a real "Other" class trained on out-of-scope notes, compare it against the threshold, and evaluate with a larger held-out set.
- Replace the topic boost with sentence embeddings if a heavier install is acceptable, and add a search set with more than 3 notes per category so hit@3 is not so generous.

## Limitations
- This small dataset is synthetic and cannot demonstrate real-world reliability on all kinds of notes.
- **How much I trust the validation score:** not much on its own. Fifty notes built from templates means the three railway-station notes (one mistake) cost 6 accuracy points, and its near-twins with train notes make easy cases easy. The gap between random-fold CV (0.97) and validation (0.88) shows how misleading in-distribution scores can be, and notes with a new template will probably score lower than 0.92. I would read 0.92 as "good on notes that look like the training templates".
- Naive Bayes probabilities are overconfident: "water the plants" gets 0.93 for Money, and the railway-station mistake is confident. The `uncertain` flag catches notes that look unlike training data, not every wrong answer.
- Topic search depends on the classifier. A query it is unsure about, or confidently wrong about, gets no topic boost or the wrong one.
- The uncertainty threshold was partly tuned on validation (as the brief permits), so its validation flag rate is not an independent estimate.
- The extended labels (Health, Family, Home, Work) come from 252 notes I wrote. Their 81% on the 27-note final set is a rough guide, not a guarantee, and a confident wrong label is possible when both models agree.

"""Words that carry no category information, shared by the classifier and search.

Why not scikit-learn's stop_words="english"? That list (from the Glasgow IR group) contains
content words that matter here: "bill" (9 of 156 training notes, all Money), "move", "due",
"find", "put", "call", "amount", "cost", "interest". Dropping them removes some of the best
Money cues. This list holds function words only.
"""

STOPWORDS = frozenset("""
a an the and or but if so than then as of to in on at by for with from into onto about
after before over under up out off again here there this that these those
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
is am are was were be been being do does did have has had
will would can could should shall may might must not no
what when where which who whom whose why how
all any some each every both few more most other such only own same too very just also still next
pm
""".split())
# "am"/"pm" are listed as stop words on purpose: in "by 10 AM" they are time-of-day markers,
# which showed up as spurious Study evidence in the error analysis. "am" is already above.

"""Frozen English prompts. Ground-truth labels are never accepted here."""

CAPTION = """Describe this short video segment in English using only visible evidence.
The images are in chronological order and have local-video timestamps.
Describe who does what, relevant objects, object state changes, and the order of actions.
Be specific about actions rather than listing scenery. Distinguish visible events from uncertainty.
Do not invent speech, identities, intentions, off-screen events, or exact times not supplied.
Use at most 120 words. Return only the factual description."""

KEYS = """Extract three complementary retrieval keys from the supplied video description.
event: observable actions and interactions; object: objects involved and visible state changes;
summary: a concise keyword-style description of the whole segment.
Use only facts in the description. Do not introduce speech or new details.
Return JSON with exactly three nonempty string fields: event, object, summary.
Each key should be a short English phrase or sentence, at most 30 words."""

FILTER = """Extract evidence relevant to the user's video highlight query from the supplied
temporally adjacent captions. Preserve the temporal IDs of the captions supporting each fact.
Summarize observed actions and states; do not infer additional actions or assign saliency scores.
Return JSON: {"evidence": [{"clip_ids": [0, 1], "text": "supported relevant information"}]}.
Use only supplied clip IDs. An empty evidence list is allowed if nothing is relevant.
Combine redundant facts and use at most 120 words in all evidence text combined."""

DECIDE = """You are gathering evidence for query-dependent video highlight detection.
Judge whether the retrieved evidence is sufficient to identify the most relevant and salient
moments for the ORIGINAL QUERY. The query may be a description rather than a question.
If useful aspects remain unresolved, propose a short retrieval query targeting those aspects.
Stay faithful to the original query. Do not ask for unrelated or hypothetical events.
Return JSON with exactly: {"sufficient": true, "search_query": ""} or
{"sufficient": false, "search_query": "a refined semantic retrieval query"}."""

SCORE = """Score query-dependent video highlights, using the original query and supplied evidence.
The chronological captions cover the whole video. Retrieved images provide additional visual
evidence at their stated timestamps; filtered neighbor captions provide relevant context.
For each requested 2-second interval, estimate how clearly and prominently the interval depicts
what the query describes, considering the video as a whole. Mere object presence is weaker
evidence than the requested action. Use the same absolute scoring rubric for every batch:
0 = unrelated; 25 = weak/background relevance; 50 = partly or moderately relevant;
75 = clearly relevant; 100 = an exceptionally clear and salient depiction of the query.
Intermediate values are allowed. Do not force each batch to contain a maximum or minimum.
Unretrieved intervals must still be judged from their captions; absence of a retrieved image
does not imply a zero score. Ground temporal judgments in the supplied timestamps.
Return JSON with only {"scores": [number, ...]}, one score per requested interval IN ORDER.
Return exactly the requested number of finite scores in [0, 100], with no explanation."""

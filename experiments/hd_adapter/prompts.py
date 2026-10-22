"""Exact HD and system prompts from the archived VideoARM adaptation."""

SYSTEM = (
    "You analyze only the supplied video pictures. All timestamps are seconds from the start of this dataset segment. "
    "Use the query in every decision. Unknown or unseen regions remain uncertain. A frame is a sample, not proof "
    "of every action in the surrounding interval. Return one bare JSON object, no markdown. "
    "Describe visible evidence briefly; do not produce a reasoning transcript."
)

HD = (
    "Score EVERY requested clip in the given order for relevance to the query and suitability as a highlight "
    "for that query. Higher means more relevant and salient. Use a common 0 to 4 scale across batches; "
    "do not normalize each batch. These are ranking scores, not calibrated probabilities. Return "
    '{"clip_scores":[{"clip_id":0,"score":2.0},...],"events":"brief visible events"}. '
    "Use exactly the supplied IDs, without duplicates or omissions; events at most 2000 characters."
)

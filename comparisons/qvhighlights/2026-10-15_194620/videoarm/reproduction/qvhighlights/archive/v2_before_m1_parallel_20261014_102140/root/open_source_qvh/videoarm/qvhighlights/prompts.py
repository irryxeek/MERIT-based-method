PROTOCOL = "qvh_video_only_v2_stop_contract"
SYSTEM = (
    "You analyze only the supplied video pictures. All timestamps are seconds from the start of this dataset segment. "
    "Use the query in every decision. Unknown or unseen regions remain uncertain. A frame is a sample, not proof "
    "of every action in the surrounding interval. Return one bare JSON object, no markdown. "
    "Describe visible evidence briefly; do not produce a reasoning transcript."
)
OVERVIEW = (
    "Give a brief query-conditioned overview with time-localized visible events and unresolved regions. "
    'Return {"events":"...","uncertainty":"..."}. Each string at most 2000 characters.'
)
SCENE = (
    "Scene Snapper: describe query-relevant events in this requested interval and unresolved questions. "
    'Return {"events":"...","uncertainty":"..."}. Each string at most 2000 characters.'
)
CLIP = (
    "Clip Analyzer: check whether the query event is present, persistent or repeated; estimate its start and end "
    "from these local samples and state uncertainty. Return "
    '{"events":"...","uncertainty":"...","candidates":[[start,end,confidence],...]}. '
    "At most ten candidates, in the requested interval; each string at most 2000 characters."
)
HD = (
    "Score EVERY requested clip in the given order for relevance to the query and suitability as a highlight "
    "for that query. Higher means more relevant and salient. Use a common 0 to 4 scale across batches; "
    "do not normalize each batch. These are ranking scores, not calibrated probabilities. Return "
    '{"clip_scores":[{"clip_id":0,"score":2.0},...],"events":"brief visible events"}. '
    "Use exactly the supplied IDs, without duplicates or omissions; events at most 2000 characters."
)
CONTROLLER = (
    "Observe the immutable initial evidence, current images, memory and budget. Choose the next useful action "
    "or stop early when enough evidence is available. Both Scene Snapper and Clip Analyzer can examine "
    "any valid time interval. Only two visual tools exist; no audio or subtitle tool. For a visual tool return exactly "
    '{"action":"scene_snapper|clip_analyzer","start":0.0,"end":10.0,"question":"short purpose"}. '
    'For stopping return exactly {"action":"stop","start":0.0,"end":0.0}; no tool question is required. '
    "start/end must be finite, positive-length and within duration for tools. "
    "HD clip_analyzer intervals must intersect at most 24 complete 2-second clips."
)
FINAL = (
    "Retrieve all supported query-relevant moments, including separated repeated events. "
    "Use only available evidence and current images. Return "
    '{"pred_relevant_windows":[[start,end,confidence],...]}, at most ten candidates with finite confidence, '
    "0 <= start < end <= duration, sorted descending by confidence. Boundaries need not lie on a 2-second grid. "
    "An empty array is permitted if there is no supported event; never fabricate a full-video candidate."
)

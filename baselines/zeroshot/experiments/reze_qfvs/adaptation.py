"""Predeclared QFVS concept prompt and label-free temporal selection ablation."""

import numpy as np

ACTION_MODE = "reze_action_v1"
CONCEPT_MODE = "visual_concept_or_v1"
SUPPRESSION_RADIUS_SHOTS = 2  # Native shots are 5 s: suppress centers within 10 s.
CONCEPT_SYSTEM_PROMPT = (
    "You are a visual concept presence verifier. Decide whether at least one of the two "
    "named concepts is visibly present in the provided video frames. The concepts denote "
    "objects, people, or places, not actions to perform. Presence in any frame is sufficient; "
    "both concepts need not appear together. Do not infer unseen objects from context. "
    "Answer Yes if either concept is visible, otherwise No. Output a single word: Yes or No."
)


def concept_query(concepts):
    if len(concepts) != 2:
        raise ValueError("Expected exactly two QFVS concepts")
    aliases = {
        "Cupglass": "cup or glass",
        "Musicalinstrument": "musical instrument",
        "Petsanimal": "pet or animal",
        "Building": "a building (an architectural structure, not the act of constructing or assembling)",
    }
    labels = [aliases.get(concept, concept.lower()) for concept in concepts]
    return f"Concept 1: {labels[0]}. Concept 2: {labels[1]}."


def concept_user_prompt(query, n_frames):
    return f"{query}\nIs at least one of these concepts visible in any of these {n_frames} frames? Answer:"


def make_qfvs_request(tokenizer, query, clip, prompt_mode=ACTION_MODE):
    from reze_backend import make_request

    if prompt_mode == ACTION_MODE:
        return make_request(tokenizer, query, clip)
    if prompt_mode != CONCEPT_MODE:
        raise ValueError("Unknown QFVS prompt mode")
    messages = [
        {"role": "system", "content": CONCEPT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "presampled_clip"},
                {"type": "text", "text": concept_user_prompt(query, len(clip["video"]))},
            ],
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "<|video_pad|>" not in prompt:
        raise ValueError("Video placeholder missing from chat template")
    return {
        "prompt": prompt,
        "multi_modal_data": {"video": [(clip["video"], clip["metadata"])]},
        "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
    }


def temporal_summary(scores, fraction=0.02, radius=SUPPRESSION_RADIUS_SHOTS):
    """Greedy score order, suppress +/- radius shots, retain the exact original budget."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid shot scores")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Invalid suppression radius")
    k = int(len(values) * fraction)
    if not 0 < k <= len(values):
        raise ValueError("Invalid summary budget")
    blocked = np.zeros(len(values), dtype=bool)
    selected = []
    for index in np.argsort(-values, kind="stable"):
        if blocked[index]:
            continue
        selected.append(int(index))
        blocked[max(0, index - radius) : min(len(values), index + radius + 1)] = True
        if len(selected) == k:
            return selected
    raise ValueError("Cannot meet the fixed budget with this suppression radius")

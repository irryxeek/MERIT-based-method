"""The observer probe must not leak the original query through the model prompt."""

from copy import deepcopy

from r2_observation_ablation import observation_prompt


def test_query_and_target_identity_cannot_enter_neutral_observer_prompt():
    trace = {
        "action": {"question": "LEAK_SENTINEL find a teal shirt", "candidate": 3},
        "frames": [{"id": "s181", "pts": [6.039, 6.073]}],
    }
    changed = deepcopy(trace)
    changed["action"] = {"question": "DIFFERENT_SENTINEL find a shower", "candidate": 24}
    prompt = observation_prompt(trace)
    assert prompt == observation_prompt(changed)
    assert "LEAK_SENTINEL" not in prompt
    assert "teal shirt" not in prompt
    assert "s181" in prompt and "6.039" in prompt


def test_source_identity_and_timestamp_remain_observable():
    trace = {"frames": [{"id": "s181", "pts": [6.039, 6.073]}]}
    changed = {"frames": [{"id": "s222", "pts": [7.4, 7.433]}]}
    assert observation_prompt(trace) != observation_prompt(changed)
    assert "s222" in observation_prompt(changed)

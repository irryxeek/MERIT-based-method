"""A fresh REZE setup may copy identities, never saliency targets, into inference."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_reze_inputs import inference_rows  # noqa: E402


def test_prepared_rows_exclude_labels_and_use_relocated_video(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"media fixture")
    rows = [
        {
            "qid": index,
            "vid": "v",
            "query": "An event",
            "duration": 10,
            "saliency_scores": [[4, 4, 4]],
            "relevant_clip_ids": [0],
        }
        for index in range(1550)
    ]
    prepared = inference_rows(rows, tmp_path)
    assert len(prepared) == 1550
    assert set(prepared[0]) == {"qid", "vid", "query", "duration", "video_path"}
    assert prepared[0]["video_path"] == str(tmp_path / "v.mp4")
    rows[-1]["qid"] = 0
    with pytest.raises(ValueError, match="unique"):
        inference_rows(rows, tmp_path)


@pytest.mark.parametrize("pairwise", [False, True])
def test_frozen_model_path_overrides_changed_environment(monkeypatch, pairwise):
    import reze_backend

    if pairwise:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/reze_evidence"))
        import backend as implementation
    else:
        implementation = reze_backend
    observed = {}

    def engine(**kwargs):
        observed.update(kwargs)
        tokenizer = SimpleNamespace(encode=lambda label, **kw: [9454 if label == "Yes" else 2753])
        return SimpleNamespace(get_tokenizer=lambda: tokenizer)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(implementation, "MODEL_PATH", "/unexpected/environment/model")
    implementation.make_engine(model_path="/verified/frozen/model")
    assert observed["model"] == "/verified/frozen/model"

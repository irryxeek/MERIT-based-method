"""Exercise the real vLLM video processor on CPU; no model forward or weight loading."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/reze_evidence"))
import protocol  # noqa: E402, F401
from backend import MODEL_PATH, make_request  # noqa: E402


@pytest.fixture(scope="module")
def native_processor():
    if not Path(MODEL_PATH).exists():
        pytest.skip("Requires the audited local tokenizer and processor configuration")
    from transformers import AutoTokenizer
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    config = ModelConfig(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 0, "video": 2},
        mm_processor_kwargs={
            "do_sample_frames": False,
            "do_resize": False,
            "size": {"shortest_edge": 131072, "longest_edge": 4 * 786432},
        },
        mm_processor_cache_gb=0,
    )
    return tokenizer, MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer)


@pytest.mark.parametrize("arm", ["P_abs", "P_pair"])
@pytest.mark.parametrize("frames", [1, 3, 4])
def test_two_video_boundaries_and_short_tail_padding(native_processor, arm, frames):
    tokenizer, processor = native_processor
    a = {
        "video": np.zeros((frames, 64, 96, 3), dtype=np.uint8),
        "metadata": {
            "fps": 2.0,
            "total_num_frames": 4,
            "frames_indices": list(range(frames)),
            "do_sample_frames": False,
        },
    }
    b = {
        "video": np.full((4, 64, 96, 3), 127, dtype=np.uint8),
        "metadata": {"fps": 2.0, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3], "do_sample_frames": False},
    }
    request = make_request(tokenizer, "A person jumps.", arm, a, b)
    result = processor.apply(request["prompt"], request["multi_modal_data"], request["mm_processor_kwargs"])
    placeholders = result["mm_placeholders"]["video"]
    assert len(placeholders) == 2
    assert placeholders[0].offset < placeholders[1].offset
    assert int(placeholders[0].is_embed.sum()) == ((frames + 1) // 2) * 6
    assert int(placeholders[1].is_embed.sum()) == 12
    assert a["metadata"]["frames_indices"] == list(range(frames))
    broken = request["prompt"].replace("Target A:\n", "Target A:").replace("\nTarget B:\n", "Target B:")
    with pytest.raises(RuntimeError, match="prompt placeholders"):
        processor.apply(broken, request["multi_modal_data"], request["mm_processor_kwargs"])


def test_irregular_timebase_renders_true_local_time(native_processor):
    tokenizer, processor = native_processor
    clip = {
        "video": np.zeros((1, 64, 96, 3), dtype=np.uint8),
        "metadata": {
            "fps": 1000000.0,
            "total_num_frames": 2000000,
            "frames_indices": [117200],
            "do_sample_frames": False,
        },
    }
    request = make_request(tokenizer, "A person jumps.", "P_pair", clip, clip)
    result = processor.apply(request["prompt"], request["multi_modal_data"], request["mm_processor_kwargs"])
    assert len(result["mm_placeholders"]["video"]) == 2
    assert tokenizer.decode(result["prompt_token_ids"]).count("<0.1 seconds>") == 2
    assert clip["metadata"]["frames_indices"] == [117200]

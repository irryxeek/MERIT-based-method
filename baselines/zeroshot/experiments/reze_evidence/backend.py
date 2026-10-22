"""Frozen stage-A prompts and native two-video next-token scoring."""

import math
import time
from copy import deepcopy

from reze_backend import MODEL_PATH
from reze_core import extract_pyes

SYSTEM = (
    "Evaluate query-dependent video highlights using only visible evidence.\n"
    "A and B are separate target intervals. Judge how clearly and completely\n"
    "the requested content is shown. Consider emotion, change, or consequences\n"
    "only when relevant to the query and supported by the supplied evidence.\n"
    "Do not prefer general excitement unrelated to the query or invent unseen events.\n"
    "Output exactly one word: Yes or No."
)
PROMPTS = {
    "P_abs": (
        "Query: {query}\n"
        "Evaluate target A. Does A clearly and directly show the requested content,\n"
        "with sufficient visible evidence to be a good highlight for this query?\n"
        "B is comparison context; A does not need to outperform B. Answer:"
    ),
    "P_pair": (
        "Query: {query}\n"
        "Is A a better highlight than B for this query, based on clearer or more\n"
        "complete visible evidence of the requested content?\n"
        "Answer Yes only when A has a supported advantage. For a tie or insufficient\n"
        "evidence of an advantage, answer No. Answer:"
    ),
}


def make_engine(model_path=None):
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model_path if model_path is not None else MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.87,
        kv_cache_memory_bytes=1024**3,
        skip_mm_profiling=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": {"count": 2, "num_frames": 4, "width": 1024, "height": 768}},
        mm_processor_kwargs={
            "do_sample_frames": False,
            "do_resize": False,
            "size": {"shortest_edge": 131072, "longest_edge": 4 * 786432},
        },
        mm_processor_cache_gb=0,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, top_p=1.0, top_k=-1)
    tokenizer = engine.get_tokenizer()
    ids = {word: tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")}
    if ids != {"Yes": [9454], "No": [2753]}:
        raise ValueError(f"Unexpected answer token IDs: {ids}")
    return engine, sampling, tokenizer, {k: v[0] for k, v in ids.items()}


def make_request(tokenizer, query, arm, a, b):
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                # Keep a token boundary before Qwen's generated timestamp.
                # Otherwise ':<' merges, breaking vLLM's placeholder matcher.
                {"type": "text", "text": "Target A:\n"},
                {"type": "video", "video": "target_a"},
                {"type": "text", "text": "\nTarget B:\n"},
                {"type": "video", "video": "target_b"},
                {"type": "text", "text": PROMPTS[arm].format(query=query)},
            ],
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if prompt.count("<|video_pad|>") != 2:
        raise ValueError("Expected exactly two native video placeholders")
    return {
        "prompt": prompt,
        # HF temporal padding mutates frames_indices for odd frame counts.
        # Each comparison must own its metadata; shared evidence stays immutable.
        "multi_modal_data": {"video": [(x["video"], deepcopy(x["metadata"])) for x in (a, b)]},
        "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
    }


def score_requests(engine, sampling, requests, ids):
    """Preserve malformed outputs as semantic failures without regeneration."""
    started = time.time()
    outputs = engine.generate(requests, sampling, use_tqdm=False)
    if len(outputs) != len(requests):
        raise RuntimeError("Engine returned an incomplete batch; runtime recovery required")
    records = []
    for output in outputs:
        answer = output.outputs[0]
        top = answer.logprobs[0] if answer.logprobs else {}
        raw = [
            {
                "id": int(key),
                "text": value.decoded_token,
                "logprob": float(value.logprob) if math.isfinite(value.logprob) else None,
                "rank": value.rank,
            }
            for key, value in top.items()
        ]
        record = {
            "generated_token_id": int(answer.token_ids[0]) if answer.token_ids else None,
            "generated_text": answer.text,
            "finish_reason": answer.finish_reason,
            "prompt_tokens": len(output.prompt_token_ids),
            "output_tokens": len(answer.token_ids),
            "top20": raw,
        }
        try:
            if len(answer.token_ids) != 1 or len(answer.logprobs or []) != 1:
                raise ValueError("Expected one next-token logprob position")
            if answer.token_ids[0] not in ids.values():
                raise ValueError("Generated answer is not Yes or No")
            if any(r["logprob"] is None for r in raw):
                raise ValueError("Nonfinite next-token logprob")
            logs = {r["id"]: r["logprob"] for r in raw}
            if not set(ids.values()) <= set(logs):
                raise ValueError("Required Yes/No logprob missing")
            record["pyes"], record["score_mode"] = extract_pyes(logs, ids["Yes"], ids["No"])
        except ValueError as error:
            record["error"] = str(error)
        records.append(record)
    return records, time.time() - started

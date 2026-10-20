"""Native Qwen3-VL video requests and unmodified-vocabulary top-20 scoring."""

import time

from reze_core import SYSTEM_PROMPT, extract_pyes, user_prompt

MODEL_PATH = "source_models/Qwen3-VL-8B-Instruct"


def make_engine():
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.87,
        kv_cache_memory_bytes=1024**3,
        skip_mm_profiling=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": {"count": 1, "num_frames": 6, "width": 1024, "height": 768}},
        # Bound vLLM's encoder-cache estimate to our six-frame pixel envelope.
        # do_resize=False means this does not alter the preprocessed pixels.
        mm_processor_kwargs={
            "do_sample_frames": False,
            "do_resize": False,
            "size": {"shortest_edge": 131072, "longest_edge": 6 * 786432},
        },
        mm_processor_cache_gb=0,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, top_p=1.0, top_k=-1)
    tokenizer = engine.get_tokenizer()
    ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ["Yes", "No"]}
    if any(len(v) != 1 for v in ids.values()):
        raise ValueError(f"Answer label is not one token: {ids}")
    return engine, sampling, tokenizer, {k: v[0] for k, v in ids.items()}


def make_request(tokenizer, query, clip):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "presampled_clip"},
                {"type": "text", "text": user_prompt(query, len(clip["video"]))},
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


def score_requests(engine, sampling, requests, ids):
    started = time.time()
    outputs = engine.generate(requests, sampling, use_tqdm=False)
    if len(outputs) != len(requests):
        raise ValueError("Missing request outputs")
    records = []
    for output in outputs:
        result = output.outputs[0]
        if len(result.token_ids) != 1 or result.logprobs is None or len(result.logprobs) != 1:
            raise ValueError("No valid next-token log probabilities")
        logs = result.logprobs[0]
        values = {int(key): float(value.logprob) for key, value in logs.items()}
        score, mode = extract_pyes(values, ids["Yes"], ids["No"])
        records.append(
            {
                "pyes": score,
                "score_mode": mode,
                "generated_token_id": int(result.token_ids[0]),
                "generated_text": result.text,
                "finish_reason": result.finish_reason,
                "prompt_tokens": len(output.prompt_token_ids),
                "top20": [
                    {"id": int(key), "text": value.decoded_token, "logprob": float(value.logprob), "rank": value.rank}
                    for key, value in logs.items()
                ],
            }
        )
    return records, time.time() - started

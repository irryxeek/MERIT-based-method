"""Audited native-video structured Qwen inference; no semantic regeneration."""

import json
import os
import re
import time
from copy import deepcopy
from functools import lru_cache

from budget import BudgetStop
from io_utils import digest, read_json, utc_now, write_json
from prompts import SYSTEM

MODEL = os.environ.get("REZE_MODEL_PATH", "models/Qwen3-VL-8B-Instruct")
ENGINE_CONFIG = {
    "model": MODEL,
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": 6144,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 1024,
    "enable_chunked_prefill": True,
    "gpu_memory_utilization": 0.90,
    # Explicit KV bytes override gpu_memory_utilization in vLLM 0.12.
    # 1 GiB holds 7,280 tokens for this model; frozen inputs fit 6,144 context.
    "kv_cache_memory_bytes": 1024**3,
    "skip_mm_profiling": True,
    "enforce_eager": True,
    "enable_prefix_caching": False,
    "generation_config": "vllm",
    "seed": 0,
    "limit_mm_per_prompt": {"image": 0, "video": {"count": 2, "num_frames": 24, "width": 1024, "height": 768}},
    "mm_processor_kwargs": {
        "do_sample_frames": False,
        "do_resize": False,
        "size": {"shortest_edge": 131072, "longest_edge": 4 * 786432},
    },
    "mm_processor_cache_gb": 0,
    "disable_log_stats": True,
}


class SemanticFailure(ValueError):
    pass


def citation_schema(model, references):
    schema = model.model_json_schema()
    for key, choices in (references or {}).items():
        if "." in key:
            definition, field = key.split(".")
            target = schema["$defs"][definition]["properties"][field]
        else:
            target = schema["properties"][key]
        if not choices:
            if model.__name__ == "MemoryDraft":
                schema["properties"]["items"]["maxItems"] = 0
                continue
            # An empty reference set permits only an empty containing array.
            target["maxItems"] = 0
            target["minItems"] = 0
        else:
            target["items"] = {"type": "string", "enum": sorted(set(choices))}
    if model.__name__ == "Comparison":
        # Encode the existing validator contract before generation. Unknown may
        # still cite observed frames; unsupported decisions remain insufficient.
        branches = []
        for preference in ("a_better", "b_better", "tie"):
            branch = deepcopy(schema)
            props = branch["properties"]
            props["preference"] = {"type": "string", "const": preference}
            for side in ("a", "b"):
                props[f"{side}_evidence"]["minItems"] = 1
            props["reason"]["minLength"] = 1
            if preference in {"a_better", "b_better"}:
                props[preference[0] + "_eligibility"]["enum"] = ["satisfied", "unknown"]
            branches.append(branch)
        for cited_a in (False, True):
            for cited_b in (False, True):
                branch = deepcopy(schema)
                props = branch["properties"]
                props["preference"] = {"type": "string", "const": "insufficient"}
                props["missing_question"] = {"type": "string", "minLength": 1}
                for side, cited in (("a", cited_a), ("b", cited_b)):
                    if cited:
                        props[f"{side}_evidence"]["minItems"] = 1
                    else:
                        props[f"{side}_evidence"]["maxItems"] = 0
                        props[f"{side}_eligibility"] = {"type": "string", "const": "unknown"}
                branches.append(branch)
        schema = {"anyOf": branches}
    return schema


def compact_grammar(schema):
    import xgrammar

    # vLLM 0.12's JSON backend reads whitespace settings globally, ignoring the
    # deprecated per-request flag. Explicit EBNF changes only the repaired calls.
    grammar = str(xgrammar.Grammar.from_json_schema(schema, any_whitespace=False, separators=(",", ":")))
    return harden_literal_choices(grammar)


def harden_literal_choices(grammar):
    """Explicit prefix tries avoid the installed compiler's literal-union overacceptance.

    Reproduced with c42_f1, obs0_c42_f0, obs0_c42_f1, obs0_c42_f2:
    its flat union incorrectly accepts c42_f2. Do not rename or repair model
    citations afterwards; constrain the same finite language before sampling.
    """
    terminal = r'"(?:\\.|[^"\\])*"'
    lines = []

    def emit(node, name):
        choices = []
        for index, (char, child) in enumerate(node.items()):
            if char is None:
                choices.append('""')
            else:
                next_name = name + "_trie_" + str(index)
                choices.append(json.dumps(char, ensure_ascii=False) + " " + next_name)
                emit(child, next_name)
        lines.append(name + " ::= (" + " | ".join(choices) + ")")

    for line in grammar.splitlines():
        if " ::= " not in line:
            lines.append(line)
            continue
        name, rhs = line.split(" ::= ", 1)
        values = re.findall(terminal, rhs)
        skeleton = re.sub(terminal, "L", rhs)
        # A sequence such as ("[" "]") is not a union of two choices.
        if len(values) < 2 or not re.fullmatch(r"[()\s]*L[()\s]*(?:\|[()\s]*L[()\s]*)+", skeleton):
            lines.append(line)
            continue
        try:
            strings = [json.loads(value) for value in values]
        except ValueError:
            lines.append(line)
            continue
        trie = {}
        for value in strings:
            node = trie
            for char in value:
                node = node.setdefault(char, {})
            node[None] = None
        emit(trie, name)
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=8)
def nvml_handle(uuid):
    import pynvml

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByUUID(uuid)


def gpu_memory_probe():
    """Read driver counters outside the model process; diagnostics never abort it."""
    import os

    try:
        import pynvml

        value = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle(os.environ["REZE_EXPECTED_GPU_UUID"]))
        return {
            "used_bytes": value.used,
            "free_bytes": value.free,
            "total_bytes": value.total,
            "scope": "device_snapshot_not_inference_peak",
        }
    except Exception as error:
        return {"probe_error": f"{type(error).__name__}: {error}"}


def reuse_response(index, key, specification, validate):
    from io_utils import sha256

    entry = index.get(key)
    if entry is None:
        return None
    if sha256(entry["path"]) != entry["file_sha256"]:
        raise ValueError("Reused response file changed")
    record = read_json(entry["path"])
    if (
        record["status"] != "complete"
        or digest(record) != entry["record_sha256"]
        or record["specification"] != specification
        or record["request_sha256"] != digest(specification)
    ):
        raise ValueError("Reused response is not the exact unchanged valid request")
    return validate(record["parsed"]), record


class Backend:
    def __init__(self, stage_dir, ledger, reuse_index=None, status_path=None):
        from vllm import LLM

        self.ledger, self.directory = ledger, stage_dir
        self.status_path = status_path or stage_dir / "status.json"
        self.reuse_index = reuse_index or {}
        self.engine = LLM(**ENGINE_CONFIG)
        self.tokenizer = self.engine.get_tokenizer()

    def request(self, text, videos):
        content = []
        for i, _ in enumerate(videos):
            content.extend(
                [{"type": "text", "text": f"Target {'AB'[i]}:\n"}, {"type": "video", "video": f"target_{i}"}]
            )
        content.append({"type": "text", "text": text})
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        request = {"prompt": prompt}
        if videos:
            request.update(
                multi_modal_data={"video": [(v["video"], deepcopy(v["metadata"])) for v in videos]},
                mm_processor_kwargs={"do_sample_frames": False, "do_resize": False},
            )
        return request

    def call(self, key, text, videos, evidence, schema, cap, arms, validate, references=None, output_schema=None):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        request = self.request(text, videos)
        compact_output = output_schema is not None or schema.__name__ in {"Comparison", "ObservationFacts", "LocalPlan"}
        output_schema = deepcopy(output_schema) if output_schema is not None else citation_schema(schema, references)
        grammar = compact_grammar(output_schema) if compact_output else None
        specification = {
            "text": text,
            "system": SYSTEM,
            "rendered_prompt": request["prompt"],
            "evidence": evidence,
            "schema": output_schema,
            "output_cap": cap,
            "arms": arms,
        }
        if compact_output:
            specification["structured_output"] = {"mode": "compact_ebnf_v2", "grammar_sha256": digest(grammar)}
        request_hash = digest(specification)
        reused = reuse_response(self.reuse_index, key, specification, validate)
        if reused is not None:
            self.ledger.record_reuse(key, self.reuse_index[key])
            value, record = reused
            return value, {**record, "response_path": self.reuse_index[key]["path"]}
        prior = self.ledger.attempts_for(key)
        if prior and prior[-1]["request_sha256"] != request_hash:
            raise ValueError("Cached request provenance mismatch")
        if prior and prior[-1]["status"] in {"complete", "invalid"}:
            path = self.directory / "attempts" / f"{prior[-1]['number']:05d}.json"
            record = read_json(path)
            if digest(record) != prior[-1]["record_sha256"]:
                raise ValueError("Cached response drift")
            if record["status"] == "invalid":
                raise SemanticFailure(record["error"])
            return validate(record["parsed"]), {**record, "response_path": str(path)}
        # Use vLLM's actual multimodal expansion before reserving a request.
        processed = self.engine.llm_engine.input_processor.input_preprocessor.preprocess(deepcopy(request))
        prompt_tokens = len(processed["prompt_token_ids"])
        del processed
        if prompt_tokens + cap > ENGINE_CONFIG["max_model_len"]:
            raise BudgetStop("Model context cannot hold this exact evidence and output reservation")
        number = self.ledger.reserve(key, prompt_tokens, cap, arms, request_hash, sum(len(v["video"]) for v in videos))
        record = {
            "key": key,
            "attempt": number,
            "request_sha256": request_hash,
            "specification": specification,
            "started_at": utc_now(),
        }
        write_json(
            getattr(self, "status_path", self.directory / "status.json"),
            {"status": "generating", "key": key, "updated_at": utc_now()},
        )
        before = time.monotonic()
        try:
            memory_before = gpu_memory_probe()
            output = self.engine.generate(
                [request],
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=cap,
                    structured_outputs=(
                        StructuredOutputsParams(grammar=grammar)
                        if compact_output
                        else StructuredOutputsParams(json=output_schema)
                    ),
                ),
                use_tqdm=False,
            )[0]
            answer = output.outputs[0]
            record.update(
                prompt_tokens=len(output.prompt_token_ids),
                output_tokens=len(answer.token_ids),
                text=answer.text,
                finish_reason=answer.finish_reason,
                inference_seconds=time.monotonic() - before,
                gpu_memory_before=memory_before,
                gpu_memory_after=gpu_memory_probe(),
            )
        except Exception as error:
            record.update(
                status="runtime_error",
                error=f"{type(error).__name__}: {error}",
                inference_seconds=time.monotonic() - before,
            )
            write_json(self.directory / "attempts" / f"{number:05d}.json", record)
            self.ledger.settle(number, record)
            raise RuntimeError("Model runtime failed; preserve attempt and use bounded process recovery") from error
        try:
            if answer.finish_reason == "length":
                raise ValueError("Output truncated at the fixed cap")
            record["parsed"] = schema.model_validate_json(answer.text).model_dump()
            value = validate(record["parsed"])
            record["status"] = "complete"
        except ValueError as error:
            record.update(status="invalid", error=str(error))
        write_json(self.directory / "attempts" / f"{number:05d}.json", record)
        self.ledger.settle(number, record)
        if record["status"] == "invalid":
            raise SemanticFailure(record["error"])
        return value, {**record, "response_path": str(self.directory / "attempts" / f"{number:05d}.json")}

import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import argparse
import json
from videoarm.qvhighlights.common import load_manifest, read_jsonl
from videoarm.qvhighlights.contracts import validate_predictions

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--task", choices=["mr", "hd"], required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--allow-empty", action="store_true")
    a = ap.parse_args()
    print(
        json.dumps(
            validate_predictions(
                read_jsonl(a.predictions),
                load_manifest(a.manifest),
                a.task,
                allow_partial=a.allow_partial,
                allow_empty=a.allow_empty,
            )
        )
    )

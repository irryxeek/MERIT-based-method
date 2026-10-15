"""Re-parse historical responses diagnostically; never rewrite v1 run outcomes."""

import _bootstrap  # noqa: F401
import json
import os
import sqlite3
from pathlib import Path

from videoarm.qvhighlights.common import ARTIFACTS, atomic_json
from videoarm.qvhighlights.pipeline import parse_action

root = Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
cases = []
for p in sorted((root / "runs").glob("*v1*/state.sqlite")):
    db = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    metadata = json.loads(
        db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
    )
    for r in db.execute(
        "SELECT qid,stage,attempt,request,response FROM requests WHERE stage LIKE 'controller_%' AND status='completed'"
    ):
        response = json.loads(r["response"])
        if response.get("parse_error") != "controller action schema mismatch":
            continue
        context = json.loads(json.loads(r["request"])["text"].splitlines()[0])
        try:
            parsed = parse_action(
                json.loads(response["raw"]),
                context["duration"],
                metadata["config"]["task"],
            )
            accepted, error = True, None
        except ValueError as e:
            parsed, accepted, error = None, False, str(e)
        cases.append(
            {
                "run": p.parent.name,
                "qid": r["qid"],
                "stage": r["stage"],
                "attempt": r["attempt"],
                "raw": response["raw"],
                "accepted_by_v2_stop_contract": accepted,
                "parsed": parsed,
                "error": error,
            }
        )
    db.close()
result = {
    "diagnostic_only": True,
    "original_v1_runs_modified": False,
    "historical_schema_failures": len(cases),
    "accepted_by_v2_stop_contract": sum(
        c["accepted_by_v2_stop_contract"] for c in cases
    ),
    "cases": cases,
    "interpretation": "This is parser verification on saved raw responses, not recovered predictions or new inference quality.",
}
atomic_json(ARTIFACTS / "checks/stop_contract_reparse_v2.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "cases"}))

"""Record an existing Python environment without installing pip or dependencies."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from packaging.requirements import Requirement

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
packages = {d.metadata["Name"]: d for d in importlib.metadata.distributions()}
versions = {n.lower().replace("_", "-"): d.version for n, d in packages.items()}
conflicts = []
for name, d in packages.items():
    for raw in d.requires or []:
        try:
            r = Requirement(raw)
            if r.marker and not r.marker.evaluate({"extra": ""}):
                continue
            installed = versions.get(r.name.lower().replace("_", "-"))
            if installed is None or installed not in r.specifier:
                conflicts.append(
                    {"package": name, "requirement": raw, "installed": installed}
                )
        except ValueError:
            conflicts.append({"package": name, "unparsed_requirement": raw})
out = Path(a.out)
out.write_text(
    "\n".join(
        f"{name}=={packages[name].version}" for name in sorted(packages, key=str.lower)
    )
    + "\n"
)
out.with_suffix(".json").write_text(
    json.dumps(
        {
            "python": platform.python_version(),
            "packages": {n: d.version for n, d in packages.items()},
            "dependency_conflicts": conflicts,
            "note": "importlib metadata snapshot; does not install dependencies or prove GPU compatibility",
        },
        indent=2,
    )
)
print(json.dumps({"packages": len(packages), "dependency_conflicts": conflicts}))

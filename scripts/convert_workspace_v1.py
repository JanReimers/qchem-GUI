"""Convert a qviz-workspace/1 .qproj.h5 (flat method/basis/symmetry/multiplicity attrs) to /2 (one RunSpec per run).

    .venv/bin/python scripts/convert_workspace_v1.py OLD.qproj.h5 NEW.qproj.h5

One-off, by design: the app reads /2 only (no compatibility reader).  A /1 run was always SOLID² (or the analytic
stand-in), so it converts to a SOLID² run with its basis STATED, so it re-runs exactly what it ran.
"""
import pathlib
import sys

import h5py

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qviz.project import WORKSPACE_FORMAT   # noqa: E402
from qviz.spec import Accuracy, molecule    # noqa: E402

_THEORY = {"HF": ("HF", None), "LDA": ("DFT", "LDA"), "Xalpha": ("DFT", "Xalpha")}


def convert(src: str, dst: str) -> int:
    with h5py.File(src, "r") as a, h5py.File(dst, "w") as b:
        if str(a.attrs.get("format")) != "qviz-workspace/1":
            raise SystemExit(f"{src} is not a qviz-workspace/1 file")
        b.attrs.update(format=WORKSPACE_FORMAT, title=a.attrs.get("title", "Untitled"),
                       created=a.attrs.get("created", ""), selected=a.attrs.get("selected", -1))
        g = b.create_group("runs")
        for key in sorted(a["runs"]):
            r = a["runs"][key]
            theory, functional = _THEORY[str(r.attrs["method"])]
            mult = int(r.attrs.get("multiplicity", 0)) or None
            scf = {"impose_symmetry": bool(r.attrs.get("symmetry", False))}
            spec = molecule(str(r.attrs["label"]), [int(z) for z in r["numbers"][:]], [float(x) for x in r["positions"][:]],
                            theory=theory, functional=functional, multiplicity=mult,
                            accuracy=Accuracy(representation={"basis": str(r.attrs["basis"])}, scf=scf))
            g.create_group(key).attrs.update(spec=spec.to_json(), backend="SOLID²",
                                             energy=r.attrs.get("energy"), status="pending")
        return len(a["runs"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"converted {convert(sys.argv[1], sys.argv[2])} run(s) -> {sys.argv[2]}")

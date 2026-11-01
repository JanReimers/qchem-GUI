"""
qviz.runner -- the ONE place an external code is launched (doc/RunSpecPlan.md §4).

⛔ Every oracle on this box (CP2K, QE, ABINIT, VASP) is an MPI build, and a bare binary HANGS (the OpenMPI
singleton-init hang).  So a translator's Job.command never carries a launcher: `MpiRunner` prepends
`mpirun -np N` (`extra_args` for a code that needs flags after its argv), and there is no other code path that executes
an oracle.  Discovery checks a binary's presence (os.access), never runs it.

Run data has one home (D-RUNDATA): ~/Code/<app>-runs/<Material>/<prefix>.<ext>, the files given a self-describing
prefix because several runs share a Material directory.  The resolved record (<prefix>.record.json: spec, resolved
settings, command) is written BEFORE the process starts, so a crash still leaves it.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

def run_dir(app: str, material: str) -> pathlib.Path:
    """~/Code/<app>-runs/<Material>/ (created), the convention qchem's scripts/rundir implements ($RUNDIR_ROOT
    replaces ~/Code, as it does there -- the tests point it at a temporary directory)."""
    root = pathlib.Path(os.environ.get("RUNDIR_ROOT", pathlib.Path.home() / "Code"))
    d = root / f"{app}-runs" / _safe(material)
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_prefix(d: pathlib.Path, stem: str) -> str:
    """<stem>.rNNN -- the next number not yet used in `d` (a number is never reused)."""
    stem = _safe(stem)
    used = [int(m.group(1)) for p in d.glob(f"{stem}.r*.*") if (m := re.match(rf"{re.escape(stem)}\.r(\d+)\.", p.name))]
    return f"{stem}.r{max(used, default=0) + 1:03d}"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", s).strip("_") or "run"


@dataclass(frozen=True)
class MpiRunner:
    """Launch `command` under mpirun in `cwd`, streaming stdout line by line to `on_line` and into `<prefix>.out`."""
    ranks: int = 1
    extra_args: tuple[str, ...] = ()          # appended AFTER the program argv
    threads: int = 1                          # OMP_NUM_THREADS; 1 keeps runs comparable to the banked serial timings

    def argv(self, command: tuple[str, ...]) -> list[str]:
        assert command, "an empty command"
        return ["mpirun", "-np", str(self.ranks), *command, *self.extra_args]

    def run(self, command: tuple[str, ...], cwd: pathlib.Path, prefix: str, files: dict[str, bytes], record: dict,
            on_line: Optional[Callable[[str], None]] = None) -> tuple[int, pathlib.Path]:
        cwd.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (cwd / name).write_bytes(data)
        argv = self.argv(command)
        (cwd / f"{prefix}.record.json").write_text(json.dumps(record | {"argv": argv, "cwd": str(cwd)}, indent=2, default=str))
        out = cwd / f"{prefix}.out"
        env = os.environ | {"OMP_NUM_THREADS": str(self.threads)}
        with out.open("w") as log, subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            try:
                for line in p.stdout:
                    log.write(line)
                    if on_line:
                        on_line(line)
            except BaseException:            # the stream asked to stop (RunStopped) or broke: never leave the code running
                p.terminate()                # SIGTERM: mpirun forwards it to its ranks
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                raise
        return p.returncode, out

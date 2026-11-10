"""
qviz.states -- where a run's SAVED SCF STATE lives, and how much disk the saved states take.

A code that can save its converged (or last) iterate writes it to a file this module names; the Result records the
path (Result.state_file), the workspace keeps it, and a later run can CONTINUE from it -- a different accelerator, tighter
gates -- instead of starting from scratch.  Today that is SOLID²: a crystal through its deck (state.save / state.restartFrom), a
molecule through the binding (Calculator.save_state / restart_from); both are Calculation::SaveState files ("qchem-state 2.0").

The files are the app's own, under STATE_DIR (QCHEM_GUI_STATES overrides it: tests).  A file is named for the deck that
wrote it (label + a hash of the deck), so running the same deck again overwrites its own file instead of piling up
copies.  States are large and accumulate: the Runs panel shows each run's size, the total and the disk's free space, and
removing a run (or "Delete saved state") deletes its file when no other run in the workspace points at it.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
from typing import Iterable, Optional


def state_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("QCHEM_GUI_STATES") or pathlib.Path.home() / ".local/share/qchem-GUI/states")


def path_for(label: str, key: str) -> str:
    """The state file for a run of `label` whose input text is `key` (the deck): the same deck, the same file."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "run"
    return str(state_dir() / f"{stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.h5")


def size(path: Optional[str]) -> int:
    """Bytes on disk (0: no path, or the file is gone)."""
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def of(run) -> Optional[str]:
    """The run's saved state file when it is still on disk, else None."""
    p = getattr(run.result, "state_file", None) if run.result is not None else None
    return p if p and os.path.exists(p) else None


def usage(runs: Iterable) -> tuple[int, int]:
    """(files, bytes) of the saved states these runs point at -- a file shared by two runs counted once."""
    files = {p for p in (of(r) for r in runs) if p}
    return len(files), sum(size(p) for p in files)


def free_bytes() -> int:
    """Free space on the disk the states go to."""
    d = state_dir()
    while not d.exists() and d != d.parent:
        d = d.parent
    return shutil.disk_usage(d).free


def release(run, others: Iterable) -> Optional[str]:
    """Delete `run`'s state file unless another of `others` points at it; returns the path deleted (None: kept / none)."""
    p = of(run)
    if p is None or any(of(o) == p for o in others if o is not run):
        return None
    os.remove(p)
    run.result.state_file = None
    return p


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"

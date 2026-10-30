"""
qviz.capability -- what a back end can do, as DATA (doc/RunSpecPlan.md §2).

Option 2.2: the user states WHAT (a RunSpec); each back end's Capabilities decide
whether it can run it.  `can_run` collects EVERY reason it refuses, not the first, so
the back-end strip can say "CP2K: no orbital-resolved U; no SCAN in this build".

A capability is earned by a parse-tested run, not by a manual page: the files that fill
these are written by each adapter's tests (stage S2+).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .spec import RunSpec


@dataclass(frozen=True)
class Capability:
    """One feature: `check(spec)` is True when the spec does NOT need what this back end lacks."""
    feature: str                                  # dotted key into the spec, e.g. "physics.hubbard.form=Liechtenstein"
    check: Callable[[RunSpec], bool]              # True = this spec is fine on this feature
    reason: str                                   # shown when check fails: "no orbital-resolved U"
    note: str = ""                                # shown on hover when it passes: "linear response via hp.x"


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Capabilities:
    backend: str
    features: list[Capability] = field(default_factory=list)
    installed: bool = True                        # discovery found the executable / extension
    why_absent: Optional[str] = None

    def can_run(self, spec: RunSpec) -> Verdict:
        if not self.installed:
            return Verdict(False, (self.why_absent or f"{self.backend} is not installed",))
        reasons = tuple(c.reason for c in self.features if not c.check(spec))
        return Verdict(not reasons, reasons)


def filter_backends(spec: RunSpec, caps: list[Capabilities]) -> dict[str, Verdict]:
    """The back-end strip: one verdict per back end, for one spec."""
    return {c.backend: c.can_run(spec) for c in caps}

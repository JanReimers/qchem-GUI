"""
qviz.policyinfo -- the decisions the library's POLICIES made for a run, as its run record states them.

Every policy reports itself (CLAUDE.md framework-vs-efficiency; doc/OpenWork.md, "Every existing policy reports itself under
policy"): each run of the record carries a `policy` array, one row per decision in firing order,
    {name, decided, verdict: refuse | resource | warn | choose, reason, cost?: {wallSeconds?, memoryBytes?}}
The library owns that contract; this module only reads it, never rebuilds a decision from the spec.  A refusal leads
(refuse, resource), a warn-and-run follows with its cost estimate, and `choose` -- an automatic setting a policy picked
because the user stated none -- is listed apart, quietly: it is how the user finds every policy that acted, not an alarm.
An absent or empty array means no policy acted.
"""
from __future__ import annotations

from typing import Optional

#: lead order: a refusal first, then a warning; `choose` is listed apart
LOUD = ("refuse", "resource", "warn")
_MARK = {"refuse": "✗ refused", "resource": "✗ refused on resources", "warn": "⚠ warning"}


def decisions(record: Optional[dict]) -> list[dict]:
    """Every policy row of the record, in filing order, each with the `run` that filed it; a row two runs repeat once."""
    out, seen = [], set()
    for key, run in (record or {}).items():
        for row in (run or {}).get("policy") or []:
            ident = (row.get("name"), row.get("decided"), row.get("verdict"))
            if ident not in seen:
                seen.add(ident)
                out.append({**row, "run": key.rsplit("@", 1)[0]})
    return out


def _cost(c: Optional[dict]) -> str:
    """'840 s, 3.2 GB' from a cost estimate ('' when none)."""
    if not c:
        return ""
    parts = []
    if isinstance(c.get("wallSeconds"), (int, float)):
        s = c["wallSeconds"]
        parts.append(f"{s:.0f} s" if s < 600 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h")
    if isinstance(c.get("memoryBytes"), (int, float)):
        b = c["memoryBytes"]
        parts.append(f"{b / 2**30:.1f} GB" if b >= 2**30 else f"{b / 2**20:.0f} MB")
    return ", ".join(parts)


def line(d: dict) -> str:
    """One decision as the user reads it: what was decided, why, the cost, and the policy's own name (to grep for)."""
    cost = _cost(d.get("cost"))
    if cost:
        cost = f" (would need {cost})" if d.get("verdict") == "resource" else f" (estimated {cost})"
    head = _MARK.get(d.get("verdict"), "")
    return (f"{head}: " if head else "") + f"{d.get('decided', '?')}" + (f" -- {d['reason']}" if d.get("reason") else "") + cost \
        + (f"  [{d['name']}]" if d.get("name") else "")


def lines(record: Optional[dict]) -> tuple[list[str], list[str]]:
    """(the loud lines -- refusals, then warnings -- , the automatic choices), each in filing order."""
    ds = decisions(record)
    loud = [line(d) for v in LOUD for d in ds if d.get("verdict") == v]
    chosen = [line(d) for d in ds if d.get("verdict") not in LOUD]
    return loud, chosen

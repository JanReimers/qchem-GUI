"""
qviz.draft -- the ONE editable spec the user and the agent share (doc/RunSpecPlan.md §5.1).

Both authors edit through the same calls; every change is recorded as (author, rationale, before, after) on one
undo stack, so an agent edit is a diff the user sees and can revert, and the agent holds no state that is not in
the draft or in the chat.  A change that leaves an invalid spec is REFUSED (the draft never holds a spec that
`validate` rejects) and the refusal names every defect -- that text is what the agent reads back.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .spec import RunSpec, SpecError


@dataclass(frozen=True)
class Edit:
    author: str                       # "user" | "agent"
    rationale: str
    before: Optional[RunSpec]
    after: RunSpec

    def diff(self) -> list[tuple[str, Any, Any]]:
        return spec_diff(self.before, self.after)


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7386 JSON merge patch: an object merges key by key, null deletes, anything else replaces."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out = dict(target) if isinstance(target, dict) else {}
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge_patch(out.get(k), v)
    return out


def _flatten(x: Any, path: str = "") -> dict[str, Any]:
    if isinstance(x, dict) and x:
        out = {}
        for k, v in x.items():
            out.update(_flatten(v, f"{path}.{k}" if path else k))
        return out
    return {path: x}


def spec_diff(a: Optional[RunSpec], b: RunSpec) -> list[tuple[str, Any, Any]]:
    """(dotted path, old, new) for every leaf that differs; lists compare whole."""
    fa, fb = _flatten(a.to_dict()) if a else {}, _flatten(b.to_dict())
    return [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]


class SpecDraft:
    def __init__(self, spec: Optional[RunSpec] = None):
        self._spec = spec
        self._edits: list[Edit] = []
        self._listeners: list[Callable[["SpecDraft", Optional[Edit]], None]] = []

    @property
    def spec(self) -> Optional[RunSpec]:
        return self._spec

    @property
    def history(self) -> list[Edit]:
        return list(self._edits)

    def on_change(self, cb: Callable[["SpecDraft", Optional[Edit]], None]) -> None:
        self._listeners.append(cb)

    def _commit(self, new: RunSpec, author: str, rationale: str) -> Edit:
        e = Edit(author, rationale, self._spec, new)
        self._edits.append(e)
        self._spec = new
        for cb in self._listeners:
            cb(self, e)
        return e

    def set(self, spec: RunSpec, author: str, rationale: str = "") -> Edit:
        """Replace the draft (a new structure, a spec the user typed).  Raises SpecError when `spec` is invalid."""
        return self._commit(spec.validate(), author, rationale)

    def patch(self, patch: dict, author: str, rationale: str = "") -> Edit:
        """Merge-patch the draft.  Raises SpecError (the draft unchanged) when the result does not read or validate."""
        if self._spec is None:
            raise SpecError("the draft is empty: build a structure first")
        new = RunSpec.from_dict(merge_patch(self._spec.to_dict(), patch)).validate()
        return self._commit(new, author, rationale)

    def undo(self) -> Optional[Edit]:
        """Revert the last edit, whoever made it; returns it (None when there is nothing to undo)."""
        if not self._edits:
            return None
        e = self._edits.pop()
        self._spec = e.before
        for cb in self._listeners:
            cb(self, None)
        return e

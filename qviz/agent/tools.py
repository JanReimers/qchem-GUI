"""
qviz.agent.tools -- the agent's tool set (doc/RunSpecPlan.md §5.2) over the functions the GUI already has.

Three kinds, and the split IS the safety model (§5.3):
  * READ tools answer questions and change nothing (the spec, validation, the back-end strip, the resolved settings,
    local structure lookup, run results, the curated notes);
  * WRITE tools change only the shared SpecDraft, as an agent-authored edit the user sees and can undo;
  * the RUN tool only puts a RunRequest card in the chat -- `accept()` is wired to the user's click, never to the agent.
There is no tool that starts, cancels or deletes a run, writes a file, or launches a binary.

A tool result is a dict.  `is_result` marks run outputs (the consent policy may keep those on the box); a result
from a licensed back end (VASP) is flagged `licensed` and never leaves the box.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import compare, molecules
from ..backends import solid2_deck
from ..backends.base import BackendRegistry
from ..spec import Z_OF, RunSpec, SpecError, molecule
from ..workspace import Workspace
from .client import ToolSchema
from ..draft import SpecDraft

LICENSED_BACKENDS = {"VASP"}

#: Banked experience the agent cites instead of inventing advice (search_docs).  Each note names its source.
NOTES = (
    ("MnO AFM-II free run", "Kerker + Pulay density mixing converges a FREE MnO AFM-II run and holds the order; the Ladder/GDM/"
     "MOM stack fights the smeared d frontier.  Recipe: qchem decks/MnO_AFM2_free.json (CLAUDE.md, doc/Benchmark.md)."),
    ("restart accelerator", "A restart from a converged or coarse-grid density should use GDM (2 iterations on NaF); DIIS steps the "
     "density away first (qchem CLAUDE.md, measured 2026-10-02)."),
    ("molecular XC mesh", "The molecular facade's default XC mesh (30 radial, 12 directions) is 0.52 Ha/atom off on F2; the GUI "
     "states a mesh from epsilon (80/29 for 1e-6 <= eps < 1e-3)."),
    ("cross-code basis", "CP2K is handed SOLID2's own basis file converted; d shells are refused (spherical vs Cartesian spans)."),
)


@dataclass
class RunRequest:
    """A run-request card: what the agent ASKS to run.  Only `accept`, wired to the user's click, starts anything."""
    id: int
    spec: RunSpec
    backends: list[str]
    purpose: str
    state: str = "waiting"                      # waiting | accepted | declined
    _start: Optional[Callable[["RunRequest"], None]] = field(default=None, repr=False)

    def accept(self) -> None:
        assert self.state == "waiting", self.state
        self.state = "accepted"
        if self._start:
            self._start(self)

    def decline(self) -> None:
        self.state = "declined"


def _schema(props: dict, required=()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


class Toolset:
    def __init__(self, draft: SpecDraft, registry: BackendRegistry, workspace: Optional[Workspace] = None,
                 start_run: Optional[Callable[[RunRequest], None]] = None):
        """`start_run` is what a card's accept() calls (the GUI: add the draft's spec on each named back end)."""
        self.draft, self.registry, self.workspace = draft, registry, workspace
        self._start_run = start_run
        self.cards: list[RunRequest] = []
        self._card_ids = itertools.count(1)
        self._card_listeners: list[Callable[[RunRequest], None]] = []
        self._tools: dict[str, tuple[ToolSchema, Callable[..., dict]]] = {}
        R, W = "Read only.", "Edits the shared draft (the user sees the diff and can undo it)."
        self._add("get_spec", f"{R} The current draft spec as JSON, or null.", _schema({}), self.get_spec)
        self._add("validate_spec", f"{R} Validate the draft, or a full spec given as JSON.",
                  _schema({"spec": {"type": "object"}}), self.validate_spec)
        self._add("list_backends", f"{R} The back ends, whether installed, and whether synthetic.", _schema({}), self.list_backends)
        self._add("can_run", f"{R} Per back end: can it run the draft, and if not why.", _schema({}), self.can_run)
        self._add("explain_resolved", f"{R} The native settings a back end would use for the draft (epsilon -> cutoffs, meshes).",
                  _schema({"backend": {"type": "string"}}, ["backend"]), self.explain_resolved)
        self._add("lookup_structure", f"{R} Search the LOCAL structure lists (molecule presets, qchem materials.json).",
                  _schema({"query": {"type": "string"}}, ["query"]), self.lookup_structure)
        self._add("list_runs", f"{R} The runs in the workspace: id, label, back end, status.", _schema({}), self.list_runs)
        self._add("get_result", f"{R} A run's parsed result: status, energy, iterations, resolved settings.",
                  _schema({"run_id": {"type": "integer"}}, ["run_id"]), self.get_result)
        self._add("compare_results", f"{R} The cross-code table for a run and its legs (same spec on other back ends).",
                  _schema({"run_id": {"type": "integer"}}, ["run_id"]), self.compare_results)
        self._add("search_docs", f"{R} The curated notes of banked experience.", _schema({"query": {"type": "string"}}, ["query"]),
                  self.search_docs)
        self._add("propose_spec_patch", f"{W} Apply a JSON merge patch (RFC 7386) to the draft, with the reason.",
                  _schema({"patch": {"type": "object"}, "rationale": {"type": "string"}}, ["patch", "rationale"]), self.propose_spec_patch)
        self._add("build_structure", f"{W} Replace the draft's structure: a preset molecule by name; a molecule no preset has, "
                  "from explicit atoms (element + Cartesian position in angstrom; the geometry is then yours, so say it is "
                  "approximate); or a crystal prototype (rocksalt | diamond | fcc) with elements and the cubic lattice "
                  "constant a_bohr.  A new structure starts as Hartree-Fock: set the method with propose_spec_patch.",
                  _schema({"kind": {"enum": ["molecule", "crystal"]}, "name": {"type": "string"},
                           "atoms": {"type": "array", "items": {"type": "object", "properties": {
                               "element": {"type": "string"}, "xyz_angstrom": {"type": "array", "items": {"type": "number"},
                                                                                 "minItems": 3, "maxItems": 3}},
                               "required": ["element", "xyz_angstrom"]}},
                           "prototype": {"enum": ["rocksalt", "diamond", "fcc"]}, "elements": {"type": "array", "items": {"type": "string"}},
                           "a_bohr": {"type": "number"}}, ["kind"]), self.build_structure)
        self._add("undo_last_edit", f"{W} Revert the last draft edit (either author's).", _schema({}), self.undo_last_edit)
        self._add("request_run", "Puts a run-request card in the chat for the draft on the named back ends (empty = the "
                  "strip's choice).  It does NOT start anything: only the user's click on the card does.",
                  _schema({"backends": {"type": "array", "items": {"type": "string"}}, "purpose": {"type": "string"}}, ["purpose"]),
                  self.request_run)

    # -- plumbing -------------------------------------------------------------
    def _add(self, name, description, schema, fn):
        self._tools[name] = (ToolSchema(name, description, schema), fn)

    @property
    def schemas(self) -> list[ToolSchema]:
        return [s for s, _ in self._tools.values()]

    def on_card(self, cb: Callable[[RunRequest], None]) -> None:
        self._card_listeners.append(cb)

    def call(self, name: str, args: dict) -> dict:
        """Run tool `name`.  Never raises for a bad call: the error is the result, so the model can correct itself."""
        if name not in self._tools:
            return {"error": f"no tool {name!r}; tools: {sorted(self._tools)}"}
        try:
            return self._tools[name][1](**args)
        except (SpecError, TypeError, KeyError, ValueError) as e:
            return {"error": f"{name}: {e}"}

    def _need_draft(self) -> RunSpec:
        if self.draft.spec is None:
            raise SpecError("the draft is empty: build a structure first")
        return self.draft.spec

    def _run(self, run_id: int):
        if self.workspace is None or not 0 <= run_id < len(self.workspace.runs):
            raise KeyError(f"no run {run_id}")
        return self.workspace.runs[run_id]

    # -- read -----------------------------------------------------------------
    def get_spec(self) -> dict:
        s = self.draft.spec
        return {"spec": s.to_dict() if s else None, "summary": s.summary if s else "the draft is empty"}

    def validate_spec(self, spec: Optional[dict] = None) -> dict:
        s = RunSpec.from_dict(spec) if spec is not None else self._need_draft()
        s.validate()
        return {"valid": True, "summary": f"valid: {s.summary}"}

    def list_backends(self) -> dict:
        rows = []
        for n in self.registry.names():
            c = self.registry.get(n).capabilities()
            rows.append({"name": n, "installed": c.installed, "synthetic": c.synthetic, "why_absent": c.why_absent})
        return {"backends": rows, "summary": ", ".join(r["name"] + ("" if r["installed"] else " (absent)") for r in rows)}

    def can_run(self) -> dict:
        s = self._need_draft()
        v = self.registry.verdicts(s)
        rows = {n: {"ok": bool(x), "reasons": list(x.reasons)} for n, x in v.items()}
        return {"verdicts": rows, "summary": f"{s.summary}: " + "; ".join(
            f"{n} {'can run it' if r['ok'] else 'cannot (' + ', '.join(r['reasons']) + ')'}" for n, r in rows.items())}

    def explain_resolved(self, backend: str) -> dict:
        s = self._need_draft()
        verdict = self.registry.verdicts(s)[backend]
        if not verdict:
            return {"error": f"{backend} cannot run the draft: {', '.join(verdict.reasons)}"}
        r = self.registry.get(backend).translate(s).resolved
        return {"resolved": r, "summary": f"{backend}: " + "; ".join(f"{k} = {v}" for k, v in r.items())}

    def lookup_structure(self, query: str) -> dict:
        q = query.lower()
        mols = [n for n in molecules.MOLECULES if q in n.lower()]
        try:
            mats = [(n, e.get("comment", "")) for n, e in solid2_deck.materials().items() if not n.startswith("_")
                    and (q in n.lower() or q in e.get("comment", "").lower())]
        except OSError:
            mats = []
        return {"molecules": mols, "materials": [{"name": n, "comment": c} for n, c in mats],
                "summary": f"molecules: {', '.join(mols) or 'none'}; materials: {', '.join(n for n, _ in mats) or 'none'}"}

    def list_runs(self) -> dict:
        runs = [{"run_id": i, "label": r.spec.label, "backend": r.backend_name, "status": r.status.name.lower()}
                for i, r in enumerate(self.workspace.runs if self.workspace else [])]
        return {"runs": runs, "summary": "; ".join(f"#{x['run_id']} {x['label']} on {x['backend']} ({x['status']})" for x in runs)
                or "no runs"}

    def get_result(self, run_id: int) -> dict:
        r = self._run(run_id)
        out = {"is_result": True, "run_id": run_id, "backend": r.backend_name, "status": r.status.name.lower(),
               "licensed": r.backend_name in LICENSED_BACKENDS}
        if r.is_ready:
            res = r.result
            out |= {"converged": res.converged, "energy_Ha": res.energy_Ha, "iterations": len(res.scf), "resolved": res.resolved}
            out["summary"] = f"run #{run_id} on {r.backend_name}: E = {res.energy_Ha} Ha, {len(res.scf)} iterations"
        else:
            out["summary"] = f"run #{run_id} on {r.backend_name} is {out['status']}"
        return out

    def compare_results(self, run_id: int) -> dict:
        legs = compare.legs(self.workspace, self._run(run_id))
        heads, rows = compare.code_table(legs)
        return {"is_result": True, "licensed": any(r.backend_name in LICENSED_BACKENDS for r in legs),
                "heads": heads, "rows": [[k, *v] for k, v in rows],
                "summary": " | ".join(heads) + "\n" + "\n".join(f"{k}: " + " | ".join(v) for k, v in rows)}

    def search_docs(self, query: str) -> dict:
        words = query.lower().split()
        hits = [{"title": t, "text": x} for t, x in NOTES if any(w in (t + " " + x).lower() for w in words)]
        return {"notes": hits, "summary": "\n".join(f"{h['title']}: {h['text']}" for h in hits) or "no note matches"}

    # -- write (the draft only) -----------------------------------------------
    def propose_spec_patch(self, patch: dict, rationale: str) -> dict:
        e = self.draft.patch(patch, "agent", rationale)
        return {"applied": True, "diff": [list(d) for d in e.diff()],
                "summary": "draft: " + "; ".join(f"{p} {o!r} -> {n!r}" for p, o, n in e.diff()) + f" ({rationale})"}

    def build_structure(self, kind: str, name: str = "", prototype: str = "", elements: tuple = (), a_bohr: float = 0.0,
                        atoms: tuple = ()) -> dict:
        if kind == "molecule" and atoms:
            unknown = [a["element"] for a in atoms if a["element"] not in Z_OF]
            if unknown:
                return {"error": f"unknown elements {unknown}"}
            spec = molecule(name or "molecule", [Z_OF[a["element"]] for a in atoms],
                            [x * molecules._A for a in atoms for x in a["xyz_angstrom"]])
            e = self.draft.set(spec, "agent", f"build_structure {name or 'molecule'} from {len(atoms)} given atoms")
            return {"applied": True, "summary": f"draft is now {spec.summary} ({len(atoms)} atoms, geometry as given -- "
                                                "not a preset or database structure)", "diff_paths": len(e.diff())}
        if kind == "molecule":
            key = next((k for k in molecules.MOLECULES if name.lower() in k.lower()), None)
            if key is None:
                return {"error": f"no preset molecule matches {name!r}; presets: {list(molecules.MOLECULES)}"}
            spec = molecules.preset_spec(key)
        else:
            spec = _crystal(prototype, list(elements), a_bohr)
        e = self.draft.set(spec, "agent", f"build_structure {kind} {name or prototype}")
        return {"applied": True, "summary": f"draft is now {spec.summary} ({len(spec.structure.sites)} sites)",
                "diff_paths": len(e.diff())}

    def undo_last_edit(self) -> dict:
        e = self.draft.undo()
        return {"summary": f"reverted the {e.author}'s edit ({e.rationale})" if e else "nothing to undo"}

    # -- run (a card, never a start) ------------------------------------------
    def request_run(self, purpose: str, backends: tuple = ()) -> dict:
        s = self._need_draft()
        names = list(backends) or [n for n, (_able, ticked, _tip) in self.registry.strip(s).items() if ticked]
        verdicts = self.registry.verdicts(s)
        bad = {n: list(verdicts[n].reasons) for n in names if n in verdicts and not verdicts[n]}
        unknown = [n for n in names if n not in verdicts]
        if unknown or bad or not names:
            return {"error": f"cannot request: unknown back ends {unknown}" if unknown else
                    f"cannot request: {bad}" if bad else "no back end can run the draft"}
        card = RunRequest(next(self._card_ids), s, names, purpose, _start=self._start_run)
        self.cards.append(card)
        for cb in self._card_listeners:
            cb(card)
        return {"card": card.id, "summary": f"run-request card #{card.id} ({s.summary} on {', '.join(names)}) waits for the user's click"}


def _crystal(prototype: str, elements: list[str], a: float) -> RunSpec:
    h = a / 2
    fcc = ((0.0, h, h), (h, 0.0, h), (h, h, 0.0))
    sites = {"rocksalt": [(0, 0, 0), (0.5, 0.5, 0.5)], "diamond": [(0, 0, 0), (0.25, 0.25, 0.25)], "fcc": [(0, 0, 0)]}
    if prototype not in sites:
        raise ValueError(f"prototype {prototype!r} not in {sorted(sites)}")
    fr = sites[prototype]
    if len(elements) not in (1, len(fr)):
        raise ValueError(f"{prototype} has {len(fr)} sites: give 1 or {len(fr)} elements")
    els = elements * len(fr) if len(elements) == 1 else elements
    if a <= 0:
        raise ValueError("a_bohr (the conventional cubic lattice constant) must be > 0")
    return RunSpec.from_dict({
        "label": f"{''.join(dict.fromkeys(els))} {prototype}",
        "structure": {"kind": "crystal", "lattice_bohr": [list(r) for r in fcc],
                      "sites": [{"element": e, "frac": list(f)} for e, f in zip(els, fr)]},
        "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential", "spin": {"polarized": False}},
        "accuracy": {"kmesh": [1, 1, 1]}}).validate()

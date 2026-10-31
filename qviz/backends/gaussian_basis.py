"""
qviz.backends.gaussian_basis -- ONE Gaussian basis, written in another code's format (doc/RunSpecPlan.md §3, S4).

A spec that states `representation.basis` names a SOLID² basis (qchem's BasisSetData/<name>.bsd, Gaussian94
format).  A back end that reads Gaussian bases from a file of its own (CP2K) gets THAT file, converted -- not a
look-alike chosen from its own library by name -- so a cross-code comparison compares codes, not basis sets.
The file is the single source; there is no name-mapping table to drift.

Angular form: CP2K's Gaussians are spherical, SOLID²'s molecular facade is Cartesian (MoleculeSetup Angular
default).  For s and p the two spans are identical; from d on the Cartesian set carries extra functions (the
6d s-contaminant), so a conversion with l >= 2 is REFUSED rather than silently compared across different spans.
"""
from __future__ import annotations

import functools
import os
import pathlib

_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}


def bsd_dir() -> pathlib.Path:
    """qchem's BasisSetData/ ($QCHEM_SRC, default ~/Code/qchem-main -- the same tree the CP2K oracle decks come from)."""
    return pathlib.Path(os.environ.get("QCHEM_SRC", pathlib.Path.home() / "Code/qchem-main")) / "BasisSetData"


@functools.lru_cache(maxsize=None)
def read_bsd(name: str) -> dict[str, list[tuple[int, tuple[tuple[float, float], ...]]]]:
    """BasisSetData/<name>.bsd as {ELEMENT: [(l, ((alpha, coef), ...)), ...]} in file order.  Raises OSError when the
    file is absent, ValueError on a shell type it does not know (an SP shell, say)."""
    lines = (bsd_dir() / f"{name}.bsd").read_text().splitlines()
    body = lines[lines.index("!") + 1:] if "!" in lines else lines
    out: dict = {}
    el, i = None, 0
    while i < len(body):
        tok = body[i].split()
        i += 1
        if not tok or tok[0].startswith("BASIS="):
            continue
        if tok[0] == "****":
            el = None
            continue
        if el is None:
            el = tok[0].upper()
            out[el] = []
            continue
        if tok[0].upper() not in _L:
            raise ValueError(f"{name}.bsd: shell type {tok[0]!r} for {el} is not converted")
        n = int(tok[1])
        prims = tuple((float(a.replace("D", "E")), float(c.replace("D", "E")))
                      for a, c in (body[i + k].split()[:2] for k in range(n)))
        i += n
        out[el].append((_L[tok[0].upper()], prims))
    return out


@functools.lru_cache(maxsize=None)
def pp_valence(name: str) -> dict[str, int]:
    """The `PP-PROVENANCE El=q ...` header of <name>.bsd: the GTH valence charge each block was generated for (qchem
    refuses a run whose pseudopotential differs -- Gaussian::CheckPPProvenance).  {} when the file declares none."""
    for line in (bsd_dir() / f"{name}.bsd").read_text().splitlines():
        if line.startswith("PP-PROVENANCE"):
            return {k.upper(): int(v) for k, v in (t.split("=") for t in line.split()[1:])}
    return {}


def why_not_cp2k(name: str, elements) -> str | None:
    """None if `name` converts for every element in `elements`, else the reason (for a capability)."""
    try:
        data = read_bsd(name)
    except (OSError, ValueError) as e:
        return f"basis {name}: {e}"
    missing = sorted({e.upper() for e in elements} - set(data))
    if missing:
        return f"basis {name} has no block for {', '.join(missing)}"
    high = sorted({e for e in elements if any(l >= 2 for l, _ in data[e.upper()])})
    if high:
        return (f"basis {name} has d (or higher) shells on {', '.join(high)}: CP2K is spherical, SOLID²'s molecular "
                "basis Cartesian -- the spans differ")
    return None


def max_exponent(name: str, elements) -> float:
    """The hardest primitive exponent among `elements`' blocks -- what a plane-wave / GPW grid has to resolve."""
    data = read_bsd(name)
    return max(a for e in elements for _l, prims in data[e.upper()] for a, _c in prims)


def cp2k_name(name: str) -> str:
    return f"SOLID2-{name.upper()}"


def to_cp2k(name: str, elements) -> str:
    """The CP2K BASIS_SET file text holding `name`'s blocks for `elements`, each under cp2k_name(name): one set per
    shell, `n l l nprim 1` (n = l+1: CP2K uses it only to label the atomic guess)."""
    assert why_not_cp2k(name, elements) is None, why_not_cp2k(name, elements)
    data = read_bsd(name)
    out = [f"# {name}.bsd ({bsd_dir()}), converted by qviz for a cross-code comparison"]
    for el in sorted({e.upper() for e in elements}):
        shells = data[el]
        out += [f"{el.capitalize()} {cp2k_name(name)}", f" {len(shells)}"]
        for l, prims in shells:
            out.append(f" {l + 1} {l} {l} {len(prims)} 1")
            out += [f"  {a:.10f} {c:.10f}" for a, c in prims]
    return "\n".join(out) + "\n"

"""qviz.bse -- orbital bases from the Basis Set Exchange (https://www.basissetexchange.org), kept locally (OpenWork §2).

The Exchange's public REST API lists every set (/api/metadata/) and serves each in Gaussian94 layout
(/api/basis/<name>/format/gaussian94/) with its citations (/api/references/<name>/format/txt/).  A download is UNTRUSTED
data: it lands in its own fresh directory, is read as text only, and is TRANSLATED by the library
(qchem_py.import_basis -> BasisSet::Gaussian::ImportGaussian94), which refuses what it cannot represent (an ECP block, a
scaled shell) by name.  Nothing in it is executed, and nothing in it chooses a path.

The translated file is saved under the per-user folder (~/.solid2/bases/<name>.v<version>.bsd, the ~/.solid2 ruling), with a
sidecar <same>.json holding the set's name, version, function types, source URL, fetch date and citations, so a run stays
reproducible offline and says where its basis came from.  ECP sets are not offered (there is no ECP route yet).
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from typing import Callable, Optional

from .backends import gaussian_basis
from .spec import RunSpec

API = "https://www.basissetexchange.org/api"
#: tests point this at a temp folder; None = ~/.solid2/bases
BASES_DIR: Optional[str] = None
_TIMEOUT = 30          # seconds per request
_MAX_BYTES = 64 << 20  # the largest set (all elements, high L) is a few MB: anything bigger is not a basis


class BSEError(RuntimeError):
    """A fetch that failed, or a set the library refused, in words a person can act on."""


def bases_dir() -> pathlib.Path:
    return pathlib.Path(BASES_DIR) if BASES_DIR else pathlib.Path.home() / ".solid2" / "bases"


def _get(url: str) -> bytes:
    """GET `url` (https only), at most _MAX_BYTES.  Replaced in tests: they never touch the network."""
    if not url.startswith("https://"):
        raise BSEError(f"refusing a non-https URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "qchem-GUI"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        data = r.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise BSEError(f"{url}: more than {_MAX_BYTES >> 20} MB -- not a basis set")
    return data


fetch: Callable[[str], bytes] = _get


def _url(path: str, **query) -> str:
    q = urllib.parse.urlencode({k: v for k, v in query.items() if v})
    return f"{API}/{path}" + (f"?{q}" if q else "")


_metadata: Optional[dict] = None


def metadata(refresh: bool = False) -> dict:
    """The Exchange's catalogue {key: {display_name, description, role, function_types, latest_version, versions}}, fetched
    once per session (and kept beside the bases, so the picker still lists what was seen when offline)."""
    global _metadata
    cache = bases_dir() / "bse-metadata.json"
    if _metadata is None or refresh:
        try:
            _metadata = json.loads(fetch(_url("metadata/")).decode())
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(_metadata))
        except Exception as e:                                       # noqa: BLE001 -- offline: the last catalogue seen
            if not cache.exists():
                raise BSEError(f"the Basis Set Exchange catalogue is unreachable ({e}) and none was saved before") from e
            _metadata = json.loads(cache.read_text())
    return _metadata


def _elements(v: dict) -> set[int]:
    return {int(z) for z in v.get("elements", [])}


def candidates(Zs: set[int], text: str = "") -> list[dict]:
    """The orbital sets whose latest version covers every Z in `Zs`, ECP sets left out (no ECP route), matching `text`
    (name or description, any case): [{key, name, description, version, family, spherical}], by name."""
    out = []
    needle = text.strip().lower()
    for key, m in metadata().items():
        if m.get("role") != "orbital" or "scalar_ecp" in m.get("function_types", []):
            continue
        v = m.get("versions", {}).get(m.get("latest_version", ""), {})
        if not Zs <= _elements(v):
            continue
        if needle and needle not in key.lower() and needle not in m.get("description", "").lower() \
                and needle not in m.get("display_name", "").lower():
            continue
        out.append({"key": key, "name": m.get("display_name", key), "description": m.get("description", ""),
                    "version": m.get("latest_version", ""), "family": m.get("family", ""),
                    "spherical": "gto_spherical" in m.get("function_types", [])})
    return sorted(out, key=lambda c: c["name"].lower())


def _stem(key: str, version: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", key) + f".v{version}"


def installed() -> list[dict]:
    """The downloaded sets (their sidecars), by name."""
    d = bases_dir()
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        if p.name == "bse-metadata.json":
            continue
        try:
            side = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (d / side.get("file", "")).exists():
            out.append(side | {"path": str(d / side["file"])})
    return sorted(out, key=lambda s: s["name"].lower())


def download(key: str, qc=None) -> dict:
    """Fetch set `key` (every element of its latest version) and its citations, translate it through the library, and
    save it under bases_dir(): returns its sidecar (with "path").  Raises BSEError with the Exchange's or the library's
    reason.  Call it off the GUI thread (a network round trip)."""
    m = metadata().get(key)
    if m is None:
        raise BSEError(f"'{key}' is not in the Basis Set Exchange catalogue")
    if m.get("role") != "orbital":
        raise BSEError(f"'{key}' is a {m.get('role')} set, not an orbital basis")
    if "scalar_ecp" in m.get("function_types", []):
        raise BSEError(f"'{key}' carries effective core potentials: the library has no ECP route yet")
    version = m.get("latest_version", "")
    if qc is None:
        from .backends.solid2 import _import_qchem_py
        qc, why = _import_qchem_py()
        if qc is None:
            raise BSEError(why)
    target = bases_dir()
    target.mkdir(parents=True, exist_ok=True)
    incoming = pathlib.Path(tempfile.mkdtemp(prefix=".incoming-", dir=target))   # this download's own directory
    try:
        try:
            raw = fetch(_url(f"basis/{urllib.parse.quote(key)}/format/gaussian94/", version=version))
            refs = fetch(_url(f"references/{urllib.parse.quote(key)}/format/txt/", version=version))
        except BSEError:
            raise
        except Exception as e:                                       # noqa: BLE001 -- the network's own words
            raise BSEError(f"downloading {key} from the Basis Set Exchange failed: {e}") from e
        (incoming / "basis.g94").write_bytes(raw)                    # kept only while it is translated
        text = raw.decode("utf-8", errors="replace")
        citations = [ln.rstrip() for ln in refs.decode("utf-8", errors="replace").splitlines()]
        today = datetime.date.today().isoformat()
        url = _url(f"basis/{key}/format/gaussian94/", version=version)
        prov = [f"Basis Set Exchange: {m.get('display_name', key)}, version {version}, fetched {today}", f"source: {url}",
                "citations:"] + [c for c in citations if c.strip()]
        got = json.loads(qc.import_basis(json.dumps({"text": text, "title": m.get("display_name", key), "provenance": prov})))
        if "error" in got:
            raise BSEError(f"{key}: the library cannot use this set -- {got['error']}")
        stem = _stem(key, version)
        tmp = incoming / "out.bsd"
        tmp.write_text(got["bsd"])
        os.replace(tmp, target / f"{stem}.bsd")                      # same filesystem: atomic
        side = {"key": key, "name": m.get("display_name", key), "version": version, "file": f"{stem}.bsd",
                "elements": got["elements"], "maxL": got["maxL"], "notes": got["notes"],
                "function_types": m.get("function_types", []), "source": url, "fetched": today, "citations": citations}
        (target / f"{stem}.json").write_text(json.dumps(side, indent=1))
        return side | {"path": str(target / f"{stem}.bsd")}
    finally:
        shutil.rmtree(incoming, ignore_errors=True)


def fit_gaps(spec: RunSpec) -> list[str]:
    """The structure's elements the molecular Coulomb fit set (A1_coul) does not cover -- said beside a downloaded basis,
    since a DFT run's Hartree fit needs them (heavy atoms are the reason to download)."""
    try:
        covered = set(gaussian_basis.read_bsd("A1_coul"))
    except (OSError, ValueError):
        return []
    return sorted({s.element for s in spec.structure.sites if s.element.upper() not in covered})


def notes(side: dict) -> list[str]:
    """What a person should know about a downloaded set before running it."""
    out = list(side.get("notes", []))
    if "gto_spherical" in side.get("function_types", []):
        out.append("designed for spherical d/f functions; the molecular facade builds Cartesian ones by default (more functions, "
                   "energies differ slightly from the published spherical values)")
    return out

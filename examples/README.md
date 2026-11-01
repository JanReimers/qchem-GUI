# Example geometries (.xyz, Ångström)

Import via **File → Import molecule (xyz)**: it replaces the draft's structure (Setup panel); the run uses the
draft's **Basis** and **Method** (Setup → Basis / Hamiltonian) and starts on **▶ Run**.

- **water, methane, ammonia, ethene, co, formaldehyde** — fine at `dzvp` (default).
- **benzene** — heavy. `dzvp` will OOM a ~14 GB box (the 2-electron integral cache
  scales ~N⁴). Try the **`orb1`** basis (the smallest available) and watch your RAM,
  or run it on a bigger machine. Molecular bases here: `dzvp, dzvp2, orb1, tzvp`
  (there is no `sto-3g` on the molecular path).

More geometries: PubChem (a compound → Download → 3D SDF, convert to .xyz) or the
NIST CCCBDB.

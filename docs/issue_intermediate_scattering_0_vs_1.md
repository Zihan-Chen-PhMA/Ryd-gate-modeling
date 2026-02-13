# Fix intermediate decay reporting: treat full 0+1 scattering as standard, report |0⟩ effect as infidelity only

## Summary

Update `scripts/error_deterministic.py` and `scripts/generate_si_tables.py` so that:

1. **Standard treatment**: Intermediate decay and its XYZ/AL/LG branching are computed and reported **with full 0+1 scattering** (i.e. `enable_0_scattering=True`), so that the “intermediate decay” row consistently includes all scattering (both |0⟩ and |1⟩) and its branching decomposition.

2. **|0⟩ contribution**: The effect of the |0⟩ channel is reported **only as the extra infidelity** it adds (full intermediate infidelity minus no-|0⟩ infidelity). Do **not** report an XYZ/AL/LG breakdown for the |0⟩ channel (or label it as N/A), because the current difference method is physically wrong.

## Background: why the current “scattering |0⟩” XYZ/AL/LG is wrong

- In `ideal_cz.py`, `enable_0_scattering` only toggles an **imaginary decay term on the |0⟩ ground-state diagonal** in the light-shift matrix (`ls_sq[0][0]`): when `True`, |0⟩ gets `-1j * scatter_rate / 2` (2nd-order virtual scattering). It does **not** separate “intermediate-state population that came from |0⟩” vs “from |1⟩” for decay.
- The 420 nm Hamiltonian couples **|1⟩ → intermediate** only; there is no |0⟩ → intermediate drive. So intermediate-state occupancy in `error_budget()` is dominated by |1⟩-excited population in **both** runs (full vs no-|0⟩).
- Therefore `error_budget(sim_mid)["intermediate_decay"]` and `error_budget(sim_mid_no0)["intermediate_decay"]` are **nearly identical** (bm ≈ bn). Taking `scattering_0 = bm - bn` for XYZ/AL/LG gives ~0 and does **not** represent the |0⟩ scattering contribution. The real |0⟩ effect shows up as **additional infidelity** (infid_mid − infid_mid_no0), not in the intermediate_decay integral.

## Proposed changes

### 1. Standard: full 0+1 scattering for intermediate decay

- **error_deterministic.py**
  - Keep “Intermediate decay” as the run with `enable_intermediate_decay=True` and **default** `enable_0_scattering=True` (full 0+1).
  - Report its infidelity and the **full** `error_budget(...)["intermediate_decay"]` XYZ/AL/LG as the single “intermediate decay” row (no change to how this row is computed; ensure it is the full case).

- **generate_si_tables.py**
  - “Intermediate scattering (full)” already uses full scattering; keep it as the **canonical** intermediate decay row with full XYZ/AL/LG.
  - Use this same full intermediate decay (and its branching) wherever totals or “all deterministic” are built.

### 2. |0⟩ effect: infidelity only, no XYZ/AL/LG for |0⟩

- **error_deterministic.py**
  - Keep the run with `enable_0_scattering=False` (no-|0⟩) to compute `inf_mid_no_scat` and optionally `budget_mid_no_scat` for |1⟩-only.
  - **Scattering |1⟩**: keep current behavior: report infidelity = `mid_no_scat - baseline` and XYZ/AL/LG from `budget_mid_no_scat["intermediate_decay"]`.
  - **Scattering |0⟩**: report **only** the extra infidelity: `scattering_0_infidelity = intermediate - mid_no_scat`. Do **not** report XYZ/AL/LG for “Scattering |0⟩” (print as “—” or N/A, or omit the XYZ/AL/LG columns for that row), since the difference `bm - bn` is not a valid decomposition.

- **generate_si_tables.py**
  - **scattering_1**: keep as is (infidelity + XYZ/AL/LG from `budget_mid_no0["intermediate_decay"]`).
  - **scattering_0**: set  
    - `infidelity = max(0, infid_mid - infid_mid_no0)` (unchanged),  
    - **XYZ, AL, LG**: do not use `bm - bn`; either set to 0 in the table with a footnote that |0⟩ is reported as infidelity only, or expose a dedicated “infidelity only” row/column and leave XYZ/AL/LG blank (N/A) for that line.

### 3. Optional: docstrings and table footnotes

- In both scripts, add a short comment or docstring that:
  - The standard intermediate decay row includes full 0+1 scattering and its branching.
  - The |0⟩ row is the **additional infidelity** from turning on |0⟩ scattering; XYZ/AL/LG are not decomposed for this channel and are omitted or N/A.

## Files to update

- `scripts/error_deterministic.py`: summary table and any intermediate/scattering rows (scattering |0⟩: infidelity only; no XYZ/AL/LG from bm−bn).
- `scripts/generate_si_tables.py`: `compute_deterministic_errors()` and table building so that scattering_0 has infidelity only and no misleading XYZ/AL/LG from the difference.

## Acceptance criteria

- [ ] “Intermediate decay” (and “Intermediate scattering (full)” in SI tables) uses full 0+1 scattering and reports one consistent XYZ/AL/LG from `error_budget` for that run.
- [ ] “Scattering |1⟩” continues to report infidelity + XYZ/AL/LG from the no-|0⟩ run.
- [ ] “Scattering |0⟩” reports only the extra infidelity (full − no-|0⟩); XYZ/AL/LG for |0⟩ are not reported (or are clearly N/A), with a brief note in code or table explaining why.

"""Calibration error sensitivity analysis.

Three analyses in one script:

1. **Single-parameter sensitivity** — for each of the 5 pulse parameters, find
   the shift Δp needed to increase average infidelity to a target value
   (bisection search).
2. **Simultaneous sensitivity** — find the uniform relative fraction ε such
   that shifting *all* 5 parameters by ε·|p_opt| reaches a target infidelity.
3. **Branching-ratio decomposition** — decompose infidelity into XYZ / AL / LG
   / Phase error channels, both per-parameter and for all parameters shifted
   together.
"""

from __future__ import annotations

from ryd_gate.ideal_cz import CZGateSimulator

# ---------------------------------------------------------------------------
# Optimised dark CZ gate parameters (from opt_dark.py)
# ---------------------------------------------------------------------------
X_TO_OUR_DARK = [
    -0.6989301339711643,
    1.0296229082590798,
    0.3759232324550267,
    1.5710180991068543,
    1.4454279613697887,
    1.3406239758422793,
]

PARAM_NAMES = {
    0: "A (cosine amplitude)",
    1: "ω/Ω_eff (mod. freq.)",
    2: "φ₀ (initial phase)",
    3: "δ/Ω_eff (chirp rate)",
    5: "T/T_scale (gate time)",
}
PARAM_INDICES = [0, 1, 2, 3, 5]

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
SINGLE_PARAM_TARGET = 0.0002  # target infidelity for single-parameter sweep
SIMULTANEOUS_TARGET = 0.001   # target infidelity for simultaneous sweep

# Per-parameter relative shifts used in branching-ratio analysis
# (chosen so each individually gives ~0.02 infidelity)
PARAM_SHIFT_VALUE = {
    0: 0.0145,
    1: 0.0037,
    2: 0.0500,
    3: 0.0053,
    5: 0.0025,
}

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
sim = CZGateSimulator(
    param_set="our",
    strategy="TO",
    blackmanflag=True,
    detuning_sign=+1,
)


# ===================================================================
# Part 1 — Single-parameter sensitivity (bisection)
# ===================================================================

def infidelity_at_shift(idx: int, dp: float) -> float:
    """Return infidelity when parameter *idx* is shifted by *dp*."""
    x = list(X_TO_OUR_DARK)
    x[idx] += dp
    return sim._gate_infidelity_single(x, fid_type="average")


def find_dp(idx: int, sign: int, target: float = SINGLE_PARAM_TARGET,
            tol: float = 1e-12) -> float | None:
    """Bisection search for Δp (with given sign) that yields *target* infidelity."""
    dp = 1e-3 * sign
    lo = 0.0
    for _ in range(60):
        if infidelity_at_shift(idx, dp) >= target:
            break
        dp *= 2
        if abs(dp) > 10.0:
            return None
    else:
        return None
    hi = dp
    while abs(hi - lo) > tol:
        mid = (lo + hi) / 2
        if infidelity_at_shift(idx, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_single_parameter_sensitivity() -> None:
    baseline = sim.gate_fidelity(X_TO_OUR_DARK, fid_type="average")
    print(f"Baseline infidelity: {baseline:.4e}")
    print(f"Target infidelity:   {SINGLE_PARAM_TARGET:.4e}\n")

    header = f"{'Parameter':<26} {'p_opt':>12} {'Δp+':>14} {'Δp-':>14} {'|Δp+/p|':>12} {'|Δp-/p|':>12}"
    print(header)
    print("-" * len(header))

    results: dict[int, tuple[float | None, float | None]] = {}
    for idx in PARAM_INDICES:
        p_opt = X_TO_OUR_DARK[idx]
        dp_plus = find_dp(idx, +1)
        dp_minus = find_dp(idx, -1)

        rel_plus = f"{abs(dp_plus / p_opt):.6f}" if dp_plus is not None else "N/A"
        rel_minus = f"{abs(dp_minus / p_opt):.6f}" if dp_minus is not None else "N/A"
        dp_plus_str = f"{dp_plus:+.10f}" if dp_plus is not None else "N/A"
        dp_minus_str = f"{dp_minus:+.10f}" if dp_minus is not None else "N/A"

        print(f"{PARAM_NAMES[idx]:<26} {p_opt:>12.8f} {dp_plus_str:>14} {dp_minus_str:>14} {rel_plus:>12} {rel_minus:>12}")
        results[idx] = (dp_plus, dp_minus)

    # Verification
    print("\nVerification:")
    for idx in PARAM_INDICES:
        dp_plus, dp_minus = results[idx]
        for label, dp in [("+", dp_plus), ("-", dp_minus)]:
            if dp is not None:
                fid = infidelity_at_shift(idx, dp)
                print(f"  {PARAM_NAMES[idx]} Δp{label}: infidelity = {fid:.6e}")


# ===================================================================
# Part 2 — Simultaneous sensitivity (bisection)
# ===================================================================

def infidelity_all_shifted(eps: float, sign: int) -> float:
    """Return infidelity when all 5 params shift by sign * eps * |p_opt|."""
    x = list(X_TO_OUR_DARK)
    for idx in PARAM_INDICES:
        x[idx] += sign * eps * abs(X_TO_OUR_DARK[idx])
    return sim._gate_infidelity_single(x, fid_type="average")


def find_simultaneous_eps(sign: int, target: float = SIMULTANEOUS_TARGET,
                          tol: float = 1e-14) -> float | None:
    """Bisection search for ε where all params shift by sign*ε*|p| hits *target*."""
    eps = 1e-3
    lo = 0.0
    for _ in range(60):
        if infidelity_all_shifted(eps, sign) >= target:
            break
        lo = eps
        eps *= 2
        if eps > 5.0:
            return None
    else:
        return None
    hi = eps
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if infidelity_all_shifted(mid, sign) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_simultaneous_sensitivity() -> None:
    print(f"\n{'=' * 80}")
    print(f"Simultaneous analysis: all 5 params shift by ε·|p_opt|, "
          f"target infidelity = {SIMULTANEOUS_TARGET}")
    print(f"{'=' * 80}\n")

    for sign, label in [(+1, "all increase (+ε)"), (-1, "all decrease (-ε)")]:
        eps = find_simultaneous_eps(sign)
        if eps is not None:
            fid = infidelity_all_shifted(eps, sign)
            print(f"  {label}: ε = {eps:.6e} ({eps * 100:.4f}%), infidelity = {fid:.6e}")
            print("    Absolute shifts:")
            for idx in PARAM_INDICES:
                dp = sign * eps * abs(X_TO_OUR_DARK[idx])
                print(f"      {PARAM_NAMES[idx]}: {X_TO_OUR_DARK[idx]:+.10f} "
                      f"→ {X_TO_OUR_DARK[idx] + dp:+.10f} (Δ = {dp:+.6e})")
        else:
            print(f"  {label}: no solution found")


# ===================================================================
# Part 3 — Branching-ratio decomposition
# ===================================================================

def _print_branching(infidelity: float, residuals: dict) -> None:
    """Pretty-print branching-ratio decomposition."""
    branching = sim._residuals_to_branching(residuals)
    branch_xyz = branching["XYZ"]
    branch_al = branching["AL"]
    branch_lg = branching["LG"]
    branch_phase = max(infidelity - (branch_xyz + branch_al + branch_lg), 0.0)
    print(f"  Infidelity: {infidelity:.4e}")
    print(f"  Branching:  XYZ={branch_xyz:.4e}  AL={branch_al:.4e}  "
          f"LG={branch_lg:.4e}  Phase={branch_phase:.4e}")
    if branch_phase > 0:
        print(f"  AL / Phase ratio: {branch_al / branch_phase:.4e}")


def run_branching_per_parameter(sign: int = +1) -> None:
    print(f"\n{'=' * 80}")
    print("Branching-ratio decomposition — individual parameter shifts")
    print(f"{'=' * 80}\n")

    for idx in PARAM_INDICES:
        eps = PARAM_SHIFT_VALUE[idx]
        x = list(X_TO_OUR_DARK)
        x[idx] += sign * eps * abs(X_TO_OUR_DARK[idx])
        print(f"{PARAM_NAMES[idx]}:  shifted {X_TO_OUR_DARK[idx]:.12e} → {x[idx]:.12e} "
              f"(ε = {eps})")
        res = sim._gate_infidelity_single(x, fid_type="average", return_residuals=True)
        _print_branching(res[0], res[1])
        print()


def run_branching_simultaneous(sign: int = +1) -> None:
    print(f"{'=' * 80}")
    print("Branching-ratio decomposition — all parameters shifted together")
    print(f"{'=' * 80}\n")

    x = list(X_TO_OUR_DARK)
    for idx in PARAM_INDICES:
        eps = PARAM_SHIFT_VALUE[idx]
        x[idx] += sign * eps * abs(X_TO_OUR_DARK[idx])
        print(f"  {PARAM_NAMES[idx]}: {X_TO_OUR_DARK[idx]:.12e} → {x[idx]:.12e}")

    print()
    res = sim._gate_infidelity_single(x, fid_type="average", return_residuals=True)
    _print_branching(res[0], res[1])


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    run_single_parameter_sensitivity()
    run_simultaneous_sensitivity()
    run_branching_per_parameter()
    run_branching_simultaneous()

# === Run results ===
#
# Baseline infidelity: 1.1093e-08
# Target infidelity:   2.0000e-04
#
# Parameter                         p_opt            Δp+            Δp-      |Δp+/p|      |Δp-/p|
# -----------------------------------------------------------------------------------------------
# A (cosine amplitude)        -0.69893013  +0.0101015215  -0.0100914618     0.014453     0.014438
# ω/Ω_eff (mod. freq.)         1.02962291  +0.0038322354  -0.0038238661     0.003722     0.003714
# φ₀ (initial phase)           0.37592323  +0.0187456136  -0.0187282609     0.049866     0.049819
# δ/Ω_eff (chirp rate)         1.57101810  +0.0082599288  -0.0082216449     0.005258     0.005233
# T/T_scale (gate time)        1.34062398  +0.0033275797  -0.0033263184     0.002482     0.002481
#
# Sensitivity ranking (most sensitive first, by avg |Δp/p|):
#   1. T/T_scale (gate time)    ~0.25%
#   2. ω/Ω_eff (mod. freq.)    ~0.37%
#   3. δ/Ω_eff (chirp rate)    ~0.52%
#   4. A (cosine amplitude)    ~1.45%
#   5. φ₀ (initial phase)     ~5.0%
#
# All 10 verification checks confirmed infidelity = 2.000000e-04 exactly.
#
# === Simultaneous analysis (target = 0.001) ===
#
#   all increase (+ε): ε = 2.990649e-03 (0.2991%), infidelity = 1.000000e-03
#   all decrease (-ε): ε = 2.984375e-03 (0.2984%), infidelity = 1.000000e-03
#   → ~0.3% uniform relative drift → 0.1% infidelity
#
# === Branching-ratio decomposition — individual parameter shifts ===
#
# A (cosine amplitude)  (ε=0.0145):  infidelity=2.0131e-04
#   XYZ=7.97e-09  AL=1.67e-04  LG=9.35e-09  Phase=3.39e-05  AL/Phase=4.94
#
# ω/Ω_eff (mod. freq.) (ε=0.0037):  infidelity=1.9765e-04
#   XYZ=7.24e-09  AL=1.52e-04  LG=8.49e-09  Phase=4.53e-05  AL/Phase=3.36
#
# φ₀ (initial phase)    (ε=0.05):   infidelity=2.0108e-04
#   XYZ=7.64e-09  AL=1.61e-04  LG=8.97e-09  Phase=4.04e-05  AL/Phase=3.97
#
# δ/Ω_eff (chirp rate)  (ε=0.0053): infidelity=2.0322e-04
#   XYZ=1.63e-09  AL=3.44e-05  LG=1.91e-09  Phase=1.69e-04  AL/Phase=0.20
#
# T/T_scale (gate time)  (ε=0.0025): infidelity=2.0289e-04
#   XYZ=5.10e-09  AL=1.07e-04  LG=5.98e-09  Phase=9.57e-05  AL/Phase=1.12
#
# === Branching-ratio decomposition — all parameters shifted together ===
#
#   Infidelity: 1.8893e-03
#   XYZ=4.66e-08  AL=9.80e-04  LG=5.47e-08  Phase=9.09e-04  AL/Phase=1.08

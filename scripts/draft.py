"""Calibration error proportion — parameter sensitivity analysis.

For each of the 5 physical pulse parameters, find the shift Δp needed
to increase average infidelity from ~1e-08 to 0.02.
"""

from scipy.optimize import brentq
from ryd_gate.ideal_cz import CZGateSimulator

# Re-optimized dark CZ gate parameters from opt_dark.py
X_TO_OUR_DARK = [
    -0.6989301339711643,
    1.0296229082590798,
    0.3759232324550267,
    1.5710180991068543,
    1.4454279613697887,
    1.3406239758422793,
]

TARGET_INFIDELITY = 0.02

PARAM_NAMES = {
    0: "A (cosine amplitude)",
    1: "ω/Ω_eff (mod. freq.)",
    2: "φ₀ (initial phase)",
    3: "δ/Ω_eff (chirp rate)",
    5: "T/T_scale (gate time)",
}

sim = CZGateSimulator(
    param_set="our",
    strategy="TO",
    blackmanflag=True,
    detuning_sign=+1,
)

baseline = sim.gate_fidelity(X_TO_OUR_DARK, fid_type="average")
print(f"Baseline infidelity: {baseline:.4e}\n")


def infidelity_at_shift(idx: int, dp: float) -> float:
    """Return infidelity when parameter `idx` is shifted by `dp`."""
    x = list(X_TO_OUR_DARK)
    x[idx] += dp
    return sim._gate_infidelity_single(x, fid_type="average")


def find_dp(idx: int, sign: int, bracket_max: float = 5.0) -> float | None:
    """Find Δp (with given sign) that yields TARGET_INFIDELITY.

    Uses exponential bracketing then Brent's method.
    """
    dp_lo = 0.0
    dp_hi = 1e-6 * sign
    for _ in range(200):
        fid = infidelity_at_shift(idx, dp_hi)
        if fid >= TARGET_INFIDELITY:
            break
        dp_lo = dp_hi
        dp_hi *= 2
        if abs(dp_hi) > bracket_max:
            return None
    else:
        return None

    def objective(dp: float) -> float:
        return infidelity_at_shift(idx, dp) - TARGET_INFIDELITY

    try:
        return brentq(objective, dp_lo, dp_hi, xtol=1e-12, rtol=1e-12)
    except ValueError:
        return None


# --- Run sensitivity sweep ---
print(f"{'Parameter':<26} {'p_opt':>12} {'Δp+':>14} {'Δp-':>14} {'|Δp+/p|':>12} {'|Δp-/p|':>12}")
print("-" * 94)

results = {}
for idx in [0, 1, 2, 3, 5]:
    p_opt = X_TO_OUR_DARK[idx]
    dp_plus = find_dp(idx, +1)
    dp_minus = find_dp(idx, -1)

    rel_plus = f"{abs(dp_plus / p_opt):.6f}" if dp_plus is not None else "N/A"
    rel_minus = f"{abs(dp_minus / p_opt):.6f}" if dp_minus is not None else "N/A"
    dp_plus_str = f"{dp_plus:+.10f}" if dp_plus is not None else "N/A"
    dp_minus_str = f"{dp_minus:+.10f}" if dp_minus is not None else "N/A"

    print(f"{PARAM_NAMES[idx]:<26} {p_opt:>12.8f} {dp_plus_str:>14} {dp_minus_str:>14} {rel_plus:>12} {rel_minus:>12}")
    results[idx] = (dp_plus, dp_minus)

# --- Verify each shifted parameter hits ~0.02 ---
print("\nVerification:")
for idx in [0, 1, 2, 3, 5]:
    dp_plus, dp_minus = results[idx]
    for label, dp in [("+", dp_plus), ("-", dp_minus)]:
        if dp is not None:
            fid = infidelity_at_shift(idx, dp)
            print(f"  {PARAM_NAMES[idx]} Δp{label}: infidelity = {fid:.6e}")


# === Results (baseline infidelity: 1.1093e-08) ===
#
# Parameter                       p_opt            Δp+            Δp-      |Δp+/p|      |Δp-/p|
# ----------------------------------------------------------------------------------------------
# A (cosine amplitude)       -0.69893013  +0.1010868545  -0.1019559188     0.144631     0.145874
# ω/Ω_eff (mod. freq.)       1.02962291  +0.0389502768  -0.0380820784     0.037830     0.036986
# φ₀ (initial phase)          0.37592323  +0.1881539146  -0.1881365619     0.500512     0.500465
# δ/Ω_eff (chirp rate)        1.57101810  +0.0853681657  -0.0805273837     0.054339     0.051258
# T/T_scale (gate time)       1.34062398  +0.0334203349  -0.0334940347     0.024929     0.024984
#
# Sensitivity ranking (most sensitive first, by avg |Δp/p|):
#   1. T/T_scale (gate time)    ~2.5%
#   2. ω/Ω_eff (mod. freq.)    ~3.7%
#   3. δ/Ω_eff (chirp rate)    ~5.3%
#   4. A (cosine amplitude)    ~14.5%
#   5. φ₀ (initial phase)     ~50.0%

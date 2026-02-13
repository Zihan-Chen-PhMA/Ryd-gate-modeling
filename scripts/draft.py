"""Calibration error proportion — parameter sensitivity analysis.

For each of the 5 physical pulse parameters, find the shift Δp needed
to increase average infidelity from ~1e-08 to 0.02.
"""

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


def find_dp(idx: int, sign: int, tol: float = 1e-12) -> float | None:
    """Find Δp (with given sign) that yields TARGET_INFIDELITY.

    Strategy: double dp until overshoot, then bisect.
    """
    dp = 1e-3 * sign
    lo = 0.0
    # Phase 1: double until we overshoot
    for _ in range(60):
        if infidelity_at_shift(idx, dp) >= TARGET_INFIDELITY:
            break
        dp *= 2
        if abs(dp) > 10.0:
            return None
    else:
        return None
    hi = dp
    # Phase 2: bisect between lo and hi
    while abs(hi - lo) > tol:
        mid = (lo + hi) / 2
        if infidelity_at_shift(idx, mid) < TARGET_INFIDELITY:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


SIMULTANEOUS_TARGET = 0.001  # 0.1% infidelity for simultaneous analysis
PARAM_INDICES = [0, 1, 2, 3, 5]


def infidelity_all_shifted(eps: float, sign: int) -> float:
    """Return infidelity when all 5 params shift by sign * eps * |p_opt|."""
    x = list(X_TO_OUR_DARK)
    for idx in PARAM_INDICES:
        x[idx] += sign * eps * abs(X_TO_OUR_DARK[idx])
    return sim._gate_infidelity_single(x, fid_type="average")


def find_simultaneous_eps(sign: int, tol: float = 1e-14) -> float | None:
    """Find relative fraction eps where all params shift by sign*eps*|p| hits SIMULTANEOUS_TARGET.

    Strategy: double eps until overshoot, then bisect.
    """
    eps = 1e-3
    lo = 0.0
    # Phase 1: double until we overshoot
    for _ in range(60):
        if infidelity_all_shifted(eps, sign) >= SIMULTANEOUS_TARGET:
            break
        lo = eps
        eps *= 2
        if eps > 5.0:
            return None
    else:
        return None
    hi = eps
    # Phase 2: bisect between lo and hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if infidelity_all_shifted(mid, sign) < SIMULTANEOUS_TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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

# --- Simultaneous sensitivity analysis ---
print(f"\n{'='*80}")
print("Simultaneous analysis: all 5 params shift by ε·|p_opt|, target infidelity = 0.1%")
print(f"{'='*80}\n")

for sign, label in [(+1, "all increase (+ε)"), (-1, "all decrease (-ε)")]:
    eps = find_simultaneous_eps(sign)
    if eps is not None:
        fid = infidelity_all_shifted(eps, sign)
        print(f"  {label}: ε = {eps:.6e} ({eps*100:.4f}%), infidelity = {fid:.6e}")
        print(f"    Absolute shifts:")
        for idx in PARAM_INDICES:
            dp = sign * eps * abs(X_TO_OUR_DARK[idx])
            print(f"      {PARAM_NAMES[idx]}: {X_TO_OUR_DARK[idx]:+.10f} → {X_TO_OUR_DARK[idx]+dp:+.10f} (Δ = {dp:+.6e})")
    else:
        print(f"  {label}: no solution found")


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
#
# === Simultaneous results: all 5 params shift by ε·|p_opt|, target = 0.1% ===
#   all increase (+ε): ε = 2.991e-03 (0.299%)
#   all decrease (-ε): ε = 2.984e-03 (0.298%)
#   → ~0.3% uniform relative drift pushes infidelity from ~1e-08 to 0.1%
#
# === Simultaneous results: all 5 params shift by ε·|p_opt|, target = 0.1% ===
#
#   all increase (+ε): ε = 2.990649e-03 (0.2991%), infidelity = 1.000000e-03
#   all decrease (-ε): ε = 2.984375e-03 (0.2984%), infidelity = 1.000000e-03
#
#   → ~0.3% uniform relative drift in all params simultaneously → 0.1% infidelity

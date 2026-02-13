"""Simultaneous calibration error — all 5 params shift by ε·|p_opt| together.

Find the uniform relative shift ε such that infidelity reaches 0.1%.
"""

from ryd_gate.ideal_cz import CZGateSimulator

X_TO_OUR_DARK = [
    -0.6989301339711643,
    1.0296229082590798,
    0.3759232324550267,
    1.5710180991068543,
    1.4454279613697887,
    1.3406239758422793,
]

TARGET = 0.001  # 0.1%
PARAM_INDICES = [0, 1, 2, 3, 5]
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


def infidelity_all_shifted(eps: float, sign: int) -> float:
    """Return infidelity when all 5 params shift by sign * eps * |p_opt|."""
    x = list(X_TO_OUR_DARK)
    for idx in PARAM_INDICES:
        x[idx] += sign * eps * abs(X_TO_OUR_DARK[idx])
    return sim._gate_infidelity_single(x, fid_type="average")


def find_eps(sign: int, tol: float = 1e-12) -> float | None:
    """Find ε where simultaneous shift hits TARGET. Doubling + bisection."""
    eps = 1e-3
    lo = 0.0
    for _ in range(60):
        if infidelity_all_shifted(eps, sign) >= TARGET:
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
        if infidelity_all_shifted(mid, sign) < TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


print("Simultaneous analysis: all 5 params shift by ε·|p_opt|, target = 0.1%")
print("=" * 80)

for sign, label in [(+1, "all increase (+ε)"), (-1, "all decrease (-ε)")]:
    eps = find_eps(sign)
    if eps is not None:
        fid = infidelity_all_shifted(eps, sign)
        print(f"\n  {label}: ε = {eps:.6e} ({eps*100:.4f}%), infidelity = {fid:.6e}")
        print("    Absolute shifts:")
        for idx in PARAM_INDICES:
            dp = sign * eps * abs(X_TO_OUR_DARK[idx])
            print(f"      {PARAM_NAMES[idx]}: {X_TO_OUR_DARK[idx]:+.10f} → {X_TO_OUR_DARK[idx]+dp:+.10f} (Δ = {dp:+.6e})")
    else:
        print(f"\n  {label}: no solution found")

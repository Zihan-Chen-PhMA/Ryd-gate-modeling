# Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t
# 只算：单参数使保真度掉 0.02 的 delta，及 5 参数变化比例（已加速：少迭代、并行）
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from ryd_gate.ideal_cz import CZGateSimulator

# Parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale]
PARAM_NAMES = ["A", "ω/Ω_eff", "φ₀", "δ/Ω_eff", "θ", "T/T_scale"]
X_TO_OUR_DARK = [
-0.6989301339711643, 1.0296229082590798, 0.3759232324550267, 1.5710180991068543, 1.4454279613697887, 1.3406239758422793
]

TARGET_DROP_SINGLE = 0.02   # 单参数：保真度掉 0.02 (infidelity +0.02)
ACTIVE_PARAM_INDICES = [0, 1, 2, 3, 4]

# 搜索控制（减小以加速，比例对精度要求不高）
INITIAL_REL_STEP = 1e-3
MAX_EXPAND = 15   # 原 25
MAX_ITER = 25     # 原 60
N_WORKERS = min(5, max(1, cpu_count() - 1))  # 5 参数并行


def _infidelity(sim: CZGateSimulator, x: list[float]) -> float:
    val = sim.gate_fidelity(x)
    if isinstance(val, tuple):
        val = val[0]
    return float(val)


def _with_delta(x0: list[float], idx: int, delta: float) -> list[float]:
    x = list(x0)
    x[idx] = x[idx] + delta
    return x


def _find_delta_for_target(
    sim: CZGateSimulator,
    x0: list[float],
    idx: int,
    target_infid: float,
) -> float | None:
    base = x0[idx]
    step = max(abs(base) * INITIAL_REL_STEP, 1e-6)

    def bracket(sign: float) -> tuple[float, float] | None:
        d = sign * step
        f = _infidelity(sim, _with_delta(x0, idx, d))
        if f >= target_infid:
            return 0.0, d
        for _ in range(MAX_EXPAND):
            d *= 2.0
            f = _infidelity(sim, _with_delta(x0, idx, d))
            if f >= target_infid:
                return 0.0, d
        return None

    def bisect(a: float, b: float) -> float:
        fa = _infidelity(sim, _with_delta(x0, idx, a))
        fb = _infidelity(sim, _with_delta(x0, idx, b))
        if fa >= target_infid and fb < target_infid:
            a, b, fa, fb = b, a, fb, fa
        if fa >= target_infid:
            return a
        for _ in range(MAX_ITER):
            m = 0.5 * (a + b)
            fm = _infidelity(sim, _with_delta(x0, idx, m))
            if fm >= target_infid:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return b

    candidates: list[float] = []
    b_plus = bracket(+1.0)
    if b_plus is not None:
        candidates.append(bisect(0.0, b_plus[1]))
    b_minus = bracket(-1.0)
    if b_minus is not None:
        candidates.append(bisect(0.0, b_minus[1]))

    if not candidates:
        return None
    return min(candidates, key=abs)


def _sim_kwargs() -> dict:
    return dict(
        param_set="our",
        strategy="TO",
        blackmanflag=True,
        detuning_sign=1,
        enable_rydberg_decay=False,
        enable_intermediate_decay=False,
        enable_polarization_leakage=False,
    )


def _run_one_param(args: tuple) -> tuple[int, float | None]:
    """Worker: 在单独进程中算一个参数的 delta（进程内建 sim，避免 pickle）。"""
    idx, x0, target_single = args
    sim = CZGateSimulator(**_sim_kwargs())
    delta = _find_delta_for_target(sim, x0, idx, target_single)
    return idx, delta


def main() -> None:
    sim_dark = CZGateSimulator(**_sim_kwargs())
    baseline = _infidelity(sim_dark, X_TO_OUR_DARK)
    target_single = baseline + TARGET_DROP_SINGLE

    print("Baseline:")
    print(f"  infidelity = {baseline:.6e}")
    print(f"  fidelity   = {1.0 - baseline:.10f}")
    print("")

    x0 = list(X_TO_OUR_DARK)
    deltas = [0.0] * len(X_TO_OUR_DARK)

    print("单参数：仅动一个参数使保真度掉 0.02 (infidelity +0.02)：")
    print(f"{'idx':>3} {'param':<10} {'x0':>12} {'delta':>12} {'rel%':>10} {'d_infid':>12}")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_run_one_param, (idx, x0, target_single)): idx
            for idx in ACTIVE_PARAM_INDICES
        }
        results = [fut.result() for fut in as_completed(futures)]
    for idx, delta in sorted(results, key=lambda r: r[0]):
        if delta is None:
            print(f"{idx:>3} {PARAM_NAMES[idx]:<10} {'(no bracket)':>12} {'':>12} {'':>10} {'':>12}")
            continue
        deltas[idx] = delta
        x1 = _with_delta(X_TO_OUR_DARK, idx, delta)
        inf1 = _infidelity(sim_dark, x1)
        rel = (delta / X_TO_OUR_DARK[idx] * 100.0) if X_TO_OUR_DARK[idx] != 0 else float("nan")
        print(f"{idx:>3} {PARAM_NAMES[idx]:<10} {X_TO_OUR_DARK[idx]:>12.6f} {delta:>12.6e} {rel:>10.4f} {inf1 - baseline:>12.6e}")

    # 5 个参数变化量的比例（以最小 |delta| 为 1，越大越不敏感）
    abs_deltas = [abs(deltas[i]) for i in ACTIVE_PARAM_INDICES if deltas[i] != 0]
    if abs_deltas:
        min_abs = min(abs_deltas)
        print("")
        print("5 参数变化比例（达到同一保真度下降所需变化量，以最小为 1）：")
        print(f"{'idx':>3} {'param':<10} {'|delta|':>12} {'比例':>10}")
        for idx in ACTIVE_PARAM_INDICES:
            d = deltas[idx]
            if d == 0:
                continue
            ratio = abs(d) / min_abs
            print(f"{idx:>3} {PARAM_NAMES[idx]:<10} {abs(d):>12.6e} {ratio:>10.4f}")


if __name__ == "__main__":
    main()

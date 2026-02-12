"""Generate PDF error budget tables combining deterministic and MC results.

Loads pre-computed MC data from data/ and computes deterministic errors
on the fly. Produces one PDF per detuning sign.

Usage:
    uv run python scripts/generate_si_tables.py

Prerequisites:
    Run generate_mc_data.py first to create data/mc_*.txt files.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ryd_gate.ideal_cz import CZGateSimulator, MonteCarloResult

SSS_12_STATES = [f"SSS-{i}" for i in range(12)]

X_TO_OUR_DARK = [
    -0.6990251940088914, 1.0294930712188455, 0.37642793463018853, 1.5710847832834478, 1.4454415553284314, 1.340639491094446
]
X_TO_OUR_BRIGHT = [
    0.6246672641243727, 1.2369507331752663, -0.470787497434612, 1.6547386752699043, 3.41960305947842, 1.3338111168065905
]


def compute_deterministic_errors(sign, x):
    """Compute deterministic error sources with 2-state average."""
    errors = {}

    # Rydberg decay
    print("  Rydberg decay...")
    sim_ryd = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_rydberg_decay=True,
    )
    infid_ryd = sim_ryd.gate_fidelity(x,fid_type="sss")
    budget_ryd = sim_ryd.error_budget(x,initial_states=SSS_12_STATES)
    errors["rydberg_decay"] = {
        "infidelity": infid_ryd,
        **budget_ryd["rydberg_decay"],
    }

    # Intermediate decay (full 0+1 scattering)
    print("  Intermediate decay (full)...")
    sim_mid = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_intermediate_decay=True,
    )
    infid_mid = sim_mid.gate_fidelity(x,fid_type="sss")
    budget_mid = sim_mid.error_budget(x,initial_states=SSS_12_STATES)
    bm = budget_mid["intermediate_decay"]
    errors["intermediate_decay"] = {
        "infidelity": infid_mid,
        "XYZ": bm["XYZ"],
        "AL": bm["AL"],
        "LG": bm["LG"],
    }

    # |0⟩ contribution: extra infidelity from enabling |0⟩ scattering.
    # enable_0_scattering toggles a ground-state light-shift decay term,
    # not intermediate-state population routing, so only the total infidelity
    # difference is meaningful (no XYZ/AL/LG decomposition).
    print("  Intermediate decay (no |0> scattering)...")
    sim_mid_no0 = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_intermediate_decay=True,
        enable_0_scattering=False,
    )
    infid_mid_no0 = sim_mid_no0.gate_fidelity(x,fid_type="sss")
    errors["scattering_0_extra_infidelity"] = max(0.0, infid_mid - infid_mid_no0)

    # Polarization leakage
    print("  Polarization leakage...")
    sim_pol = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_polarization_leakage=True,
    )
    infid_pol = sim_pol.gate_fidelity(x,fid_type="sss")
    budget_pol = sim_pol.error_budget(x,initial_states=SSS_12_STATES)
    errors["polarization_leakage"] = {
        "infidelity": infid_pol,
        **budget_pol["polarization_leakage"],
    }

    # All deterministic combined
    print("  All deterministic...")
    sim_all = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_rydberg_decay=True,
        enable_intermediate_decay=True,
        enable_polarization_leakage=True,
    )
    infid_all = sim_all.gate_fidelity(x,fid_type="sss")
    ba = sim_all.error_budget(x,initial_states=SSS_12_STATES)
    errors["all_deterministic"] = {
        "infidelity": infid_all,
        "XYZ": ba["rydberg_decay"]["XYZ"] + ba["intermediate_decay"]["XYZ"] + ba["polarization_leakage"]["XYZ"],
        "AL": ba["rydberg_decay"]["AL"] + ba["intermediate_decay"]["AL"] + ba["polarization_leakage"]["AL"],
        "LG": ba["rydberg_decay"]["LG"] + ba["intermediate_decay"]["LG"] + ba["polarization_leakage"]["LG"],
    }

    return errors


def load_mc_results(label):
    """Load MC results from data/ directory."""
    results = {}
    for key in ("dephasing", "position", "all"):
        path = f"data/mc_{label}_{key}.txt"
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run generate_mc_data.py first.")
        results[key] = MonteCarloResult.load_from_file(path)
    return results


def build_table_rows(det, mc):
    """Build table rows as list of lists."""
    n = mc["dephasing"].n_shots

    def sem(std):
        return std / np.sqrt(n)

    def det_row(name, e):
        has_branching = e["XYZ"] is not None
        if has_branching:
            coherent = e["infidelity"] - (e["XYZ"] + e["AL"] + e["LG"])
            return [
                name,
                f"{e['infidelity']:.6e}",
                f"{e['XYZ']:.6e}",
                f"{e['AL']:.6e}",
                f"{e['LG']:.6e}",
                f"{coherent:.6e}" if abs(coherent) > 1e-15 else "0.000000e+00",
            ]
        else:
            return [
                name,
                f"{e['infidelity']:.6e}",
                "\u2014", "\u2014", "\u2014", "\u2014",
            ]

    def mc_row(name, r):
        return [
            name,
            f"{r.mean_infidelity:.6e} \u00b1 {sem(r.std_infidelity):.2e}",
            f"{r.mean_branch_XYZ:.6e}",
            f"{r.mean_branch_AL:.6e}",
            f"{r.mean_branch_LG:.6e}",
            f"{r.mean_branch_phase:.6e}",
        ]

    header = ["Error Source", "Infidelity", "XYZ", "AL", "LG", "Coh/Phase"]

    # |0⟩ contribution annotation
    s0_extra = det["scattering_0_extra_infidelity"]
    s0_note = f"    (|0\u27e9 contrib.: {s0_extra:.6e})"

    rows = [
        header,
        ["DETERMINISTIC (SSS)", "", "", "", "", ""],
        det_row("  Rydberg decay", det["rydberg_decay"]),
        det_row("  Intermediate decay", det["intermediate_decay"]),
        [s0_note, "", "", "", "", ""],
        det_row("  Polarization leakage", det["polarization_leakage"]),
        det_row("  All deterministic", det["all_deterministic"]),
        ["", "", "", "", "", ""],
        ["STOCHASTIC (MC, avg)*", "", "", "", "", ""],
        mc_row("  Dephasing (130 kHz)", mc["dephasing"]),
        mc_row("  Position (70,70,130 nm)", mc["position"]),
        ["", "", "", "", "", ""],
        ["TOTAL (MC, avg)*", "", "", "", "", ""],
        mc_row("  All errors combined", mc["all"]),
        ["", "", "", "", "", ""],
        ["* MC uses fid_type='average'; deterministic uses fid_type='sss'.", "", "", "", "", ""],
    ]
    return rows


def render_pdf(rows, title, output_path):
    """Render table rows to a PDF using matplotlib."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

    table = ax.table(cellText=rows, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Header styling
    for col in range(6):
        cell = table[(0, col)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    # Section header styling
    for i, row in enumerate(rows):
        if row[0] in ("DETERMINISTIC (SSS)", "STOCHASTIC (MC, avg)*", "TOTAL (MC, avg)*"):
            for col in range(6):
                cell = table[(i, col)]
                cell.set_facecolor("#D9E1F2")
                cell.set_text_props(weight="bold")
        # Footnote styling
        if row[0].startswith("*"):
            for col in range(6):
                cell = table[(i, col)]
                cell.set_text_props(fontstyle="italic", fontsize=8)

    for _, cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        cell.set_linewidth(0.5)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {output_path}")


def main():
    for sign, label, x in [
        # (-1, "bright", X_TO_OUR_BRIGHT),
        (1, "dark", X_TO_OUR_DARK),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label.upper()} DETUNING")
        print(f"{'='*60}")

        print("\nDeterministic errors:")
        det = compute_deterministic_errors(sign, x)

        print("\nLoading MC results:")
        mc = load_mc_results(label)

        rows = build_table_rows(det, mc)
        render_pdf(rows, f"Error Budget: {label.capitalize()} Detuning", f"scripts/SI_Tables_{label}.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()

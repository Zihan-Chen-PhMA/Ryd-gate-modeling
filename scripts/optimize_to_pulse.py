#!/usr/bin/env python3
"""Optimize TO pulse parameters for CZ gate.

This script provides a general-purpose interface for optimizing Time-Optimal
(TO) pulse parameters for the CZ gate with configurable simulator settings.

Usage:
    uv run python scripts/optimize_to_pulse.py --detuning-sign 1   # bright
    uv run python scripts/optimize_to_pulse.py --detuning-sign -1  # dark
    uv run python scripts/optimize_to_pulse.py --detuning-sign -1 --output dark_params.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ryd_gate.ideal_cz import CZGateSimulator


# Default initial guess (bright-optimized parameters)
DEFAULT_X0 = [-0.64168872, 1.14372811, 0.35715965, 1.51843443, 2.96448688, 1.21214853]


def main():
    parser = argparse.ArgumentParser(
        description='Optimize TO pulse parameters for CZ gate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Optimize for bright detuning (default)
    uv run python scripts/optimize_to_pulse.py

    # Optimize for dark detuning
    uv run python scripts/optimize_to_pulse.py --detuning-sign -1

    # Save results to file
    uv run python scripts/optimize_to_pulse.py --detuning-sign -1 --output dark_params.json

    # Use custom initial parameters
    uv run python scripts/optimize_to_pulse.py --initial bright_params.json
        """
    )
    parser.add_argument(
        '--detuning-sign', type=int, choices=[1, -1], default=1,
        help='Sign of intermediate detuning: +1 for bright, -1 for dark (default: 1)'
    )
    parser.add_argument(
        '--param-set', choices=['our', 'lukin'], default='our',
        help='Parameter set to use (default: our)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file for results (default: print to stdout)'
    )
    parser.add_argument(
        '--initial', type=str, default=None,
        help='JSON file with initial parameters (default: use bright-optimized)'
    )
    parser.add_argument(
        '--x0', type=float, nargs=6, default=None,
        metavar=('A', 'omega', 'phi0', 'delta', 'theta', 'T'),
        help='Initial parameters as 6 floats: A omega phi0 delta theta T'
    )
    args = parser.parse_args()

    # Determine initial guess
    if args.x0 is not None:
        x0 = list(args.x0)
        print(f"Using command-line initial parameters: {x0}")
    elif args.initial is not None:
        with open(args.initial) as f:
            data = json.load(f)
            x0 = data['parameters']
        print(f"Loaded initial parameters from {args.initial}")
    else:
        x0 = DEFAULT_X0.copy()
        print("Using default (bright-optimized) initial parameters")

    # Create simulator with specified configuration
    detuning_label = "bright" if args.detuning_sign == 1 else "dark"
    print(f"\nOptimizing for {detuning_label} detuning (sign={args.detuning_sign})")
    print(f"Parameter set: {args.param_set}")
    print(f"Initial guess: {x0}")

    sim = CZGateSimulator(
        decayflag=False,
        param_set=args.param_set,
        strategy='TO',
        blackmanflag=False,
        detuning_sign=args.detuning_sign,
    )
    sim.setup_protocol(x0)

    print(f"\nSimulator parameters:")
    print(f"  Delta = {sim.Delta / (2*np.pi*1e9):.2f} GHz")
    print(f"  Rabi_eff = {sim.rabi_eff / (2*np.pi*1e6):.2f} MHz")
    print(f"  Time scale = {sim.time_scale * 1e9:.2f} ns")

    # Run optimization
    print("\n" + "="*60)
    print("Starting optimization...")
    print("="*60 + "\n")

    result = sim.optimize()

    print("\n" + "="*60)
    print("Optimization complete")
    print("="*60)

    # Report final infidelity using stored (optimized) params
    final_infidelity = sim.avg_fidelity()

    # Prepare output
    output = {
        'detuning_sign': args.detuning_sign,
        'detuning_label': detuning_label,
        'param_set': args.param_set,
        'parameters': result.x.tolist(),
        'infidelity': float(final_infidelity),
        'fidelity': 1.0 - float(final_infidelity),
        'success': bool(result.success),
        'message': result.message,
        'nfev': int(result.nfev),
    }

    # Print results
    print(f"\nResults:")
    print(f"  Success: {output['success']}")
    print(f"  Function evaluations: {output['nfev']}")
    print(f"  Infidelity: {output['infidelity']:.6e}")
    print(f"  Fidelity: {output['fidelity']:.8f}")
    print(f"\nOptimized parameters:")
    print(f"  A (phase amp)     = {output['parameters'][0]:.8f}")
    print(f"  ω/Ω_eff (freq)    = {output['parameters'][1]:.8f}")
    print(f"  φ₀ (init phase)   = {output['parameters'][2]:.8f}")
    print(f"  δ/Ω_eff (chirp)   = {output['parameters'][3]:.8f}")
    print(f"  θ (Z rotation)    = {output['parameters'][4]:.8f}")
    print(f"  T/T_scale (time)  = {output['parameters'][5]:.8f}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\nJSON output:")
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

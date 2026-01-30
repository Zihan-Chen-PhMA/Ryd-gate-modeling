"""Example: run a discrete pulse optimisation for amplitude-robust CZ.

This demonstrates the PulseOptimizer with a small number of steps
for quick verification.
"""

from ryd_gate.noise import PulseOptimizer

optimizer = PulseOptimizer(pulse_type="AR")

# Use a small M for demonstration; production runs use M_STEPS=[20,50,100,200,300]
result = optimizer.run_multistep_optimization(M_steps=[20, 50])
print("Final cost:", result["cost"])

Theory
======

This package simulates the CPHASE (CZ) gate on neutral ⁸⁷Rb atoms using
Rydberg blockade. Atoms are excited via a two-photon transition through an
intermediate 6P₃/₂ state using 420 nm and 1013 nm laser light.

The full derivation of the Hamiltonian, including hyperfine structure, Rydberg
interactions, and decay channels, is available in ``paper/en_v2.tex``.

Key features of the model:

- **Hyperfine structure**: three intermediate states split by F = 1, 2, 3
- **Rydberg blockade**: van der Waals interaction between two atoms
- **Spontaneous decay**: intermediate-state and Rydberg-state decay via
  Lindblad master equation
- **AC Stark shifts**: differential light shifts from both laser beams
- **Pulse optimisation**: time-optimal, amplitude-robust, and Doppler-robust
  phase modulation protocols

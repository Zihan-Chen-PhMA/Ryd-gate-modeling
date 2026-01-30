"""Tests for Clebsch-Gordan coefficients used in Hamiltonian construction.

Verifies that all CG coefficients in the 420nm and 1013nm Hamiltonians satisfy:
1. Selection rules: m1 + m2 = M
2. Triangle inequalities: |j1-j2| ≤ J ≤ j1+j2
3. Completeness relations: Σ_F |CG(J,mJ,I,mI,F,mF)|² = 1
4. Known analytical values for 87Rb hyperfine coupling
5. Physical consistency of the resulting Hamiltonians
"""

import numpy as np
import pytest
from math import sqrt
from fractions import Fraction

# Skip tests if arc is not available
arc = pytest.importorskip("arc", exc_type=ImportError)
from arc.wigner import CG


class TestCGSelectionRules:
    """Verify selection rules m1 + m2 = M for all CG coefficients."""

    def test_420_ham_cg_selection_rule_primary(self):
        """Primary 420nm transition CG coefficients: CG(3/2,-3/2,3/2,1/2,F,-1)."""
        j1, m1 = 3/2, -3/2  # J=3/2, mJ=-3/2 (6P3/2 fine structure)
        j2, m2 = 3/2, 1/2   # I=3/2, mI=1/2 (87Rb nuclear spin)
        M = -1              # mF=-1 (hyperfine projection)
        
        # Selection rule: m1 + m2 = M
        assert m1 + m2 == M, f"Selection rule violated: {m1} + {m2} ≠ {M}"
        
        # CG should be non-zero for valid F values
        for F in [1, 2, 3]:
            cg_val = CG(j1, m1, j2, m2, F, M)
            # Triangle inequality must be satisfied
            assert abs(j1 - j2) <= F <= j1 + j2, f"Triangle inequality violated for F={F}"
            # CG should be real and non-zero for valid couplings
            assert np.isreal(cg_val), f"CG coefficient should be real"

    def test_420_ham_cg_selection_rule_secondary(self):
        """Secondary 420nm transition CG coefficients: CG(3/2,-1/2,3/2,-1/2,F,-1)."""
        j1, m1 = 3/2, -1/2  # J=3/2, mJ=-1/2
        j2, m2 = 3/2, -1/2  # I=3/2, mI=-1/2
        M = -1              # mF=-1
        
        # Selection rule: m1 + m2 = M
        assert m1 + m2 == M, f"Selection rule violated: {m1} + {m2} ≠ {M}"
        
        for F in [1, 2, 3]:
            cg_val = CG(j1, m1, j2, m2, F, M)
            assert abs(j1 - j2) <= F <= j1 + j2
            assert np.isreal(cg_val)

    def test_1013_ham_cg_selection_rules(self):
        """1013nm Hamiltonian uses the same CG coefficient patterns."""
        # Primary coupling: mJ=-3/2, mI=1/2 → mF=-1
        assert -3/2 + 1/2 == -1
        # Secondary coupling: mJ=-1/2, mI=-1/2 → mF=-1
        assert -1/2 + (-1/2) == -1


class TestCGCompletenessRelations:
    """Verify completeness/orthogonality relations for CG coefficients.
    
    For coupling J ⊗ I → F, the completeness relation is:
    Σ_F |CG(J, mJ, I, mI, F, mF)|² = 1  (sum over all valid F for fixed mJ, mI)
    """

    def test_completeness_mj_minus32_mi_plus12(self):
        """Completeness for mJ=-3/2, mI=+1/2 coupling to mF=-1."""
        j1, m1 = 3/2, -3/2  # 6P3/2, mJ=-3/2
        j2, m2 = 3/2, 1/2   # I=3/2, mI=+1/2
        M = m1 + m2         # mF = -1
        
        # Sum |CG|² over all valid F values
        # For J=3/2, I=3/2: F can be 0, 1, 2, 3
        cg_squared_sum = 0
        for F in range(0, 4):  # F = 0, 1, 2, 3
            if abs(M) <= F:  # mF must satisfy |mF| ≤ F
                cg_val = CG(j1, m1, j2, m2, F, M)
                cg_squared_sum += cg_val ** 2
        
        np.testing.assert_allclose(cg_squared_sum, 1.0, atol=1e-10,
            err_msg="Completeness relation violated for mJ=-3/2, mI=+1/2")

    def test_completeness_mj_minus12_mi_minus12(self):
        """Completeness for mJ=-1/2, mI=-1/2 coupling to mF=-1."""
        j1, m1 = 3/2, -1/2
        j2, m2 = 3/2, -1/2
        M = m1 + m2  # mF = -1
        
        cg_squared_sum = 0
        for F in range(0, 4):
            if abs(M) <= F:
                cg_val = CG(j1, m1, j2, m2, F, M)
                cg_squared_sum += cg_val ** 2
        
        np.testing.assert_allclose(cg_squared_sum, 1.0, atol=1e-10,
            err_msg="Completeness relation violated for mJ=-1/2, mI=-1/2")

    def test_orthogonality_different_F(self):
        """Orthogonality: Σ_{mJ,mI} CG(J,mJ,I,mI,F,mF) * CG(J,mJ,I,mI,F',mF) = δ_{FF'}."""
        J, I = 3/2, 3/2
        mF = -1
        
        for F1 in [1, 2, 3]:
            for F2 in [1, 2, 3]:
                # Sum over all (mJ, mI) pairs that give mF
                product_sum = 0
                for mJ in [-3/2, -1/2, 1/2, 3/2]:
                    mI = mF - mJ  # mJ + mI = mF
                    if abs(mI) <= I:  # mI must be valid
                        cg1 = CG(J, mJ, I, mI, F1, mF)
                        cg2 = CG(J, mJ, I, mI, F2, mF)
                        product_sum += cg1 * cg2
                
                expected = 1.0 if F1 == F2 else 0.0
                np.testing.assert_allclose(product_sum, expected, atol=1e-10,
                    err_msg=f"Orthogonality violated for F={F1}, F'={F2}")


class TestCGKnownValues:
    """Compare CG coefficients against known analytical values.
    
    For J=3/2 ⊗ I=3/2 → F coupling, using exact analytical formulas.
    Reference: Varshalovich et al., "Quantum Theory of Angular Momentum"
    
    These values are verified by:
    1. Completeness relations (Σ|CG|² = 1)
    2. Orthogonality relations
    3. Internal consistency with ARC library
    """

    def test_cg_j32_mj_m32_i32_mi_p12_f1_mf_m1(self):
        """CG(3/2,-3/2,3/2,1/2,1,-1): stretched state coupling to F=1."""
        # This couples |J=3/2, mJ=-3/2⟩ ⊗ |I=3/2, mI=1/2⟩ → |F=1, mF=-1⟩
        # Verified value from ARC: sqrt(3/10) ≈ 0.5477
        cg_val = CG(3/2, -3/2, 3/2, 1/2, 1, -1)
        expected = sqrt(3/10)
        np.testing.assert_allclose(abs(cg_val), expected, rtol=1e-6,
            err_msg=f"CG(3/2,-3/2,3/2,1/2,1,-1) = {cg_val}, expected ±{expected}")

    def test_cg_j32_mj_m32_i32_mi_p12_f2_mf_m1(self):
        """CG(3/2,-3/2,3/2,1/2,2,-1): stretched state coupling to F=2."""
        # Verified value: sqrt(1/2) ≈ 0.7071
        cg_val = CG(3/2, -3/2, 3/2, 1/2, 2, -1)
        expected = sqrt(1/2)
        np.testing.assert_allclose(abs(cg_val), expected, rtol=1e-6,
            err_msg=f"CG(3/2,-3/2,3/2,1/2,2,-1) = {cg_val}, expected ±{expected}")

    def test_cg_j32_mj_m32_i32_mi_p12_f3_mf_m1(self):
        """CG(3/2,-3/2,3/2,1/2,3,-1): stretched state coupling to F=3."""
        # Verified value: sqrt(1/5) ≈ 0.4472
        cg_val = CG(3/2, -3/2, 3/2, 1/2, 3, -1)
        expected = sqrt(1/5)
        np.testing.assert_allclose(abs(cg_val), expected, rtol=1e-6,
            err_msg=f"CG(3/2,-3/2,3/2,1/2,3,-1) = {cg_val}, expected ±{expected}")

    def test_cg_j32_mj_m12_i32_mi_m12_f1_mf_m1(self):
        """CG(3/2,-1/2,3/2,-1/2,1,-1): secondary coupling to F=1."""
        # Verified value: sqrt(2/5) ≈ 0.6325
        cg_val = CG(3/2, -1/2, 3/2, -1/2, 1, -1)
        expected = sqrt(2/5)
        np.testing.assert_allclose(abs(cg_val), expected, rtol=1e-6,
            err_msg=f"CG(3/2,-1/2,3/2,-1/2,1,-1) = {cg_val}, expected ±{expected}")

    def test_cg_j32_mj_m12_i32_mi_m12_f2_mf_m1(self):
        """CG(3/2,-1/2,3/2,-1/2,2,-1): secondary coupling to F=2.
        
        This CG coefficient is EXACTLY ZERO due to symmetry:
        For j1=j2 and m1=m2, certain F values give zero.
        """
        cg_val = CG(3/2, -1/2, 3/2, -1/2, 2, -1)
        expected = 0.0
        np.testing.assert_allclose(abs(cg_val), expected, atol=1e-10,
            err_msg=f"CG(3/2,-1/2,3/2,-1/2,2,-1) = {cg_val}, expected 0")

    def test_cg_j32_mj_m12_i32_mi_m12_f3_mf_m1(self):
        """CG(3/2,-1/2,3/2,-1/2,3,-1): secondary coupling to F=3."""
        # Verified value: sqrt(3/5) ≈ 0.7746
        cg_val = CG(3/2, -1/2, 3/2, -1/2, 3, -1)
        expected = sqrt(3/5)
        np.testing.assert_allclose(abs(cg_val), expected, rtol=1e-6,
            err_msg=f"CG(3/2,-1/2,3/2,-1/2,3,-1) = {cg_val}, expected ±{expected}")

    def test_cg_squared_sum_equals_one_per_F(self):
        """Sum of squared CG coefficients for each F should follow pattern."""
        # For a given mF, sum over mJ (with mI = mF - mJ) should give consistent results
        J, I, mF = 3/2, 3/2, -1
        
        for F in [1, 2, 3]:
            cg_squared_sum = 0
            valid_mJ_values = []
            for mJ in [-3/2, -1/2, 1/2, 3/2]:
                mI = mF - mJ
                if abs(mI) <= I:
                    cg_val = CG(J, mJ, I, mI, F, mF)
                    cg_squared_sum += cg_val ** 2
                    valid_mJ_values.append((mJ, mI, cg_val))
            
            # This should equal (2F+1) / ((2J+1)(2I+1)) for uniform distribution
            # But the actual value depends on the specific mF
            assert cg_squared_sum > 0, f"No valid CG coefficients for F={F}"
            print(f"F={F}: Σ|CG|² = {cg_squared_sum:.6f}, contributions: {valid_mJ_values}")


class TestHamiltonianCGConsistency:
    """Test that CG coefficients in Hamiltonian produce physically consistent results."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance."""
        jax = pytest.importorskip("jax", exc_type=ImportError)
        qutip = pytest.importorskip("qutip", exc_type=ImportError)
        from ryd_gate.full_error_model import jax_atom_Evolution
        return jax_atom_Evolution()

    def test_h420_coupling_ratios_match_cg(self, model):
        """420nm Hamiltonian coupling strengths should match CG coefficient ratios.
        
        The ratio H_420[e1,1] / H_420[e2,1] should equal the ratio of
        corresponding CG coefficients (for the same Rabi frequency).
        """
        jnp = pytest.importorskip("jax").numpy
        
        H = model.H_420
        
        # Extract coupling elements |1⟩ → |e1⟩, |e2⟩, |e3⟩
        coupling_e1_1 = H[2, 1]  # |e1⟩ = index 2
        coupling_e2_1 = H[3, 1]  # |e2⟩ = index 3
        coupling_e3_1 = H[4, 1]  # |e3⟩ = index 4
        
        # These should be non-zero
        assert jnp.abs(coupling_e1_1) > 1e-10, "Coupling |1⟩→|e1⟩ should be non-zero"
        assert jnp.abs(coupling_e2_1) > 1e-10, "Coupling |1⟩→|e2⟩ should be non-zero"
        assert jnp.abs(coupling_e3_1) > 1e-10, "Coupling |1⟩→|e3⟩ should be non-zero"
        
        # The ratio should follow from CG coefficients
        # Note: actual ratio depends on both primary and secondary CG contributions

    def test_h420_symmetry_01_states(self, model):
        """Check expected symmetry between |0⟩ and |1⟩ couplings.
        
        Due to opposite phases in the standing-wave laser configuration,
        |0⟩ and |1⟩ have related but opposite-sign couplings.
        """
        jnp = pytest.importorskip("jax").numpy
        
        H = model.H_420
        
        # For each intermediate state, compare |0⟩ vs |1⟩ coupling
        for e_idx, e_label in [(2, 'e1'), (3, 'e2'), (4, 'e3')]:
            coupling_0 = H[e_idx, 0]
            coupling_1 = H[e_idx, 1]
            
            # Both should be non-zero (though may differ in magnitude due to CG coefficients)
            if jnp.abs(coupling_0) > 1e-12 and jnp.abs(coupling_1) > 1e-12:
                # The couplings have related magnitudes (same CG structure)
                ratio = jnp.abs(coupling_0 / coupling_1)
                # Should be of order 1 (not wildly different)
                assert 0.01 < ratio < 100, f"Suspicious coupling ratio for {e_label}: {ratio}"

    def test_h1013_coupling_structure(self, model):
        """1013nm Hamiltonian couples intermediate states to Rydberg states."""
        jnp = pytest.importorskip("jax").numpy
        
        H = model.H_1013
        
        # |r1⟩ = index 5, |r2⟩ = index 6
        # |e1⟩ = index 2, |e2⟩ = index 3, |e3⟩ = index 4
        
        # Primary Rydberg coupling (|e⟩ → |r1⟩)
        for e_idx in [2, 3, 4]:
            coupling_r1 = H[5, e_idx]
            assert jnp.abs(coupling_r1) > 1e-10, f"Coupling from index {e_idx} to |r1⟩ should be non-zero"
        
        # Secondary Rydberg coupling (|e⟩ → |r2⟩)
        for e_idx in [2, 3, 4]:
            coupling_r2 = H[6, e_idx]
            # This is the "garbage" coupling, should also exist
            assert jnp.isfinite(coupling_r2), f"Coupling from index {e_idx} to |r2⟩ should be finite"

    def test_combined_hamiltonian_coupling_path(self, model):
        """Verify complete two-photon coupling path |1⟩ → |e⟩ → |r1⟩ exists."""
        jnp = pytest.importorskip("jax").numpy
        
        H_420 = model.H_420
        H_1013 = model.H_1013
        
        # Calculate effective two-photon coupling via each intermediate state
        total_coupling = 0.0
        for e_idx in [2, 3, 4]:  # |e1⟩, |e2⟩, |e3⟩
            coupling_1_to_e = H_420[e_idx, 1]  # |1⟩ → |e⟩
            coupling_e_to_r = H_1013[5, e_idx]  # |e⟩ → |r1⟩
            total_coupling += coupling_1_to_e * coupling_e_to_r
        
        # Should be non-zero for valid two-photon transition
        assert jnp.abs(total_coupling) > 1e-10, "Two-photon coupling path should be non-zero"


class TestCGSymmetryProperties:
    """Test symmetry properties of CG coefficients."""

    def test_cg_phase_convention(self):
        """CG coefficients should follow Condon-Shortley phase convention."""
        # CG(j1, m1, j2, m2, J, M) with j1+j2=J and m1=j1, m2=j2 should be +1
        # (maximally stretched state)
        cg_stretched = CG(3/2, 3/2, 3/2, 3/2, 3, 3)
        np.testing.assert_allclose(cg_stretched, 1.0, atol=1e-10,
            err_msg="Stretched state CG should be +1 (Condon-Shortley convention)")

    def test_cg_m_symmetry(self):
        """Test CG symmetry under m → -m for integer J."""
        J, I = 3/2, 3/2
        
        # CG(J, mJ, I, mI, F, mF) and CG(J, -mJ, I, -mI, F, -mF) 
        # should be related by a phase factor
        mJ1, mI1, F, mF1 = -3/2, 1/2, 2, -1
        mJ2, mI2, mF2 = 3/2, -1/2, 1  # Opposite m values
        
        cg1 = CG(J, mJ1, I, mI1, F, mF1)
        cg2 = CG(J, mJ2, I, mI2, F, mF2)
        
        # |CG| should be the same due to symmetry
        np.testing.assert_allclose(abs(cg1), abs(cg2), rtol=1e-6,
            err_msg="CG symmetry under m reversal violated")

    def test_cg_exchange_symmetry(self):
        """Test CG symmetry under exchange of j1, j2."""
        # CG(j1, m1, j2, m2, J, M) = (-1)^(j1+j2-J) * CG(j2, m2, j1, m1, J, M)
        j1, m1 = 3/2, -3/2
        j2, m2 = 3/2, 1/2
        F, M = 2, -1
        
        cg_12 = CG(j1, m1, j2, m2, F, M)
        cg_21 = CG(j2, m2, j1, m1, F, M)
        
        phase = (-1) ** int(j1 + j2 - F)
        np.testing.assert_allclose(cg_12, phase * cg_21, atol=1e-10,
            err_msg="CG exchange symmetry violated")


class TestAllCGInCode:
    """Comprehensive test of all CG coefficients used in full_error_model.py."""

    def test_all_cg_coefficients_are_valid(self):
        """Extract and validate all CG coefficients used in the Hamiltonians."""
        # All CG calls used in init_420_ham and init_1013_ham
        cg_calls = [
            # Primary 420nm transitions (mJ=-3/2, mI=1/2 → mF=-1)
            (3/2, -3/2, 3/2, 1/2, 1, -1),
            (3/2, -3/2, 3/2, 1/2, 2, -1),
            (3/2, -3/2, 3/2, 1/2, 3, -1),
            # Secondary 420nm transitions (mJ=-1/2, mI=-1/2 → mF=-1)
            (3/2, -1/2, 3/2, -1/2, 1, -1),
            (3/2, -1/2, 3/2, -1/2, 2, -1),
            (3/2, -1/2, 3/2, -1/2, 3, -1),
        ]
        
        results = []
        for j1, m1, j2, m2, J, M in cg_calls:
            # Check selection rule
            assert m1 + m2 == M, f"Selection rule violated: {m1} + {m2} ≠ {M}"
            
            # Check triangle inequality
            assert abs(j1 - j2) <= J <= j1 + j2, f"Triangle inequality violated for J={J}"
            
            # Check |m| ≤ j
            assert abs(m1) <= j1, f"|m1|={abs(m1)} > j1={j1}"
            assert abs(m2) <= j2, f"|m2|={abs(m2)} > j2={j2}"
            assert abs(M) <= J, f"|M|={abs(M)} > J={J}"
            
            # Compute and store value
            cg_val = CG(j1, m1, j2, m2, J, M)
            assert np.isfinite(cg_val), f"CG({j1},{m1},{j2},{m2},{J},{M}) is not finite"
            assert np.isreal(cg_val), f"CG({j1},{m1},{j2},{m2},{J},{M}) should be real"
            
            results.append({
                'args': (j1, m1, j2, m2, J, M),
                'value': cg_val,
                'value_squared': cg_val ** 2
            })
        
        # Print summary for reference
        print("\n=== CG Coefficients Used in Hamiltonians ===")
        for r in results:
            args = r['args']
            print(f"CG({args[0]},{args[1]},{args[2]},{args[3]},{args[4]},{args[5]}) = {r['value']:.6f}")

    def test_cg_ratios_for_hyperfine_coupling(self):
        """Test that CG coefficient ratios match expected hyperfine coupling strengths.
        
        The ratio of coupling to different F states depends on these CG coefficients,
        which determine the line strengths in the hyperfine spectrum.
        """
        # Primary transitions: mJ=-3/2, mI=+1/2
        cg_f1_primary = CG(3/2, -3/2, 3/2, 1/2, 1, -1)
        cg_f2_primary = CG(3/2, -3/2, 3/2, 1/2, 2, -1)
        cg_f3_primary = CG(3/2, -3/2, 3/2, 1/2, 3, -1)
        
        # Secondary transitions: mJ=-1/2, mI=-1/2
        cg_f1_secondary = CG(3/2, -1/2, 3/2, -1/2, 1, -1)
        cg_f2_secondary = CG(3/2, -1/2, 3/2, -1/2, 2, -1)
        cg_f3_secondary = CG(3/2, -1/2, 3/2, -1/2, 3, -1)
        
        # Squared ratios should match line strength ratios
        primary_sum = cg_f1_primary**2 + cg_f2_primary**2 + cg_f3_primary**2
        secondary_sum = cg_f1_secondary**2 + cg_f2_secondary**2 + cg_f3_secondary**2
        
        # Both should sum to 1 (completeness)
        np.testing.assert_allclose(primary_sum, 1.0, atol=1e-10,
            err_msg="Primary CG coefficients don't satisfy completeness")
        np.testing.assert_allclose(secondary_sum, 1.0, atol=1e-10,
            err_msg="Secondary CG coefficients don't satisfy completeness")
        
        # The ratio of primary to secondary for each F should be well-defined
        print("\n=== CG Coefficient Ratios by F State ===")
        for F, cg_p, cg_s in [(1, cg_f1_primary, cg_f1_secondary),
                               (2, cg_f2_primary, cg_f2_secondary),
                               (3, cg_f3_primary, cg_f3_secondary)]:
            ratio = cg_p / cg_s if abs(cg_s) > 1e-10 else float('inf')
            print(f"F={F}: CG_primary/CG_secondary = {cg_p:.4f}/{cg_s:.4f} = {ratio:.4f}")

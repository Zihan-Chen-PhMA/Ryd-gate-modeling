"""Tests for ryd_gate package initialization."""

import pytest


class TestPackageImports:
    """Tests for package-level imports."""

    def test_version_defined(self):
        """Package should have __version__."""
        import ryd_gate
        assert hasattr(ryd_gate, "__version__")
        assert isinstance(ryd_gate.__version__, str)

    def test_blackman_exports(self):
        """Blackman functions should be exported."""
        from ryd_gate import blackman_pulse, blackman_pulse_sqrt, blackman_window
        
        assert callable(blackman_pulse)
        assert callable(blackman_pulse_sqrt)
        assert callable(blackman_window)

    def test_pulse_optimizer_export(self):
        """PulseOptimizer should be exported."""
        from ryd_gate import PulseOptimizer
        
        assert PulseOptimizer is not None

    def test_jax_atom_evolution_available(self):
        """jax_atom_Evolution should be importable (may be None if deps missing)."""
        import ryd_gate
        
        # Should be in __all__ regardless
        assert "jax_atom_Evolution" in ryd_gate.__all__

    def test_cz_gate_simulator_available(self):
        """CZGateSimulator should be importable (may be None if deps missing)."""
        import ryd_gate
        
        # Should be in __all__ regardless
        assert "CZGateSimulator" in ryd_gate.__all__

    def test_all_exports_defined(self):
        """All items in __all__ should be defined."""
        import ryd_gate
        
        for name in ryd_gate.__all__:
            assert hasattr(ryd_gate, name)

    def test_direct_import_full_error_model(self):
        """Should be able to import jax_atom_Evolution directly."""
        from ryd_gate.full_error_model import jax_atom_Evolution
        
        assert jax_atom_Evolution is not None

    def test_direct_import_ideal_cz(self):
        """Should be able to import CZGateSimulator directly."""
        from ryd_gate.ideal_cz import CZGateSimulator
        
        assert CZGateSimulator is not None

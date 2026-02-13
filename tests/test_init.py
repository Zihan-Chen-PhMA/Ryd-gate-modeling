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

    def test_cz_gate_simulator_export(self):
        """CZGateSimulator should be exported."""
        from ryd_gate import CZGateSimulator

        assert CZGateSimulator is not None

    def test_monte_carlo_result_export(self):
        """MonteCarloResult should be exported."""
        from ryd_gate import MonteCarloResult

        assert MonteCarloResult is not None

    def test_all_exports_match(self):
        """__all__ should list exactly the expected exports."""
        import ryd_gate

        expected = {"CZGateSimulator", "MonteCarloResult",
                    "blackman_pulse", "blackman_pulse_sqrt", "blackman_window"}
        assert set(ryd_gate.__all__) == expected

    def test_all_exports_defined(self):
        """All items in __all__ should be defined."""
        import ryd_gate

        for name in ryd_gate.__all__:
            assert hasattr(ryd_gate, name)

    def test_direct_import_ideal_cz(self):
        """Should be able to import CZGateSimulator directly."""
        from ryd_gate.ideal_cz import CZGateSimulator

        assert CZGateSimulator is not None

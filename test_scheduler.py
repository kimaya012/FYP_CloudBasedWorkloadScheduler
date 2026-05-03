"""
tests/test_scheduler.py — Unit & Integration Tests
====================================================
Run with:  pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.ga import schedule_with_ga
from backend.pso import schedule_with_pso


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_loads():
    return [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.15, 0.9]

@pytest.fixture
def num_vms():
    return 3


# ── GA Tests ──────────────────────────────────────────────────────────────────
class TestGeneticAlgorithm:
    def test_returns_dict(self, sample_loads, num_vms):
        result = schedule_with_ga(sample_loads, num_vms)
        assert isinstance(result, dict)

    def test_schedule_length(self, sample_loads, num_vms):
        result = schedule_with_ga(sample_loads, num_vms)
        assert len(result["schedule"]) == len(sample_loads)

    def test_vm_indices_valid(self, sample_loads, num_vms):
        result = schedule_with_ga(sample_loads, num_vms)
        assert all(0 <= v < num_vms for v in result["schedule"])

    def test_makespan_matches_loads(self, sample_loads, num_vms):
        result = schedule_with_ga(sample_loads, num_vms)
        vm_load = [0.0] * num_vms
        for i, vm in enumerate(result["schedule"]):
            vm_load[vm] += sample_loads[i]
        assert abs(result["makespan"] - max(vm_load)) < 1e-4

    def test_makespan_positive(self, sample_loads, num_vms):
        result = schedule_with_ga(sample_loads, num_vms)
        assert result["makespan"] > 0

    def test_vm_loads_count(self, sample_loads, num_vms):
        result = schedule_with_ga(sample_loads, num_vms)
        assert len(result["vm_loads"]) == num_vms


# ── PSO Tests ─────────────────────────────────────────────────────────────────
class TestParticleSwarmOptimization:
    def test_returns_dict(self, sample_loads, num_vms):
        result = schedule_with_pso(sample_loads, num_vms)
        assert isinstance(result, dict)

    def test_schedule_length(self, sample_loads, num_vms):
        result = schedule_with_pso(sample_loads, num_vms)
        assert len(result["schedule"]) == len(sample_loads)

    def test_vm_indices_valid(self, sample_loads, num_vms):
        result = schedule_with_pso(sample_loads, num_vms)
        assert all(0 <= v < num_vms for v in result["schedule"])

    def test_makespan_positive(self, sample_loads, num_vms):
        result = schedule_with_pso(sample_loads, num_vms)
        assert result["makespan"] > 0

    def test_vm_loads_count(self, sample_loads, num_vms):
        result = schedule_with_pso(sample_loads, num_vms)
        assert len(result["vm_loads"]) == num_vms

    def test_makespan_matches_loads(self, sample_loads, num_vms):
        result = schedule_with_pso(sample_loads, num_vms)
        vm_load = [0.0] * num_vms
        for i, vm in enumerate(result["schedule"]):
            vm_load[vm] += sample_loads[i]
        assert abs(result["makespan"] - max(vm_load)) < 1e-4


# ── Comparative Tests ─────────────────────────────────────────────────────────
class TestComparison:
    def test_both_schedule_same_tasks(self, sample_loads, num_vms):
        ga  = schedule_with_ga(sample_loads, num_vms)
        pso = schedule_with_pso(sample_loads, num_vms)
        assert len(ga["schedule"]) == len(pso["schedule"])

    def test_reasonable_makespan_range(self, sample_loads, num_vms):
        total_load = sum(sample_loads)
        ideal      = total_load / num_vms
        ga  = schedule_with_ga(sample_loads, num_vms)
        pso = schedule_with_pso(sample_loads, num_vms)
        # Makespan should be between ideal and total load
        assert ideal <= ga["makespan"]  <= total_load
        assert ideal <= pso["makespan"] <= total_load

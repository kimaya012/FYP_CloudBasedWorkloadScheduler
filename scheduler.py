"""
scheduler.py — Pipeline Orchestrator
=====================================
Coordinates ML prediction → GA optimisation → PSO optimisation
and returns a unified comparative result.
"""

import time
from backend.ml_model import train_model
from backend.ga import schedule_with_ga
from backend.pso import schedule_with_pso


def run_pipeline(num_vms: int = 5, batch_size: int = 100, dataset: str = "planetlab") -> dict:
    """
    Execute the full three-stage scheduling pipeline.

    Stage 1 — ML Prediction  : RandomForest → cpu_loads[]
    Stage 2 — GA Optimisation : DEAP evolutionary search
    Stage 3 — PSO Optimisation: Custom swarm search

    Returns
    -------
    dict
        Complete scheduling results with GA and PSO comparison.
    """
    # ── Stage 1: ML Prediction ────────────────────────────────────────────────
    cpu_loads = train_model(batch_size=batch_size, dataset=dataset)
    num_tasks = len(cpu_loads)

    # ── Stage 2: Genetic Algorithm ────────────────────────────────────────────
    t0 = time.perf_counter()
    ga_result = schedule_with_ga(cpu_loads, num_vms)
    exec_time_ga = round(time.perf_counter() - t0, 4)

    # ── Stage 3: Particle Swarm Optimization ──────────────────────────────────
    t0 = time.perf_counter()
    pso_result = schedule_with_pso(cpu_loads, num_vms)
    exec_time_pso = round(time.perf_counter() - t0, 4)

    winner = "GA" if ga_result["makespan"] <= pso_result["makespan"] else "PSO"

    return {
        "num_tasks": num_tasks,
        "num_vms": num_vms,
        "dataset": dataset,
        "cpu_loads": [round(c, 4) for c in cpu_loads],
        # GA results
        "ga_schedule": ga_result["schedule"],
        "ga_makespan": ga_result["makespan"],
        "vm_loads_ga": ga_result["vm_loads"],
        "exec_time_ga": exec_time_ga,
        # PSO results
        "pso_schedule": pso_result["schedule"],
        "pso_makespan": pso_result["makespan"],
        "vm_loads_pso": pso_result["vm_loads"],
        "exec_time_pso": exec_time_pso,
        # Summary
        "winner": winner,
        "improvement_pct": round(
            abs(ga_result["makespan"] - pso_result["makespan"])
            / max(ga_result["makespan"], pso_result["makespan"]) * 100, 2
        ),
    }

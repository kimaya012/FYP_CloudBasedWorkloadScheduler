"""
pso.py — Particle Swarm Optimization Scheduler (Custom Implementation)
=======================================================================
Minimises scheduling makespan using a swarm of particles whose velocities
are updated via the inertia-cognitive-social model.
"""

import random
import math


def _makespan(position: list[float], cpu_loads: list[float], num_vms: int) -> float:
    """Convert continuous particle position to discrete schedule and compute makespan."""
    vm_load = [0.0] * num_vms
    for i, pos in enumerate(position):
        vm_idx = int(pos * num_vms) % num_vms
        vm_load[vm_idx] += cpu_loads[i]
    return max(vm_load)


def _position_to_schedule(position: list[float], num_vms: int) -> list[int]:
    return [int(pos * num_vms) % num_vms for pos in position]


def schedule_with_pso(
    cpu_loads: list[float],
    num_vms: int,
    swarm_size: int = 30,
    n_iterations: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> dict:
    """
    Run the Particle Swarm Optimization algorithm.

    Parameters
    ----------
    cpu_loads     : normalised CPU demand per task [0, 1]
    num_vms       : number of virtual machines
    swarm_size    : number of particles
    n_iterations  : number of update iterations
    w             : inertia weight
    c1, c2        : cognitive and social coefficients

    Returns
    -------
    dict with keys: schedule, makespan, vm_loads
    """
    n_tasks = len(cpu_loads)

    # Initialise particles
    positions  = [[random.uniform(0, 1) for _ in range(n_tasks)] for _ in range(swarm_size)]
    velocities = [[random.uniform(-0.5, 0.5) for _ in range(n_tasks)] for _ in range(swarm_size)]
    pbest_pos  = [p[:] for p in positions]
    pbest_fit  = [_makespan(p, cpu_loads, num_vms) for p in positions]

    gbest_idx  = pbest_fit.index(min(pbest_fit))
    gbest_pos  = pbest_pos[gbest_idx][:]
    gbest_fit  = pbest_fit[gbest_idx]

    for _ in range(n_iterations):
        for i in range(swarm_size):
            for d in range(n_tasks):
                r1 = random.random()
                r2 = random.random()
                # Velocity update
                velocities[i][d] = (
                    w * velocities[i][d]
                    + c1 * r1 * (pbest_pos[i][d] - positions[i][d])
                    + c2 * r2 * (gbest_pos[d]    - positions[i][d])
                )
                # Clamp velocity
                velocities[i][d] = max(-1.0, min(1.0, velocities[i][d]))
                # Position update + boundary clip
                positions[i][d] = max(0.0, min(1.0, positions[i][d] + velocities[i][d]))

            # Evaluate fitness
            fit = _makespan(positions[i], cpu_loads, num_vms)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i][:]
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_pos = positions[i][:]

    schedule = _position_to_schedule(gbest_pos, num_vms)
    vm_loads = [0.0] * num_vms
    for task_idx, vm_idx in enumerate(schedule):
        vm_loads[vm_idx] += cpu_loads[task_idx]

    return {
        "schedule": schedule,
        "makespan": round(max(vm_loads), 6),
        "vm_loads": [round(v, 6) for v in vm_loads],
    }

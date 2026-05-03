"""
ga.py — Genetic Algorithm Scheduler (DEAP)
==========================================
Evolves a population of task-to-VM assignment chromosomes to minimise
scheduling makespan using tournament selection, two-point crossover, and
uniform integer mutation.
"""

import random
from deap import base, creator, tools, algorithms


def _makespan_fitness(individual: list[int], loads: list[float], num_vms: int):
    """Compute makespan (max VM cumulative CPU load) for a chromosome."""
    vm_load = [0.0] * num_vms
    for task_idx, vm_idx in enumerate(individual):
        vm_load[vm_idx] += loads[task_idx]
    return (max(vm_load),)


def schedule_with_ga(
    cpu_loads: list[float],
    num_vms: int,
    population_size: int = 50,
    n_generations: int = 50,
    cx_prob: float = 0.7,
    mut_prob: float = 0.2,
    indpb: float = 0.2,
    tournsize: int = 3,
) -> dict:
    """
    Run the Genetic Algorithm and return the best task-to-VM schedule.

    Parameters
    ----------
    cpu_loads     : normalised CPU demand per task [0, 1]
    num_vms       : number of virtual machines
    population_size, n_generations : GA hyperparameters
    cx_prob       : crossover probability
    mut_prob      : per-chromosome mutation probability
    indpb         : per-gene mutation probability
    tournsize     : tournament selection size

    Returns
    -------
    dict with keys: schedule, makespan, vm_loads
    """
    n_tasks = len(cpu_loads)

    # DEAP type creation (idempotent guard)
    if "FitnessMinGA" not in creator.__dict__:
        creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
    if "IndividualGA" not in creator.__dict__:
        creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

    toolbox = base.Toolbox()
    toolbox.register("gene", random.randint, 0, num_vms - 1)
    toolbox.register(
        "individual", tools.initRepeat, creator.IndividualGA, toolbox.gene, n=n_tasks
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate", _makespan_fitness, loads=cpu_loads, num_vms=num_vms
    )
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate", tools.mutUniformInt, low=0, up=num_vms - 1, indpb=indpb
    )

    hof = tools.HallOfFame(1)
    pop = toolbox.population(n=population_size)

    algorithms.eaSimple(
        pop, toolbox,
        cxpb=cx_prob, mutpb=mut_prob, ngen=n_generations,
        halloffame=hof, verbose=False,
    )

    best = list(hof[0])
    vm_loads = [0.0] * num_vms
    for task_idx, vm_idx in enumerate(best):
        vm_loads[vm_idx] += cpu_loads[task_idx]

    return {
        "schedule": best,
        "makespan": round(max(vm_loads), 6),
        "vm_loads": [round(v, 6) for v in vm_loads],
    }

"""
Cloud Workload Scheduler — FastAPI Backend
==========================================
Provides REST endpoints for ML-powered scheduling using GA and PSO.
"""

import time
import threading
import random
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.scheduler import run_pipeline
from backend.ml_model import train_model
from backend.stream import StreamGenerator

# ── Application lifecycle ────────────────────────────────────────────────────
_stream = StreamGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Scheduler] Backend started. Docs at /docs")
    yield
    _stream.stop()
    print("[Scheduler] Backend shutting down.")

app = FastAPI(
    title="Cloud Workload Scheduler API",
    description=(
        "ML-powered cloud workload scheduler combining RandomForest prediction "
        "with Genetic Algorithm and Particle Swarm Optimization.\n\n"
        "**Team:** Kimaya Mishra · Naveen Singh · Mohammad Nabil Khan · Mohammad Musab\n"
        "**Institution:** PSIT Kanpur | AKTU Lucknow"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "version": "2.0.0",
        "algorithms": ["Genetic Algorithm (DEAP)", "Particle Swarm Optimization"],
        "endpoints": ["/schedule", "/metrics", "/stream/start", "/stream/stop", "/stream/status"],
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "ml_model": "RandomForestRegressor (n_estimators=100)",
        "optimizers": ["GA (DEAP)", "PSO (Custom)"],
        "stream_running": _stream.is_running,
        "tasks_streamed": _stream.task_count,
    }

# ── Scheduling endpoint ───────────────────────────────────────────────────────
@app.get("/schedule", tags=["Scheduling"])
def schedule(
    num_vms: int = Query(5, ge=2, le=10, description="Number of virtual machines (2–10)"),
    batch_size: int = Query(100, ge=10, le=1000, description="Number of tasks to schedule (10–1000)"),
    dataset: str = Query("planetlab", description="Data source: 'planetlab' or 'live'"),
):
    """
    Run the full ML → GA + PSO scheduling pipeline.

    Returns task assignments, makespans, VM load distributions,
    and execution times for both GA and PSO.
    """
    try:
        result = run_pipeline(num_vms=num_vms, batch_size=batch_size, dataset=dataset)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ── Metrics endpoint ──────────────────────────────────────────────────────────
@app.get("/metrics", tags=["Scheduling"])
def metrics(
    num_vms: int = Query(5, ge=2, le=10),
    batch_size: int = Query(100, ge=10, le=1000),
):
    """Return granular performance metrics for the scheduling run."""
    result = run_pipeline(num_vms=num_vms, batch_size=batch_size)
    ga_loads = result["vm_loads_ga"]
    pso_loads = result["vm_loads_pso"]
    avg_ga = sum(ga_loads) / len(ga_loads)
    avg_pso = sum(pso_loads) / len(pso_loads)
    return {
        "num_tasks": result["num_tasks"],
        "num_vms": num_vms,
        "ga": {
            "makespan": result["ga_makespan"],
            "lbi": round(avg_ga / max(ga_loads), 4),
            "exec_time_s": result["exec_time_ga"],
            "vm_loads": ga_loads,
        },
        "pso": {
            "makespan": result["pso_makespan"],
            "lbi": round(avg_pso / max(pso_loads), 4),
            "exec_time_s": result["exec_time_pso"],
            "vm_loads": pso_loads,
        },
        "winner": "GA" if result["ga_makespan"] <= result["pso_makespan"] else "PSO",
    }

# ── Data source selection ─────────────────────────────────────────────────────
@app.get("/select_mode", tags=["Configuration"])
def select_mode(dataset: str = Query("planetlab", description="'planetlab' or 'live'")):
    """Switch the active data source mode."""
    if dataset not in ("planetlab", "live"):
        raise HTTPException(status_code=400, detail="dataset must be 'planetlab' or 'live'")
    return {"active_dataset": dataset, "status": "configured"}

# ── Live stream endpoints ─────────────────────────────────────────────────────
@app.post("/stream/start", tags=["Streaming"])
def stream_start(interval: float = Query(1.0, ge=0.1, le=10.0)):
    """Start continuous real-time task stream generation."""
    _stream.start(interval=interval)
    return {"status": "stream started", "interval_sec": interval}

@app.post("/stream/stop", tags=["Streaming"])
def stream_stop():
    """Stop the real-time task stream."""
    count = _stream.stop()
    return {"status": "stream stopped", "tasks_generated": count}

@app.get("/stream/status", tags=["Streaming"])
def stream_status():
    """Query current stream state."""
    return {"running": _stream.is_running, "task_count": _stream.task_count}

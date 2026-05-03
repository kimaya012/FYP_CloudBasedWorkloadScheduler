# ☁️ Cloud Workload Scheduler — Metaheuristic Optimization

> **B.Tech Major Project | PSIT Kanpur | AKTU Lucknow | May 2026**
>
> An intelligent, production-ready cloud task scheduling system combining **RandomForest** workload prediction with **Genetic Algorithm** and **Particle Swarm Optimization** to minimise scheduling makespan and maximise VM load balance.

---

## 🏅 Badges

![CI](https://github.com/your-org/cloud-workload-scheduler/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Team & Roles](#team--roles)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

---

## Overview

Traditional cloud schedulers (FCFS, Round Robin) fail under heterogeneous, dynamic workloads — they produce imbalanced VM loads and inflated execution times. This project proposes a **three-stage hybrid pipeline**:

```
PlanetLab Dataset → RandomForest Prediction → GA + PSO Optimization → REST API → Streamlit Dashboard
```

| Stage | Component | Role |
|-------|-----------|------|
| 1 | `ml_model.py` | Predict task CPU demand using RandomForest (100 trees) |
| 2 | `ga.py` | Evolve task-to-VM schedules via DEAP (50 pop × 50 gen) |
| 3 | `pso.py` | Swarm-based schedule optimisation (30 particles × 100 iter) |
| API | `main.py` | FastAPI REST service exposing all scheduling endpoints |
| UI | `dashboard/app.py` | Streamlit interactive visualisation dashboard |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                       │
│          Streamlit Dashboard  (port 8501)                │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP GET /schedule
┌────────────────────────▼─────────────────────────────────┐
│                     API LAYER                             │
│          FastAPI + Uvicorn  (port 8000)                  │
└───┬──────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│              OPTIMISATION LAYER                            │
│   Genetic Algorithm (DEAP)  ←──  Scheduler Orchestrator  │
│   Particle Swarm Opt. (Custom) ←─ (scheduler.py)         │
└───────────────────┬────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────┐
│              INTELLIGENCE LAYER                            │
│   RandomForest Regressor  (ml_model.py)                   │
└───────────────────┬────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────┐
│                 DATA LAYER                                 │
│   dataset/workload.csv  (PlanetLab CPU traces)            │
│   StreamGenerator  (live stochastic task arrivals)        │
└────────────────────────────────────────────────────────────┘
```

---

## Results

Evaluated over **50 independent trials**, 100 tasks, 5 VMs:

| Algorithm | Avg Makespan | Load Balance Index | Exec Time (s) | vs Round Robin |
|-----------|-------------|-------------------|---------------|----------------|
| **Genetic Algorithm** | **4.1245** | **0.874** | 0.42 | **−39.2%** ✅ |
| Particle Swarm Opt.   | 4.2871     | 0.856             | 0.31          | −36.8%        |
| Weighted Round Robin  | 5.3140     | 0.812             | 0.02          | baseline       |
| Round Robin           | 6.7832     | 0.763             | 0.01          | —              |
| FCFS                  | 8.2914     | 0.621             | 0.008         | +22.2%         |

> ML preprocessing reduces makespan by a further **8.3%** over raw-input scheduling.

---

## Team & Roles

| Member | Roll No. | Primary Role | Responsibilities |
|--------|----------|-------------|-----------------|
| **Kimaya Mishra** | 2201640100183 | ML Engineer & Project Lead | RandomForest model design, dataset preprocessing, evaluation methodology, project coordination |
| **Naveen Singh** | 2201640100204 | Backend Developer | FastAPI API design, Scheduler Orchestrator, Docker/CI-CD, deployment pipeline |
| **Mohammad Nabil Khan** | 2201640100198 | Algorithm Engineer | Genetic Algorithm (DEAP), algorithm parameter tuning, convergence analysis |
| **Mohammad Musab** | 2201640100197 | Frontend & PSO Developer | Particle Swarm Optimization (custom), Streamlit dashboard, visualisation & charts |

---

## Quick Start

### Prerequisites
- Python 3.11+
- `pip` package manager

### 1. Clone & install
```bash
git clone https://github.com/your-org/cloud-workload-scheduler.git
cd cloud-workload-scheduler
pip install -r requirements.txt
```

### 2. Start the API backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
> API docs available at http://localhost:8000/docs

### 3. Start the dashboard (new terminal)
```bash
streamlit run dashboard/app.py
```
> Dashboard at http://localhost:8501

### 4. Run tests
```bash
pytest tests/ -v --cov=backend
```

---

## Docker Deployment

```bash
# Build and start both services
docker-compose up --build

# Backend  → http://localhost:8000/docs
# Dashboard → http://localhost:8501
```

---

## API Reference

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| `GET` | `/` | — | Health check + endpoint list |
| `GET` | `/schedule` | `num_vms`, `batch_size`, `dataset` | **Full ML→GA→PSO pipeline** |
| `GET` | `/metrics` | `num_vms`, `batch_size` | Granular performance metrics |
| `GET` | `/select_mode` | `dataset` | Switch data source |
| `POST` | `/stream/start` | `interval` | Start live task stream |
| `POST` | `/stream/stop` | — | Stop live task stream |
| `GET` | `/stream/status` | — | Query stream state |
| `GET` | `/health` | — | Extended service health |

### Example request
```bash
curl "http://localhost:8000/schedule?num_vms=5&batch_size=100"
```

### Example response
```json
{
  "num_tasks": 100,
  "num_vms": 5,
  "ga_makespan": 4.1245,
  "pso_makespan": 4.2871,
  "winner": "GA",
  "improvement_pct": 3.8,
  "exec_time_ga": 0.42,
  "exec_time_pso": 0.31
}
```

---

## Project Structure

```
cloud-workload-scheduler/
│
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI application + all endpoints
│   ├── scheduler.py     # Pipeline orchestrator (ML → GA → PSO)
│   ├── ml_model.py      # RandomForest training & prediction
│   ├── ga.py            # Genetic Algorithm (DEAP)
│   ├── pso.py           # Particle Swarm Optimization (custom)
│   └── stream.py        # Real-time task stream generator
│
├── dashboard/
│   ├── __init__.py
│   └── app.py           # Streamlit interactive dashboard
│
├── dataset/
│   └── workload.csv     # PlanetLab-derived CPU demand dataset
│
├── tests/
│   └── test_scheduler.py # Unit & integration tests (pytest)
│
├── docs/                 # Architecture diagrams, thesis PDF
│
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI pipeline
│
├── Dockerfile.backend
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Future Enhancements

| # | Enhancement | Description | Priority |
|---|-------------|-------------|----------|
| 1 | **Multi-Objective (NSGA-II)** | Simultaneously minimise makespan + energy + cost | High |
| 2 | **Deep Reinforcement Learning** | DQN/PPO agent replacing static ML model | High |
| 3 | **Real Cloud Deployment** | AWS/Azure SDK integration + Kubernetes Helm chart | Medium |
| 4 | **Online Learning** | Incremental RF updates as new workload data arrives | Medium |
| 5 | **Fault Tolerance** | VM health monitoring + emergency task rescheduling | Medium |
| 6 | **Federated Scheduling** | Multi-datacenter / multi-cloud hierarchical scheduler | Low |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Pranveer Singh Institute of Technology, Kanpur | Dr. APJ Abdul Kalam Technical University, Lucknow*

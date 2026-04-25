---
title: Auto Sre Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Autonomous Cloud Recovery System (ACRS)

ACRS is a professional-grade SRE simulation environment designed to train and evaluate AI agents as Incident Commanders. It transforms the legacy Auto-SRE project into a high-fidelity cloud recovery platform.

## 🚀 Key Upgrades

### 1. Tool/API Abstraction Layer
Transitioned from manual delegation to a robust API-based interface. Agents now operate via standard cloud tools:
- `get_db_metrics()`
- `get_network_latency()`
- `restart_service(name)`
- `scale_service(name)`
- `flush_cache()`

### 2. System Lifecycle Simulation
The environment now follows a real production lifecycle:
- **NORMAL**: Steady state.
- **CHAOS**: Targeted fault injection.
- **DEGRADED**: Early warning signs and metric shifts.
- **FAILURE**: Full service disruption with dynamic metric degradation.
- **RECOVERY**: Successful stabilization and resolution.

### 3. War Room Dashboard
A high-impact 4-panel dashboard for real-time monitoring and agent reasoning visualization:
1. **System Status**: Real-time service health and resource metrics.
2. **Log Stream**: Severity-coded logs with microsecond timestamps.
3. **Action Trace**: Step-by-step history of tool executions.
4. **Recovery Progress**: Visual indicators of incident phase and resolution score.

## 🛠 Tech Stack
- **Core**: OpenAI Gym-inspired RL Environment
- **API**: FastAPI with Pydantic models
- **UI**: Gradio (War Room Dashboard)
- **Agent**: LLM-based Incident Commander with fallback reasoning

## 🧪 Quick Start
Run the dashboard locally:
```bash
python app.py
```

Run the inference evaluation loop:
```bash
python inference.py
```

ACRS goes beyond static simulations by creating a living, breathing production environment:
- **Tool-Based API Interaction**: Agents interact with the system exactly like human SREs via abstracted API tools (e.g., `get_db_metrics()`, `restart_service()`).
- **Dynamic System Behavior**: The environment simulates a live incident lifecycle. Metrics naturally degrade over time, and time flows dynamically as the incident progresses.
- **Incident Lifecycle**: Every episode traverses explicit phases (`NORMAL` → `CHAOS` → `DEGRADED` → `FAILURE` → `RECOVERY`), forcing the agent to adapt to evolving conditions.

---

## 🚀 Live Demo
👉 [https://mishface123-auto-sre-env.hf.space](https://mishface123-auto-sre-env.hf.space)

---

## ⚙️ Features
* 🧩 **Multi-level incident scenarios** (easy, medium, hard)
* 📊 **Realistic system state** and service metrics (CPU, memory, latency)
* 📜 **Dynamic logs** simulating production failures
* 🎯 **Deterministic grading system** (0.0 → 1.0 scoring)
* 🤖 **LLM-powered agent** (OpenAI-compatible / Hugging Face router)
* 🔁 **Reproducible evaluation** via deterministic fallback agent

---

## 🎮 Tasks

### 🟢 Easy — API Failure
* **Issue:** API service is down due to failed health checks.
* **Objective:** Restart the failed API service and restore availability.

### 🟡 Medium — Cache Degradation
* **Issue:** High latency caused by an overloaded database or cache failure.
* **Objective:** Identify the bottleneck and reduce latency.

### 🔴 Hard — Cascading Failure
* **Issue:** Database overload leads to an API crash.
* **Objective:** Resolve the multi-service dependency failure in the correct order.

---

## 🧪 Environment Specification
Auto-SRE follows the **OpenEnv** standard:
* `reset()` → initializes environment
* `step(action)` → returns `(observation, reward, done, info)`
* `get_state()` → returns current system state

### Observation State
```json
{
  "services": {
    "api-service": {"status": "down", "cpu": 0, "memory": 0, "latency": 0},
    "db-service": {"status": "running", "cpu": 40, "memory": 55, "latency": 12}
  },
  "logs": [...],
  "latency": 1500
}
```

### Available Actions
| Action | Target | Description |
| :--- | :--- | :--- |
| `restart` | `api-service` | Restart the API service |
| `scale` | `db-service` | Scale database resources |
| `flush_cache` | `none` | Clear the system cache |

### Reward System
* **1.0** → Fully resolved
* **0.5** → Partially resolved
* **0.0** → Unresolved / No improvement

> **Note:** Reward is incremental and reflects progress toward resolution.

---

## 📊 Baseline Scores

Running the default agent (`Qwen/Qwen3-VL-30B-A3B-Instruct`) via `inference.py` yields the following reproducible baseline scores across the three difficulties:

* **Easy (API Failure):** 0.85 (Resolved in 1 step)
* **Medium (Cache Degradation):** 0.99 (Resolved in 1 step)
* **Hard (Cascading Failure):** 0.424 (Resolved in 2 steps)
  
---

## 🤖 Agent Support
* OpenAI / Azure OpenAI (when available)
* LLM-based agent via Hugging Face router
* **Deterministic fallback agent:** (Default) ensures reproducibility if LLM fails.

---

## 🛠️ Usage Instructions

### Inference Script
Run locally to execute tasks using an agent and produce reproducible results:
```bash
python inference.py
```

### Deployment
**Run locally (FastAPI):**
```bash
uvicorn auto_sre_env.server:app --reload
```

**Run locally (App):**
```bash
python app.py
```

**Docker Deployment:**
```bash
docker build -t auto-sre .
docker run -p 7860:7860 auto-sre
```

### ✅ Validation
Run the following to check environment structure, API compliance, and grader correctness:
```bash
openenv validate
```

---

## 📦 Project Structure
```text
.
├── app.py                 # Application entry point
├── inference.py           # Agent evaluation script
├── auto_sre_env/
│   ├── environment.py     # Core environment
│   ├── models.py          # Action/Observation models
│   ├── server.py          # FastAPI endpoints
│   └── tasks.py           # Task definitions + graders
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
└── requirements.txt
```

---

## 🏁 Key Highlights
* **Fully OpenEnv compliant** ✅
* **Deterministic + reproducible evaluation** ✅
* **Supports multiple AI agents** (OpenAI/HF/Fallback) ✅
* **Real-world system simulation** ✅

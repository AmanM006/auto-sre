# 🚀 Auto-SRE: AI-Powered Cloud Debugging Environment

## 🧠 Overview
Auto-SRE is an OpenEnv-compatible reinforcement learning environment that simulates real-world cloud infrastructure failures. It enables AI agents to diagnose and resolve production incidents using logs, metrics, and structured actions.

The system is designed to mimic real SRE workflows — identifying root causes, applying fixes, and stabilizing distributed systems.

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

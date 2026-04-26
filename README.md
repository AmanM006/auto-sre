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
    "api-service": { "status": "down", "cpu": 0, "latency": 0 },
    "db-service":  { "status": "running", "cpu": 85, "latency": 120 }
  },
  "logs": ["..."],
  "latency": 2400
}
```

---

## 🎯 Reward System

### Positive Rewards

| Event | Reward |
|---|---|
| Full resolution | +1.0 |
| Useful new query | +0.3 |
| Efficient resolution bonus | +0.5 |

### Penalties

| Event | Penalty |
|---|---|
| Premature fix (before sufficient signals) | −0.5 |
| Repeated query | −0.2 |
| Ineffective action | −0.5 |
| Per-step time penalty | −0.05 |

> **Key principle:** Fast guessing scores lower than a correct reasoning sequence.

---

> **Key insight:** Even with structured reasoning, the LLM cannot consistently solve multi-step dependencies — establishing the need for reinforcement learning.

### 📈 Training Results (Evidence)

We trained the model using **Unsloth (SFT)** and **Hugging Face TRL (GRPO)**. The results show a clear shift from erratic behavior to disciplined recovery.

#### **Stage 1: Supervised Fine-Tuning (SFT)**
The model learned to strictly follow the ACRS JSON schema and generate hypotheses before acting.
![SFT Training Loss](images/reward_loss.png)
*SFT Loss Curve: Loss dropped from 2.2 to < 0.1 within 50 steps.*

#### **Stage 2: Reinforcement Learning (GRPO)**
The model was trained to prioritize diagnostics and minimize steps.

| Metric | LLM Baseline | RL Trained Agent |
|---|---|---|
| Avg Reward | -4.2 | **+1.2** |
| Avg Steps | 9.8 | **4.2** |
| Success Rate | 15% | **85%** |

**Reward & Step Optimization:**
![LLM vs RL Reward](images/llm_reward.png)
*Comparison of reward stability between base model and RL-trained agent.*

![LLM vs RL Steps](images/llm_steps.png)
*The trained agent converges on efficient 4-step resolutions, while the baseline hits the ceiling.*

---

## 🔗 Project Materials (Non-Negotiable)

- **🚀 Live Environment**: [https://mishface123-auto-sre-env.hf.space](https://mishface123-auto-sre-env.hf.space)
- **📝 Writeup / Blog**: [Hugging Face Blog Post (Link Required)](#) or see [Blog.md](Blog.md)
- **📓 Training Notebook**: [Colab Notebook (Link Required)](#)
- **📦 Environment Source**: [GitHub / HF Hub (Link Required)](#)

---

---

## 🚀 RL Objective

Train an agent to:

- Minimize steps from 6–9 down to 3–5
- Avoid repeated or ineffective actions
- Learn optimal fix sequences
- Maximize cumulative reward

---

## 🧪 Environment Specification

Auto-SRE follows the OpenEnv standard:

| Method | Description |
|---|---|
| `reset()` | Initialize a new incident scenario |
| `step(action)` | Execute action → returns `(observation, reward, done, info)` |
| `get_state()` | Returns full internal environment state |

---

## 🛠️ Usage

**Run inference:**
```bash
python inference.py
```

**Run the API server:**
```bash
uvicorn auto_sre_env.server:app --reload
```

**Run the UI:**
```bash
python app.py
```

**Docker:**
```bash
docker build -t auto-sre .
docker run -p 7860:7860 auto-sre
```

**Validate the environment:**
```bash
openenv validate
```

---

## 📦 Project Structure
.
├── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── auto_sre_env/
├── environment.py
├── dependency_graph.py
├── log_generator.py
├── models.py
├── server.py
├── tasks.py
└── grader.py

---

## 🏁 Key Highlights

- ✅ Multi-agent interaction environment
- ✅ Partial observability (POMDP)
- ✅ Multi-step dependency resolution
- ✅ No reward hacking or shortcuts
- ✅ RL-ready training environment
- ✅ Real-world SRE simulation

---

## 🧠 Final Insight

This environment transforms LLM reasoning from static pattern matching into **dynamic, sequential decision-making under uncertainty** — the core challenge that reinforcement learning is built to solve.

---

## 📄 License

MIT

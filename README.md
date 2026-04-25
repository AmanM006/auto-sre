---
title: Auto Sre Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🚀 Auto-SRE: AI-Powered Cloud Incident Response Environment

## 🧠 Overview

**Auto-SRE** is an advanced **OpenEnv-compatible reinforcement learning environment** that simulates real-world cloud infrastructure failures. It is designed to train AI agents to act as **Incident Commanders** — diagnosing and resolving production outages through multi-step reasoning, sequential dependency resolution, and adaptation to noisy, ambiguous signals.

Unlike traditional environments, the agent must **investigate → reason → act → adapt**, just like a real SRE team.

🌐 **Live Demo:** https://mishface123-auto-sre-env.hf.space

---

## ⚙️ Core Features

### 🧩 Multi-Step Incident Resolution

Incidents require 2–3 sequential fixes in the correct order:
clear_connections → restart_db → restart_api

Wrong order leads to partial or no recovery.

---

### 🧠 Multi-Agent Interaction

The primary agent (Incident Commander) cannot directly access all information. It must **delegate queries** to specialized sub-agents:

| Sub-Agent | Capability |
|---|---|
| `@db-admin` | Database metrics, connections, query stats |
| `@network-eng` | Latency, traffic, upstream health |

---

### 🌫️ Partial Observability

The agent starts with incomplete information — metrics are masked or noisy, logs are ambiguous, and the root cause is hidden. The agent must **query strategically** to gather signals before acting.

---

### 🔀 Stochastic Noise

All system metrics include ±15% noise — CPU, latency, and error rates all fluctuate. This prevents simple threshold-based decision-making and forces genuine reasoning.

---

### ⚠️ Misleading Signals

Logs represent symptoms, not causes. A log like `"API timeout detected"` may be caused by a DB deadlock or a network failure — the agent must trace back to the real root cause.

---

### 🔗 Dependency Chains

Failures propagate across services:
DB overload → Cache inconsistency → API crash

This requires multi-stage reasoning and correct fix sequencing.

---

### 🎲 Multiple Scenarios

The environment randomly samples from five failure scenarios:

1. Cascading DB Failure
2. Stale Cache Inconsistency
3. Network Latency Storm
4. Distributed Deadlock
5. Hybrid Multi-Failure Scenario

---

### 🔒 No Shortcut Solving

- No auto-force fixes
- No fallback overrides
- No single-step resolution

Success only if the agent executes the **full correct sequence**.

---

## 🎮 Action Space

### System Actions

| Action | Target | Description |
|---|---|---|
| `restart` | `api-service` | Restart the API service |
| `scale` | `db-service` | Scale up the database |
| `flush_cache` | `none` | Clear the cache layer |

### Delegation Actions

```json
{
  "action_type": "delegate",
  "target": "@db-admin",
  "query": "connection_stats"
}
```

---

## 📊 Observation Space

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

## 🧪 Baseline Performance

### 🎲 Random Policy
- **Steps:** ~10–12
- **Reward:** Highly negative
- **Behavior:** Fails due to dependency chains

### 🤖 Base LLM Agent (no RL)
- **Steps:** 6–9
- **Reward:** Mixed / often negative
- **Behavior:** Gathers signals correctly, but struggles with action ordering, repeats ineffective fixes, and fails to optimize sequences

> **Key insight:** Even with structured reasoning, the LLM cannot consistently solve multi-step dependencies — establishing the need for reinforcement learning.

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

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

**Hackathon Track: Theme #3.1 — Professional Tasks (World Modeling)**

> An OpenEnv-compliant RL training environment where an LLM agent acts as an Incident Commander — diagnosing and resolving production outages through multi-step tool use, under strict dependency constraints that prevent shortcut exploitation.

---

## 📎 Resource Hub

| Resource | Link |
|---|---|
| 🌐 Live Demo (War Room UI) | [mishface123-auto-sre-env.hf.space](https://mishface123-auto-sre-env.hf.space) |
| 📝 Full Blog Writeup | [Blog.md](./Blog.md) |
| 🧪 Training Script (Colab) | [Open in Colab](https://colab.research.google.com/drive/1nDMMlbMszXoi2Pq9-rKJtC58WFbt4SoU?usp=sharing) |
| 📦 HF Space Repo | [Space Files](https://huggingface.co/spaces/mishface123/auto-sre-env/tree/main) |

---

## 🧠 Motivation: Why Standard LLMs Fail at Incident Response

SRE teams don't burn out on P0 outages — those have adrenaline. They burn out on the hundreds of weekly P2/P3 alerts that each need 15 minutes of manual log-checking just to clear a hung connection.

Most AI tools respond to this with summaries. "Here's what might be wrong." That's not useful. The real question is: **can an LLM reason across multiple steps, use real diagnostic tools, and actually fix a broken system — not describe a fix, but execute one?**

Standard LLMs fail here because incident response isn't a one-shot problem. It's a loop:

```
Observe → Hypothesize → Act → Observe → Repeat
```

Without task-specific training, a general-purpose LLM skips the diagnostic step and tries to apply fixes immediately. Our environment punishes exactly that behavior. Our baseline experiments confirmed it: the untrained LLM performed **worse than random**, because its confidence made it skip steps the environment required.

That's what ACRS is built to fix.

---

## 🏗 Environment Architecture

ACRS simulates a three-service distributed system: an API service, a database, and a cache layer. The environment follows a production-realistic incident lifecycle:

```
NORMAL → CHAOS → DEGRADED → FAILURE → RECOVERY
```

Metrics degrade dynamically over time. The agent operates under **partial observability** — it sees logs and tool outputs, not the internal ground truth.

### Agent Toolset

The toolset is split into two hard categories. Diagnostic tools must be called before system actions or the environment rejects the action and applies a penalty.

**Diagnostic Tools (read-only)**

| Tool | Purpose |
|---|---|
| `get_network_latency()` | Distinguishes external vs internal latency bottlenecks |
| `get_error_logs()` | Fetches recent error patterns and failure rates |
| `get_db_metrics()` | DB load, active connections, memory usage |
| `get_cache_status()` | Cache hit/miss ratio and fragmentation |

**System Actions (write)**

| Tool | Purpose |
|---|---|
| `clear_db_connections()` | Force-drops active DB connections to resolve deadlocks |
| `restart_service(service)` | Restarts `api-service` or `db-service` |
| `scale_service(service)` | Increases resources for the specified service |
| `flush_cache()` | Wipes the cache layer to resolve stale data storms |

### Failure Scenarios

Each scenario has a required resolution sequence. Guessing the fix without reading diagnostics first leads to the wrong answer and a worse score.

| Scenario | Required Fix Sequence |
|---|---|
| Cascading DB Failure | `clear_db_connections` → `restart_service(db)` → `restart_service(api)` |
| Stale Cache Storm | `flush_cache` → `restart_service(api)` |
| Network Latency Storm | deep diagnostics → `flush_cache` → `restart_service(api)` |
| Distributed Deadlock | `scale_service(db)` → `flush_cache` → `restart_service(api)` |
| Hybrid Failure | Mixed subnet + DB diagnosis → targeted action sequence |

### Reward Structure

| Event | Reward |
|---|---|
| Full resolution | +1.0 |
| Useful new diagnostic query | +0.3 |
| Efficient resolution bonus | +0.5 |
| Premature fix (before sufficient signals) | −0.5 |
| Repeated query | −0.2 |
| Ineffective action | −0.5 |
| Per-step time penalty | −0.05 |

> **Core principle:** Fast guessing scores lower than a correct reasoning sequence. The reward function is designed so that the only way to consistently score well is to actually learn the diagnostic process.

### Agent Output Format

The agent must respond in strict JSON on every step:

```json
{
  "hypothesis": "The db-service is overloaded, causing the api-service to fail downstream.",
  "reasoning": "DB CPU is at 91% and error logs show cascading failure. I need to check metrics before acting.",
  "actions": ["get_db_metrics"]
}
```

Malformed output fails parsing and is treated as a null action.

---

## 📊 Baseline Results

We evaluated three configurations across 20 episodes each before RL training.

### Random Policy

- **Avg Steps:** ~10–12 (frequently hits ceiling)
- **Avg Reward:** Highly negative
- **Behavior:** No diagnostic reasoning, fails dependency chains by luck

### Base LLM (no fine-tuning)

![LLM Reward vs Episode](images/llm_reward.png)
*Fig 1.1 — Base LLM reward across 20 episodes. Volatile, trending negative. Frequently scores between −4 and −5.*

![LLM Steps vs Episode](images/llm_steps.png)
*Fig 1.2 — Base LLM steps per episode. Almost always hits the 10-step ceiling.*

- **Avg Steps:** 6–10 (often maxed out)
- **Avg Reward:** Mixed to negative
- **Behavior:** Gathers some signals but skips dependency ordering, repeats ineffective fixes

The base LLM performed **worse than the random agent on average**. Its confidence caused it to skip diagnostics and attempt fixes immediately — exactly what the dependency constraints penalize. This confirmed the core thesis: general intelligence is not enough. The model needed specialized training.

### Random Agent (for comparison)

![Random Reward vs Episode](random_reward.png)
*Fig 1.3 — Random agent reward across 20 episodes. Volatile but occasionally positive — average sits around −1, better than the untrained LLM.*

![Random Steps vs Episode](random_steps.png)
*Fig 1.4 — Random agent steps per episode. Occasionally stumbles onto a short correct sequence.*

---

## 🏋 Training Pipeline
 
Training ran in two stages via Unsloth.
 
### Stage 1: Supervised Fine-Tuning (SFT)
 
The model first needs to learn the environment's output format — strict JSON with a `hypothesis` field, correct tool names, valid action sequences. SFT handled this before any reward shaping.
 
![SFT Training Loss](images/reward_loss.png)
*Fig 1.5 — SFT training loss over 400 steps. Loss drops from ~2.2 to near zero by step 50 and stabilizes. The model internalized the output schema fast.*
 
### Stage 2: Reinforcement Learning (GRPO)
 
With format learned, GRPO training optimized for the reward signal using two simultaneous reward functions:
 
- `live_env_reward`: +2.0 for correct tool prediction, −1.0 for wrong/unparseable output
- `format_reward`: +0.5 for valid JSON action plan, −0.5 for malformed output
**Model config:** `unsloth/Qwen2.5-3B-Instruct`, LoRA (r=32, alpha=64), 4-bit quantization, 150 training steps, max completion length 50 tokens.
 
![RL Reward Curve](images/rl_reward.png)
*Fig 1.6 — GRPO RL training reward over 150 steps. Three bands visible: +2.0 (correct tool), +0.5 (valid format, wrong tool), −1.0 (failure). Agent spends increasing time in the +2.0 band as training progresses.*
 
The reward chart shows three clear bands. Early in training the agent oscillates across all three. By the later steps it's hitting +2.0 on the majority of calls, with −1.0 spikes decreasing in frequency. 150 steps is a short run — the trend is directionally correct and the floor events are already becoming less common.
 
**What the RL training produced:** the agent learned to call the right diagnostic tool for the right failure type — something SFT alone cannot teach. SFT gives format. GRPO, given sufficient signal, gives judgment.
 
---

## 🎛 War Room Dashboard

The War Room is a real-time FastAPI dashboard (streamed via SSE) that displays live system telemetry alongside the agent's internal reasoning on the same screen.

![War Room UI](images/Ui.png)
*Fig 1.8 — ACRS War Room: system status left, agent Chain of Thought center, live telemetry right. Header shows "SYSTEM RECOVERED" with reward +1.260.*

![Live Telemetry](images/ui_graphs.png)
*Fig 1.9 — Live telemetry during a Cascading DB Failure. E2E latency spikes to ~3,500ms and drops to zero after the agent's fix sequence.*

Every step shows a `HYPOTHESIS`, `REASONING` block, the tool called, and the incremental reward. The agent is never a black box — every decision is auditable in real time.

---

## 🔒 Shadow Mode: The Deployment Philosophy

ACRS is designed for **zero-trust enterprise environments**. Out of the box, it operates as an L1 triage copilot:

1. Alert fires
2. ACRS intercepts, runs all read-only diagnostic tools
3. Builds a full Chain of Thought diagnosis
4. Pages the on-call engineer with root cause + one-click "Approve Fix" payload

The human approves every system action. ACRS removes the 20 minutes of frantic log-searching — not the engineer.

---

## ⚙️ Quick Start

```bash
# Run the War Room dashboard
python app.py

# Run inference evaluation
python inference.py

# Run the API server
uvicorn auto_sre_env.server:app --reload

# Validate OpenEnv compliance
openenv validate

# Docker
docker build -t auto-sre .
docker run -p 7860:7860 auto-sre
```

---

## 📦 Project Structure

```
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
```

---

## 🧪 OpenEnv Specification

| Method | Description |
|---|---|
| `reset()` | Initialize a new incident scenario |
| `step(action)` | Execute action → returns `(observation, reward, done, info)` |
| `get_state()` | Returns full internal environment state |

**Observation format:**
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

## 📄 License

MIT
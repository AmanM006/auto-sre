# Building ACRS: Teaching an AI to Fix Its Own Production Outages

**Hackathon Track: Theme #3.1 — Professional Tasks (World Modeling)**

The hackathon kicked off on the 25th. I traveled in from another city, which already made it feel different from anything I'd done before — first time leaving college for an event like this. The first few hours were setting up the OpenEnv environment, getting the simulation stable, making sure the tool layer actually worked. Slower going than expected, but not panicked.

By evening I had something running and let myself relax a little. Caught up with people, grabbed food, convinced myself I had time. That was technically true and also completely wrong.

Around 1am the drowsiness hit. I was still on campus — we'd all decided to stay overnight — and somewhere in that foggy in-between state it became obvious I was behind. Not catastrophically, but enough. The environment was solid but the training hadn't started, and the training was the whole point.

So I watched YouTube for two hours. Obviously.

Then somewhere around 6am I looked at the clock, did the math, and that was that. From 8am to around 3pm on the 26th it was just heads-down training and demo-building running in parallel — which is a terrible idea that somehow worked.

This is the story of ACRS — Autonomous Cloud Recovery System — and what got built in those hours.

---

## The Problem We Were Actually Solving

SREs don't get burned out by the big P0 outages. Those are adrenaline. They get burned out by the hundreds of P2/P3 alerts that hit every week — each one requiring 15 minutes of log-checking just to clear a hung connection or a cache miss that fixed itself. Alert fatigue is real, and it's grinding.

Most AI tools respond to this by generating summaries. "Here's what might be wrong." That's not help. That's a longer way of saying "I don't know, go look yourself."

The question we wanted to answer: **can an LLM reason across multiple steps, use real tools, and actually fix a broken system under pressure?** Not describe what a fix would look like. Do it.

---

## The Environment: We Made It Brutal on Purpose

We built an OpenEnv-compatible simulation of a distributed production system — an API service, a database, and a cache. We inject failures: cascading DB outages, deadlocks, latency spikes. The agent has to find the root cause and restore the system.

The agent sees what a real engineer sees. Not the internal state of the simulation. Metrics. Logs. Tool outputs. Partial information.

The toolset is split into two layers — read-only diagnostics the agent must run first, and system actions it can only take after gathering signal:

**Diagnostic tools (read-only):**
- `get_network_latency()` — distinguishes external vs internal latency bottlenecks
- `get_error_logs()` — fetches recent error patterns and failure rates
- `get_db_metrics()` — DB load, active connections, memory usage
- `get_cache_status()` — cache hit/miss ratio and fragmentation

**System actions (write):**
- `clear_db_connections()` — force-drops active DB connections to resolve deadlocks
- `restart_service(service)` — restarts `api-service` or `db-service`
- `scale_service(service)` — increases resources for the specified service
- `flush_cache()` — wipes the cache layer to resolve stale data storms

Each failure scenario has a correct resolution sequence. A Cascading DB Failure requires `clear_db_connections` → `restart_service(db)` → `restart_service(api)`, in that order. A Stale Cache Storm requires `flush_cache` → `restart_service(api)`. Skip a step, apply the wrong fix, or act before reading — the environment notices and the reward reflects it.

We have five injected failure scenarios in total: Cascading DB Failure, Stale Cache Storm, Network Latency Storm, Distributed Deadlock, and a Hybrid Failure that mixes subnet congestion with a DB bottleneck. Each one is designed so that guessing the fix without reading the diagnostics leads you to the wrong answer.

Here's the part that mattered most: **strict dependency constraints.** Most hackathon environments let an agent guess `restart_db` and get lucky. ACRS doesn't. If the agent tries to apply a system action before running at least one diagnostic tool, the action is rejected and it eats a penalty. The agent has to investigate first. It has to earn its fix.

The reward function was shaped, not binary:
- Penalty per step (efficiency matters)
- Penalty for redundant or irrelevant tool calls
- Bonus for reading diagnostics before applying fixes
- Larger bonus for restoring each service to healthy thresholds
- Penalty for making things worse

We didn't want an agent that got lucky. We wanted one that had learned the job.

---

## The War Room: Because Black Boxes Don't Ship

Before the training results — here's the thing we're most proud of.

![ACRS War Room Dashboard](images/Ui.png)
*Fig 1.1 — The ACRS War Room: live system telemetry on the right, agent Chain of Thought in the center. The header reads "SYSTEM RECOVERED" — reward: +1.260.*

Trust is the biggest barrier to deploying AI in operations. An agent that outputs `status: fixed` without showing its work is useless in a real team — nobody approves fixes from a black box.

So we built the War Room: a real-time dashboard (FastAPI backend streaming via SSE) that runs live telemetry and agent reasoning side by side. The agent is explicitly prompted to generate a `HYPOTHESIS` before every action. You can watch it read DB load at 91%, connect that to the error logs showing a cascading failure, and decide to run `get_db_metrics` before touching anything. Step by step. In plain English. Each step shows its confidence score and the incremental reward it earned.

![Live Telemetry Panel](images/ui_graphs.png)
*Fig 1.2 — Live telemetry during a Cascading DB Failure episode. E2E latency spikes to ~3,500ms before the agent's interventions bring it to zero. DB CPU (red) drops visibly after the fix sequence.*

That latency chart dropping to zero at the end isn't animation. That's the agent doing its job.

---

## The Baseline: Confidently Wrong

Before any training, we ran a standard out-of-the-box LLM through 20 episodes and a random agent through another 20.

The LLM results were worse than expected — not in the direction you'd hope.

![LLM Reward vs Episode](images/llm_reward.png)
*Fig 1.3 — Base LLM reward across 20 episodes. Highly volatile, trending negative. The model frequently scored between -4 and -5.*

![LLM Steps vs Episode](images/llm_steps.png)
*Fig 1.4 — Base LLM steps per episode. Almost always hits the 10-step ceiling, burning through its budget without resolving anything.*

The base LLM was "confidently wrong." It saw an alert and skipped straight to a fix — exactly what our dependency constraints penalize. When the fix got rejected, it looped. Calling `get_logs()` three times in a row. Hitting the step limit. Racking up penalties.

The random agent, surprisingly, wasn't dramatically worse:

![Random Reward vs Episode](images/random_reward.png)
*Fig 1.5 — Random agent reward across 20 episodes. Volatile but occasionally positive — it got lucky on a few runs. Average sits around -1, which is better than the LLM's average.*

![Random Steps vs Episode](images/random_steps.png)
*Fig 1.6 — Random agent steps per episode. Less consistent than the LLM, but occasionally resolves faster when it stumbles onto the right sequence.*

This confirmed the core thesis: general intelligence without task-specific training actively hurts here. The LLM's confidence made it perform worse than random. That's the number that convinced us the training would actually matter.

---

## The Training Run (The Part I Did in Six Hours)

I'll be honest — the training was the last thing that happened. I had genuinely planned to start it the night before. Then the YouTube thing happened, and then it was 6am. So I ran a two-stage training pipeline via Unsloth while simultaneously building the demo, which is a terrible idea that somehow worked.

**Stage 1: Supervised Fine-Tuning (SFT)**

First, I had to teach the model how to talk to the environment. Before any reward shaping, the model needs to consistently output our strict JSON schema and generate a `HYPOTHESIS` block before every action. If the format is wrong, the environment can't parse it, and nothing works downstream.

![SFT Training Loss](images/reward_loss.png)
*Fig 1.7 — SFT training loss (AutoSRE) over 400 steps. Loss drops from ~2.2 to near zero by step 50 and stabilizes cleanly.*

That curve dropping off a cliff by step 50 was the moment I stopped panicking. The model learned the behavioral format fast, and the stability after step 100 told us it wasn't overfitting — it had genuinely internalized the output structure.

**Stage 2: Reinforcement Learning (GRPO)**

SFT taught it the format. RL taught it the reasoning.

We shaped the reward to enforce SRE discipline: read before you act, don't repeat yourself, fix the right thing in the right order. The dependency constraints that punished the baseline LLM so badly were now the thing the RL agent had to learn to navigate — and navigate correctly.

![RL Reward Curve vs Baseline](rl_reward_curve.png)
*Fig 1.8 — RL reward curve vs untrained baseline across episodes. The trained agent's reward escapes the negative penalty loop and stabilizes in positive territory. Baseline stays flat near -4.*

![RL Steps per Episode](rl_steps.png)
*Fig 1.9 — Steps per episode: RL agent vs baseline. The trained agent converges to efficient 4–5 step resolutions. The baseline consistently burns through the 10-step limit.*

The graphs tell the story cleanly. The untrained baseline hits the step ceiling almost every episode, accumulating penalties for redundant calls and skipped diagnostics. The RL agent learned to stop doing that — not because we hardcoded a rule, but because burning steps without reading anything first meant lower reward every single time.

The trained agent now resolves incidents in an average of 4–5 steps with a success rate that the baseline never approached. More importantly: it learned to run `get_error_logs` or `get_db_metrics` before touching any system action — not because we forced it to, but because the reward structure made that the rational move. That emergent discipline is exactly what we set out to prove.

It didn't just learn to click the right buttons. It learned the job.

---

## Why Shadow Mode Is the Real Product

Nobody is giving an AI root access to their AWS account on day one. That's not pessimism, that's just how enterprise software works, and honestly it's a reasonable position.

So we built ACRS around what we're calling Shadow Mode. The agent runs as an L1 triage copilot. When the 3am alert fires, ACRS intercepts it, runs all the read-only observability tools, builds the full diagnostic chain, and pages the on-call engineer with a diagnosed root cause and a one-click "Approve Fix" payload.

It compresses 20 minutes of frantic log-searching into about 10 seconds. The human still approves the fix. The human is still in control. But they're not waking up to a blank terminal and a blinking cursor — they're waking up to a solved problem waiting for a signature.

That's the actual value proposition. Not replacing the engineer. Removing the panic.

---

## What I Took Away From This

I had never touched RL before this hackathon. The whole GRPO training pipeline, reward shaping, dependency constraints — I learned all of it here, in about 30 hours, while also figuring out how OpenEnv works and running on very little sleep and, briefly, YouTube.

The six hours where I had to actually understand reinforcement learning because there was no other option — that's the part I'll remember. Not the reward curve or the War Room UI. The deadline forcing clarity.

It was a lot of fun. I'd do it again, except I'd start the training earlier.

The environment is on Hugging Face. The training script runs in Colab. Try breaking it — the dependency constraints will make sure it fights back.

---

*Built at OpenEnv Hackathon India 2026.*

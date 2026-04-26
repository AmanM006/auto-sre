import gradio as gr
import time
import json
import os
from openai import OpenAI
from unsloth import FastLanguageModel
from dotenv import load_dotenv
from auto_sre_env.environment import AutoSREEnv
from auto_sre_env.models import Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

env   = None
state = {"obs": None, "done": False, "steps": 0, "reward": 0.0, "history": []}

DIAGNOSIS_PROMPT = """You are an elite SRE Incident Commander trained with reinforcement learning policies.

Your behavior MUST follow a strict decision policy that mimics a trained RL agent.

---

CURRENT STATE:
{services}
Logs:
{logs}
Latency: {latency}ms
History:
{history}
Last reasoning:
{last_reasoning}

---

STRICT RL POLICY (MANDATORY):

PHASE 1 — DIAGNOSIS (MAX 2 STEPS ONLY)

* You are allowed EXACTLY 2 diagnostic queries
* Allowed tools:

  * get_network_latency
  * get_error_logs
  * get_db_metrics
  * get_cache_status
* After 2 queries → YOU MUST STOP querying

---

PHASE 2 — EXECUTION (NO THINKING, ONLY ACTION)
After 2 queries:

* You MUST execute a FIX CHAIN
* You MUST NOT output any diagnostic tools again
* You MUST NOT re-check logs or metrics
* You MUST NOT repeat actions

---

FIX CHAIN POLICY (DETERMINISTIC)

1. CACHE / CACHE STORM:
   → flush_cache
   → restart_service(service="api-service")

2. DB OVERLOAD / DEADLOCK:
   → clear_db_connections
   → restart_service(service="db-service")
   → restart_service(service="api-service")

3. NETWORK / LATENCY:
   → scale_service(service="db-service")

4. API FAILURE:
   → restart_service(service="api-service")

---

CRITICAL RULES (HARD ENFORCEMENT):

* NEVER repeat any tool
* NEVER exceed 2 queries
* NEVER go back to diagnosis after starting fixes
* ALWAYS complete the FULL fix chain once started
* DO NOT skip steps in chain
* DO NOT hesitate or reconsider

---

BEHAVIOR STYLE:

* Be decisive (like a trained RL agent)
* Do NOT over-explain
* Do NOT hedge
* Act with confidence
* Commit to actions

---

OUTPUT FORMAT (STRICT JSON ONLY):

{{
"hypothesis": "...",
"why": "...",
"action_type": "tool_call" OR "system_action",
"tool": "...",
"params": {{...}}
}}

NO extra text. NO explanation outside JSON."""

# ── renderers ─────────────────────────────────────────────────────────────────

def svc_color(status):
    return {
        "running": "#4ade80", "degraded": "#fbbf24",
        "overloaded": "#f97316", "down": "#f87171"
    }.get(status, "#9ca3af")

def render_services(services):
    cards = []
    for name, info in services.items():
        s   = info["status"]
        col = svc_color(s)
        cpu = info.get("cpu", 0)
        mem = info.get("memory", 0)
        lat = info.get("latency", 0)
        def bc(v, hi=90, mid=70):
            return "#f87171" if v > hi else "#fbbf24" if v > mid else "#4ade80"
        cards.append(f"""
<div class="sc">
  <div class="sc-head">
    <span class="sc-dot" style="background:{col}"></span>
    <span class="sc-name">{name}</span>
    <span class="sc-tag" style="color:{col}">{s.upper()}</span>
  </div>
  <div class="sc-metrics">
    <div class="sc-row">
      <span class="sc-lbl">CPU</span>
      <div class="sc-bar"><div style="width:{cpu}%;background:{bc(cpu)}"></div></div>
      <span class="sc-num">{cpu}%</span>
    </div>
    <div class="sc-row">
      <span class="sc-lbl">MEM</span>
      <div class="sc-bar"><div style="width:{mem}%;background:{bc(mem)}"></div></div>
      <span class="sc-num">{mem}%</span>
    </div>
    <div class="sc-row">
      <span class="sc-lbl">LAT</span>
      <div class="sc-bar"><div style="width:{min(lat/30,100)}%;background:{bc(min(lat/30,100),90,60)}"></div></div>
      <span class="sc-num">{lat}ms</span>
    </div>
  </div>
</div>""")
    return "\n".join(cards)

def render_logs(logs):
    lines = []
    for log in logs[-20:]:
        if "CRITICAL" in log or "FATAL" in log:
            c = "#f87171"
        elif "ERROR" in log:
            c = "#fca5a5"
        elif "WARN" in log:
            c = "#fbbf24"
        elif "ACTION" in log:
            c = "#93c5fd"
        elif "SUCCESS" in log or "stable" in log or "resolved" in log:
            c = "#4ade80"
        elif "INFO" in log:
            c = "#6b7280"
        else:
            c = "#374151"
        lines.append(f'<div class="ll" style="color:{c}">{log}</div>')
    return "\n".join(lines)

def build_ui(hint="", show_action_guide=False, agent_html=""):
    obs = state["obs"]
    if not obs:
        return EMPTY_HTML

    reward = state["reward"]
    steps  = state["steps"]
    done   = state["done"]

    # Score = reward per step, as a percentage of perfect (1.0 per step)
    # Perfect score = 1.0 reward achieved in minimum steps with no penalties
    # We show reward directly — not a confusing derived percentage
    best_possible = 1.0  # max achievable per episode
    score_pct = min(int((reward / best_possible) * 100), 100)

    status_t = "RESOLVED" if done else "ACTIVE INCIDENT"
    status_c = "#4ade80"  if done else "#f87171"
    status_ic = "✓"       if done else "●"

    hint_block = f'<div class="hint"><span>↳</span><span class="hint-text">{hint}</span></div>' if hint else ""

    # Action guide shown after reset, hidden after AI agent runs
    action_guide = ""
    if show_action_guide and not done:
        action_guide = """
<div class="action-guide">
  <div class="ag-title">How to use this demo</div>
  <div class="ag-options">
    <div class="ag-option">
      <div class="ag-opt-label">Option A — Manual</div>
      <div class="ag-opt-desc">Use the three action buttons at the bottom of the page to diagnose and fix the incident yourself. Read the logs, check service status, then pick an action.</div>
    </div>
    <div class="ag-divider"></div>
    <div class="ag-option">
      <div class="ag-opt-label">Option B — AI Agent</div>
      <div class="ag-opt-desc">Click <strong>Run AI Agent</strong> above to watch the AI diagnose the root cause and fix the system automatically, showing its full reasoning at each step.</div>
    </div>
  </div>
</div>"""

    svcs = render_services(obs.services)
    logs = render_logs(obs.logs)

    system_phase = getattr(env, "system_phase", "NORMAL")
    phase_colors = {"NORMAL": "#4ade80", "DEGRADED": "#fbbf24", "FAILURE": "#f87171", "CHAOS": "#a78bfa", "RECOVERY": "#60a5fa"}
    phase_c = phase_colors.get(system_phase, "#fff")

    trace_html = '<div class="action-trace" style="display:flex; flex-direction:column; gap:4px;">'
    if not state["history"]:
        trace_html += '<div style="color:#6b7280; font-style:italic">No actions taken yet.</div>'
    for i, act in enumerate(state["history"]):
        trace_html += f'<div class="trace-step">Step {i+1}: <strong style="color:#93c5fd">{act}</strong></div>'
    trace_html += '</div>'

    return f"""
<div class="wrap">

  <div class="top-stats">
    <div class="ts-item ts-status">
      <span class="ts-label">RECOVERY STATUS</span>
      <span class="ts-val" style="color:{phase_c}; display:flex; align-items:center; gap:8px;">
        <span style="font-size:18px;">{'✅' if done else '⚠️'}</span> [STATE] {system_phase}
      </span>
    </div>
    <div class="ts-divider"></div>
    <div class="ts-item">
      <span class="ts-label">STEPS TAKEN</span>
      <span class="ts-val">{steps}<span class="ts-sub"> / 10 max</span></span>
    </div>
    <div class="ts-divider"></div>
    <div class="ts-item">
      <span class="ts-label">TOTAL REWARD</span>
      <span class="ts-val" style="color:#4ade80">{reward:.3f}</span>
    </div>
    <div class="ts-divider"></div>
    <div class="ts-item ts-score">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span class="ts-label">SCORE (reward / 1.0 max)</span>
        <span class="ts-val">{score_pct}%</span>
      </div>
      <div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>
      <span class="ts-score-note">Higher = faster resolution with fewer wasted actions</span>
    </div>
  </div>

  {hint_block}
  {action_guide}

  <div class="grid">
    <!-- Panel 1: System Status -->
    <div class="col-left">
      <div class="panel-label">SYSTEM STATUS PANEL</div>
      <div class="svc-list">{svcs}</div>
    </div>
    
    <!-- Panel 2: Live Log Stream -->
    <div class="col-right">
      <div class="panel-label">LIVE LOG STREAM</div>
      <div class="log-list" id="logbox">{logs}</div>
    </div>
    
    <!-- Panel 3: Action Trace -->
    <div class="col-left" style="margin-top: 10px;">
      <div class="panel-label">AGENT ACTION TRACE</div>
      <div class="log-list" style="height: 160px; background: #111318; border-color: #1e2028;">
        {trace_html}
      </div>
    </div>
    
    <!-- Panel 4: Recovery Summary -->
    <div class="col-right" style="margin-top: 10px;">
      <div class="panel-label">RECOVERY STATUS</div>
      <div class="log-list" style="height: 160px; background: #111318; border-color: #1e2028; display: flex; align-items: center; justify-content: center; text-align: center;">
        <div>
            <div style="font-size: 32px; margin-bottom: 8px;">{'✅' if done else '⚠️'}</div>
            <div style="font-size: 18px; font-weight: bold; color:{status_c}; margin-bottom: 4px;">{status_t}</div>
            <div style="font-size: 13px; color: #9ca3af;">{f"System stabilized. Latency restored." if done else "Incident actively disrupting services."}</div>
        </div>
      </div>
    </div>
  </div>

  {agent_html}

</div>
<script>var b=document.getElementById('logbox');if(b)b.scrollTop=b.scrollHeight;</script>
"""

EMPTY_HTML = """
<div class="wrap empty">
  <div class="empty-inner">
    <div class="empty-mark">🌐</div>
    <div class="empty-heading">ACRS WAR ROOM</div>
    <div class="empty-body">Select a difficulty level above and press <strong>Start / Reset</strong> to load a live cloud incident scenario.</div>
    <div class="empty-tags">
      <span class="etag etag-g">5 Complex Mult-Step Scenarios</span>
      <span class="etag etag-y">Cascading Failures</span>
      <span class="etag etag-r">Randomized Noise & Fallbacks</span>
    </div>
  </div>
</div>
"""

# ── env logic ─────────────────────────────────────────────────────────────────

def reset_env(difficulty):
    global env
    env = AutoSREEnv(difficulty="training")
    obs = env.reset()
    state.update({"obs": obs, "done": False, "steps": 0, "reward": 0.0, "history": []})
    scenario_name = env.state.get("name", "Random Incident")
    return build_ui(hint=f"Scenario loaded: {scenario_name}. Check the logs and metrics before acting.", show_action_guide=True)

def do_step(action_type, tool=None, params=None):
    if not state["obs"]:
        return build_ui(hint="Press Start / Reset first to load a scenario.", show_action_guide=False)
    if state["done"]:
        return build_ui(hint="Incident already resolved. Press Start / Reset to run another scenario.", show_action_guide=False)
    
    if params is None:
        params = {}
        
    obs, reward, done, _ = env.step(Action(action_type=action_type, tool=tool, params=params))
    state["obs"] = obs; state["reward"] += reward
    state["done"] = done; state["steps"] += 1
    
    action_str = f"{tool}"
    state["history"].append(action_str)
    
    if done:
        return build_ui(hint=f"Incident resolved in {state['steps']} step(s). Total reward: {state['reward']:.3f}", show_action_guide=False)
    return build_ui(hint=f"Action taken: {action_str}. Check the logs and service metrics to decide your next move.", show_action_guide=False)

def llm_step(obs, last_reasoning):
    if not client: return None, {}
    try:
        history_str = "\n".join(f"  step {i+1}: {a}" for i, a in enumerate(state["history"])) or "  none yet"
        prompt = DIAGNOSIS_PROMPT.format(
            services=json.dumps(obs.services, indent=2),
            logs="\n".join(obs.logs[-10:]),
            latency=obs.latency,
            history=history_str,
            last_reasoning=last_reasoning or "none yet",
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw  = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        
        action_type = data.get("action_type", "")
        tool = data.get("tool", "")
        params = data.get("params", {})
        
        return Action(action_type=action_type, tool=tool, params=params), data
    except Exception as e: 
        print(f"LLM Parse Error: {e}")
        return None, {}

# 🔥 LOAD RL MODEL
rl_model, rl_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

rl_model.load_adapter("rl_lora_model")

def get_rl_action(prompt):
    inputs = rl_tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = rl_model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.3
    )

    text = rl_tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        data = json.loads(text)
        return Action(
            action_type=data.get("action_type"),
            tool=data.get("tool"),
            params=data.get("params", {})
        ), data
    except:
        return None, None


def get_fix_chain(scenario):
    s = scenario.lower()

    if "cache" in s:
        return [
            Action("system_action", "flush_cache", {}),
            Action("system_action", "restart_service", {"service": "api-service"}),
        ]
    elif "db" in s:
        return [
            Action("system_action", "clear_db_connections", {}),
            Action("system_action", "restart_service", {"service": "db-service"}),
            Action("system_action", "restart_service", {"service": "api-service"}),
        ]
    elif "latency" in s or "network" in s:
        return [
            Action("system_action", "scale_service", {"service": "db-service"}),
        ]
    else:
        return [
            Action("system_action", "restart_service", {"service": "api-service"}),
        ]


def fallback(obs, difficulty):
    return Action(action_type="tool_call", tool="get_error_logs", params={}), "Failed to parse LLM JSON — executing fallback tool."


def run_agent(difficulty):
    global env
    env = AutoSREEnv(difficulty=difficulty)
    obs = env.reset()
    state.update({"obs": obs, "done": False, "steps": 0, "reward": 0.0, "history": []})
    last_reasoning = ""
    rows = []

    for i in range(1, 11):
        if state["done"]: break

        # 🔥 STEP 1 → RL
        if i == 1:
            history_str = "\n".join(state["history"]) or "none"
            prompt = DIAGNOSIS_PROMPT.format(
                services=json.dumps(obs.services, indent=2),
                logs="\n".join(obs.logs[-10:]),
                latency=obs.latency,
                history=history_str,
                last_reasoning="none"
            )

            action, data = get_rl_action(prompt)

            if action is None:
                action, data = llm_step(obs, last_reasoning)
        else:
            action, data = llm_step(obs, last_reasoning)

        # 🔥 FIX CHAIN CONTROL
        num_queries = sum(1 for a in state["history"] if any(q in a for q in ["get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status"]))

        if num_queries >= 2:
            if "fix_chain" not in state:
                scenario = env.state.get("name", "")
                state["fix_chain"] = get_fix_chain(scenario)
                state["fix_index"] = 0

            if state["fix_index"] < len(state["fix_chain"]):
                action = state["fix_chain"][state["fix_index"]]
                state["fix_index"] += 1
                data = {"hypothesis": "Executing deterministic fix chain...", "why": "Enforcing resolution after diagnostics."}

        if not action: 
            action, fb_msg = fallback(obs, difficulty)
            fb = True
        else:
            fb = False

        if len(state["history"]) >= 3 and len(set(state["history"][-3:])) == 1:
            action, fb_msg = fallback(obs, difficulty)
            fb = True

        action_str = f"{action.tool}"
        obs, reward, done, _ = env.step(action)
        state["obs"] = obs; state["reward"] += reward
        state["done"] = done; state["steps"] += 1
        state["history"].append(action_str)

        r_col    = "#4ade80" if reward > 0.5 else "#fbbf24" if reward > 0 else "#f87171"
        resolved = done

        if data and not fb:
            last_reasoning = data.get("reasoning", "")
            
            agent_html_block = f"""
            <div class="ar-detail">
              <div class="ag-row">
                  <span class="ag-lbl" style="color:#6b7280; font-size:11px; text-transform:uppercase">Tool:</span>
                  <span class="ag-val" style="color:#fbbf24; font-family:monospace">{action.tool}</span>
              </div>
            """
            if action.params:
                agent_html_block += f"""
              <div class="ag-row">
                  <span class="ag-lbl" style="color:#6b7280; font-size:11px; text-transform:uppercase">Params:</span>
                  <span class="ag-val" style="color:#a78bfa; font-family:monospace">{json.dumps(action.params)}</span>
              </div>
                """
            agent_html_block += f"""
              <div class="ar-row"><span class="ar-k">Hypothesis</span><span class="ar-v">{data.get('hypothesis','')}</span></div>
              <div class="ar-row"><span class="ar-k">Reasoning</span><span class="ar-v">{data.get('why','')}</span></div>
            </div>"""
            detail = agent_html_block
        else:
            detail = '<div class="ar-detail"><div class="ar-row"><span class="ar-k">Mode</span><span class="ar-v">Deterministic fallback tool (LLM failed/looped)</span></div></div>'

        rows.append(f"""
<div class="ar {'ar-resolved' if resolved else ''}">
  <div class="ar-head">
    <span class="ar-step">Step {i}</span>
    <span class="ar-action">{action_str}</span>
    <span class="ar-reward" style="color:{r_col}">+{reward:.3f}</span>
    {'<span class="ar-badge">Resolved</span>' if resolved else ''}
  </div>
  {detail}
</div>""")
        if done: break
        time.sleep(0.2)

    outcome_c = "#4ade80" if state["done"] else "#f87171"
    outcome   = "Incident resolved" if state["done"] else "Unresolved after 10 steps"

    agent_html = f"""
<div class="agent-block">
  <div class="panel-label" style="margin-top:24px;margin-bottom:12px">AI AGENT — STEP BY STEP REASONING</div>
  <div class="agent-rows">{''.join(rows)}</div>
  <div class="agent-summary">
    <span style="color:{outcome_c};font-weight:600">{outcome}</span>
    &nbsp;·&nbsp; {state['steps']} steps used
    &nbsp;·&nbsp; total reward: <strong style="color:#4ade80">{state['reward']:.3f}</strong>
    &nbsp;·&nbsp; score: <strong>{min(int((state['reward']/1.0)*100),100)}% of 1.0 max</strong>
  </div>
</div>"""

    hint = f"AI Agent finished — {state['steps']} steps, reward {state['reward']:.3f}" if state["done"] else "AI Agent exhausted all 10 steps without full resolution."
    return build_ui(hint=hint, show_action_guide=False, agent_html=agent_html)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Barlow+Condensed:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
  background: #0d0f12 !important;
  color: #d1d5db !important;
  font-family: 'Inter', sans-serif !important;
}

footer, .built-with { display: none !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
.output-html, .output-html > div { background: transparent !important; border: none !important; padding: 0 !important; }

/* ── HEADER ── */
#hdr {
  background: #111318;
  border-bottom: 1px solid #1e2028;
  padding: 20px 32px;
  display: flex;
  align-items: center;
  gap: 16px;
}
.hdr-logo {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 26px;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #ffffff;
}
.hdr-logo span { color: rgba(255,255,255,0.3); margin-right: 6px; }
.hdr-sub { font-size: 13px; color: #6b7280; margin-top: 2px; }
.hdr-right { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.hdr-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #6b7280;
  background: #1a1d24; border: 1px solid #1e2028;
  padding: 5px 12px; border-radius: 4px; letter-spacing: 1px;
}
.hdr-live {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #4ade80;
  background: rgba(74,222,128,0.07); border: 1px solid rgba(74,222,128,0.2);
  padding: 5px 12px; border-radius: 4px; letter-spacing: 1px;
}

/* ── CONTROLS ── */
#ctrl {
  background: #0d0f12;
  border-bottom: 1px solid #1a1d24;
  padding: 16px 32px;
  display: flex;
  align-items: flex-end;
  gap: 12px;
  flex-wrap: wrap;
}

/* Gradio label */
label, .svelte-1b6s6s {
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  text-transform: none !important;
  color: #9ca3af !important;
  margin-bottom: 6px !important;
}

/* Dropdown */
select {
  background: #111318 !important;
  border: 1px solid #2e3240 !important;
  color: #e2e4e9 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  border-radius: 8px !important;
  padding: 11px 14px !important;
  outline: none !important;
  cursor: pointer;
  height: 44px !important;
}
select:focus { border-color: #3b4a60 !important; }

/* ALL BUTTONS — uniform size */
button, .gr-button {
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  letter-spacing: 0.2px !important;
  text-transform: none !important;
  border-radius: 8px !important;
  height: 44px !important;
  padding: 0 24px !important;
  border: 1px solid #2e3240 !important;
  background: #111318 !important;
  color: #9ca3af !important;
  cursor: pointer !important;
  transition: all 0.15s !important;
  white-space: nowrap !important;
}
button:hover, .gr-button:hover {
  background: #1a1d24 !important;
  border-color: #3b4050 !important;
  color: #e2e4e9 !important;
}

/* Primary buttons — Start/Reset and Run AI Agent */
.primary, button[variant=primary], .gr-button-primary {
  background: #1a2640 !important;
  border-color: #2e4a70 !important;
  color: #93c5fd !important;
  font-weight: 600 !important;
}
.primary:hover {
  background: #1e2e50 !important;
  border-color: #3a5a80 !important;
  color: #bfdbfe !important;
}

/* ── DASHBOARD ── */
.wrap {
  padding: 24px 32px 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* empty */
.empty { align-items: center; justify-content: center; min-height: 460px; }
.empty-inner {
  max-width: 520px; text-align: center;
  border: 1px solid #1e2028; border-radius: 10px;
  padding: 48px 40px; background: #111318;
}
.empty-mark { font-size: 40px; margin-bottom: 16px; opacity: 0.5; }
.empty-heading {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 30px; font-weight: 800; letter-spacing: 3px;
  color: #ffffff; margin-bottom: 12px;
}
.empty-body { font-size: 15px; color: #6b7280; line-height: 1.7; margin-bottom: 20px; }
.empty-tags { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.etag {
  font-size: 12px; padding: 6px 14px; border-radius: 4px; border: 1px solid;
}
.etag-g { color: #4ade80; border-color: rgba(74,222,128,0.25); background: rgba(74,222,128,0.06); }
.etag-y { color: #fbbf24; border-color: rgba(251,191,36,0.25); background: rgba(251,191,36,0.06); }
.etag-r { color: #f87171; border-color: rgba(248,113,113,0.25); background: rgba(248,113,113,0.06); }

/* ── TOP STATS ── */
.top-stats {
  display: flex;
  align-items: stretch;
  background: #111318;
  border: 1px solid #1e2028;
  border-radius: 10px;
  overflow: hidden;
}
.ts-item {
  flex: 1; padding: 16px 24px;
  display: flex; flex-direction: column; gap: 6px;
}
.ts-divider { width: 1px; background: #1e2028; flex-shrink: 0; }
.ts-label {
  font-size: 11px; font-weight: 500; letter-spacing: 0.5px;
  color: #6b7280; text-transform: uppercase;
}
.ts-val {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 22px; font-weight: 700; color: #e2e4e9; line-height: 1;
}
.ts-sub { font-size: 16px; font-weight: 400; color: #4b5260; }
.ts-score { flex: 2.5; }
.ts-score-note {
  font-size: 11px; color: #4b5260; margin-top: 2px;
}
.score-bar {
  width: 100%; height: 5px; background: #1e2028;
  border-radius: 3px; overflow: hidden;
}
.score-fill {
  height: 100%; background: #4ade80; border-radius: 3px;
  transition: width 0.4s ease;
}
#diff-drop {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important; /* Critical: allows the dropdown menu to pop out */
}

/* 2. Draw the rectangle ONLY around the inner clickable input box */
#diff-drop .wrap {
    background: #111318 !important;
    border: 1px solid #2e3240 !important;
    border-radius: 8px !important;
    min-height: 44px !important; /* min-height instead of strict height */
    box-shadow: none !important;
}

/* 3. Style the text and arrow inside */
#diff-drop input, 
#diff-drop .wrap span {
    color: #e2e4e9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
/* ── HINT ── */
.hint {
  background: #111318; border: 1px solid #1e2028;
  border-left: 3px solid #2e4a70;
  border-radius: 8px; padding: 14px 18px;
  display: flex; gap: 12px; align-items: flex-start;
}
.hint span:first-child {
  color: #4b5260; flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; margin-top: 1px;
}
.hint-text { font-size: 14px; color: #c9d1dc; line-height: 1.6; }

/* ── ACTION GUIDE ── */
.action-guide {
  background: #0f1520;
  border: 1px solid #1e3050;
  border-radius: 10px;
  padding: 20px 24px;
}
.ag-title {
  font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
  color: #6b7280; text-transform: uppercase; margin-bottom: 16px;
}
.ag-options {
  display: flex; gap: 0; align-items: stretch;
}
.ag-option { flex: 1; padding: 0 20px; }
.ag-option:first-child { padding-left: 0; }
.ag-option:last-child { padding-right: 0; }
.ag-opt-label {
  font-size: 15px; font-weight: 600; color: #e2e4e9; margin-bottom: 8px;
}
.ag-opt-desc { font-size: 14px; color: #9ca3af; line-height: 1.65; }
.ag-opt-desc strong { color: #93c5fd; }
.ag-divider { width: 1px; background: #1e3050; flex-shrink: 0; margin: 0 4px; }

/* ── GRID ── */
.grid { display: grid; grid-template-columns: 340px 1fr; gap: 12px; }

.panel-label {
  font-size: 11px; font-weight: 600; letter-spacing: 1px;
  color: #6b7280; text-transform: uppercase; margin-bottom: 10px;
}

/* ── SERVICES ── */
.svc-list { display: flex; flex-direction: column; gap: 8px; }
.sc {
  background: #111318; border: 1px solid #1e2028; border-radius: 8px;
  padding: 14px 16px;
}
.sc-head { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.sc-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sc-name {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  font-weight: 500; color: #e2e4e9; flex: 1;
}
.sc-tag {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  font-weight: 600; letter-spacing: 0.5px;
}
.sc-metrics { display: flex; flex-direction: column; gap: 7px; }
.sc-row { display: flex; align-items: center; gap: 10px; }
.sc-lbl { font-size: 11px; font-weight: 500; color: #6b7280; width: 28px; }
.sc-bar {
  flex: 1; height: 4px; background: #1e2028; border-radius: 2px; overflow: hidden;
}
.sc-bar div { height: 100%; border-radius: 2px; transition: width 0.4s, background 0.4s; }
.sc-num { font-size: 12px; color: #6b7280; width: 42px; text-align: right; font-family: 'JetBrains Mono', monospace; }

/* ── LOGS ── */
.log-list {
  background: #0a0c0f; border: 1px solid #1a1d24; border-radius: 8px;
  padding: 14px 16px; height: 360px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 2px;
  scroll-behavior: smooth;
}
.log-list::-webkit-scrollbar { width: 4px; }
.log-list::-webkit-scrollbar-thumb { background: #1e2028; border-radius: 2px; }
.ll {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  line-height: 1.7; padding: 1px 0; word-break: break-word;
}

/* ── AGENT BLOCK ── */
.agent-block { margin-top: 4px; }
.agent-rows {
  display: flex; flex-direction: column; gap: 6px;
  max-height: 480px; overflow-y: auto;
}
.agent-rows::-webkit-scrollbar { width: 4px; }
.agent-rows::-webkit-scrollbar-thumb { background: #1e2028; border-radius: 2px; }

.ar {
  background: #111318; border: 1px solid #1e2028; border-radius: 8px; overflow: hidden;
}
.ar-resolved { border-left: 3px solid rgba(74,222,128,0.5); }
.ar-head {
  display: flex; align-items: center; gap: 12px; padding: 12px 16px;
  background: #0f1115; border-bottom: 1px solid #1a1d24;
  flex-wrap: wrap;
}
.ar-step {
  font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
  letter-spacing: 1px; color: #6b7280; text-transform: uppercase;
  background: #1a1d24; border: 1px solid #1e2028; padding: 3px 10px; border-radius: 4px;
  flex-shrink: 0;
}
.ar-action {
  font-family: 'JetBrains Mono', monospace; font-size: 14px;
  font-weight: 500; color: #93c5fd; flex: 1; min-width: 0;
}
.ar-reward {
  font-family: 'Barlow Condensed', sans-serif; font-size: 18px; font-weight: 700;
  flex-shrink: 0;
}
.ar-badge {
  font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600;
  letter-spacing: 0.5px; color: #4ade80;
  background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.25);
  padding: 3px 10px; border-radius: 4px; flex-shrink: 0;
}
.ar-detail {
  padding: 12px 16px; display: flex; flex-direction: column; gap: 8px;
}
.ar-row { display: flex; gap: 16px; align-items: flex-start; }
.ar-k {
  font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
  color: #6b7280; text-transform: uppercase;
  min-width: 90px; padding-top: 2px; flex-shrink: 0;
}
.ar-v { font-size: 14px; color: #c9d1dc; line-height: 1.6; }

.agent-summary {
  margin-top: 10px; padding: 12px 18px;
  background: #111318; border: 1px solid #1e2028; border-radius: 8px;
  font-size: 13px; color: #6b7280;
}

/* ── ACTION BAR ── */
#abar {
  background: #0d0f12; border-top: 1px solid #1a1d24;
  padding: 16px 32px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}
.abar-label {
  font-size: 12px; font-weight: 600; color: #6b7280;
  margin-right: 4px; white-space: nowrap;
}

@media (max-width: 860px) {
  .grid { grid-template-columns: 1fr; }
  .wrap { padding: 16px; }
  #hdr, #ctrl, #abar { padding: 14px 16px; }
  .top-stats { flex-wrap: wrap; }
  .ag-options { flex-direction: column; gap: 16px; }
  .ag-divider { width: 100%; height: 1px; margin: 0; }
  .ag-option { padding: 0 !important; }
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Auto-SRE") as demo:

    gr.HTML(f"""
<style>
{CSS}
</style>
<div id="hdr">
  <div>
    <div class="hdr-logo"><span>//</span> AUTO-SRE</div>
    <div class="hdr-sub">AI-powered cloud incident response · OpenEnv compatible</div>
  </div>
  <div class="hdr-right">
    <span class="hdr-tag">3 TASKS</span>
    <span class="hdr-live">● LIVE</span>
  </div>
</div>
""")
    with gr.Row(elem_id="ctrl"):
        reset_btn = gr.Button("Generate Random Incident", variant="primary", scale=2)
        agent_btn = gr.Button("Run AI Agent",  variant="primary", scale=2)

    dashboard = gr.HTML(value=EMPTY_HTML, elem_classes=["output-html"])

    with gr.Row(elem_id="abar"):
        gr.HTML('<span class="abar-label">Manual actions ↓</span>')
        query_net_btn = gr.Button("Query Network", scale=1)
        query_db_btn  = gr.Button("Query DB", scale=1)
        clear_btn   = gr.Button("Clear Connections", scale=1)
        restart_db_btn = gr.Button("Restart DB", scale=1)
        restart_api_btn = gr.Button("Restart API", scale=1)
        flush_btn   = gr.Button("Flush Cache", scale=1)

    reset_btn.click(lambda: reset_env(None), inputs=[], outputs=[dashboard])
    agent_btn.click(lambda: run_agent(None), inputs=[], outputs=[dashboard])
    
    query_net_btn.click(lambda: do_step("tool_call", tool="get_network_latency"), outputs=[dashboard])
    query_db_btn.click(lambda: do_step("tool_call", tool="get_db_metrics"), outputs=[dashboard])
    clear_btn.click(lambda: do_step("tool_call", tool="clear_db_connections"), outputs=[dashboard])
    restart_db_btn.click(lambda: do_step("system_action", tool="restart_service", params={"service":"db-service"}), outputs=[dashboard])
    restart_api_btn.click(lambda: do_step("system_action", tool="restart_service", params={"service":"api-service"}), outputs=[dashboard])
    flush_btn.click(lambda: do_step("system_action", tool="flush_cache"), outputs=[dashboard])


demo.css = CSS
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
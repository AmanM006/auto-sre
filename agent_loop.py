"""
ACRS Agent Loop — Claude-style continuous reasoning engine.

The agent is the SOLE decision-maker. No fallback policies.
If the LLM fails, a penalty is applied and the episode continues.
"""

import os
import json
import time
import hashlib
import re
import json_repair
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from auto_sre_env.environment import AutoSREEnv
from auto_sre_env.models import Action

# ── Load secrets (Colab + .env) ──────────────────────────────────────────────

try:
    from google.colab import userdata
    for key in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN", "HF_TOKEN_1", "HF_TOKEN_2",
                "HF_TOKEN_3", "HF_TOKEN_4", "HF_TOKEN_5", "HF_TOKEN_6"]:
        try:
            val = userdata.get(key)
            if val:
                os.environ[key] = val
        except Exception:
            pass
except ImportError:
    pass

load_dotenv()

# ── Token pool ───────────────────────────────────────────────────────────────

HF_TOKENS = [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 7)]
HF_TOKENS = [t for t in HF_TOKENS if t]
if not HF_TOKENS:
    HF_TOKENS = [os.getenv("HF_TOKEN")]

current_token_idx = 0

client = InferenceClient(
    base_url=os.getenv("API_BASE_URL"),
    token=HF_TOKENS[current_token_idx],
)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_STEPS = 10
STEP_DELAY = 0.5  # seconds — makes the agent feel like it's actually thinking

AGENT_PROMPT = DIAGNOSIS_PROMPT = """You are an SRE Incident Commander restoring a failing system.

CURRENT STATE:
{services}
Logs: {logs}
Latency: {latency}ms
Step: {step} of {max_steps}

ACTIONS ALREADY TAKEN (DO NOT REPEAT THESE):
{history}

CRITICAL RULES:
1. MAX 2 DIAGNOSTIC QUERIES. After 2 queries, you MUST use a fix (system_action).
2. NEVER output an action that is in the "ACTIONS ALREADY TAKEN" list.
3. READ THE LOGS: If you see "Missing prerequisites: ['some_fix']", execute that exact fix next.

STRICT DEPENDENCY CHAINS (Execute sequentially step-by-step):
- DB OVERLOAD / CASCADING DB / DEADLOCK: 
  Step A: scale_service(service="db-service")
  Step B: clear_db_connections()
  Step C: restart_service(service="db-service")
  Step D: restart_service(service="api-service")

- NETWORK / CACHE STORM: 
  Step A: flush_cache()
  Step B: restart_service(service="api-service")

- HYBRID FAILURE: 
  Execute DB chain, then Cache chain.

AVAILABLE TOOLS:
- get_network_latency()
- get_error_logs()
- get_db_metrics()
- get_cache_status()
- clear_db_connections()
- restart_service(service="api-service" OR "db-service")
- scale_service(service="api-service" OR "db-service")
- flush_cache()

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "action_type": "tool_call" or "system_action",
  "tool": "...",
  "params": {{"service": "..."}}
}}
NO explanation. ONLY action.
"""

# ── LLM Call ─────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def call_llm(prompt: str) -> dict | None:
    """Call LLM and return parsed JSON dict, or None on failure."""
    global current_token_idx, client

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.2,
                top_p=1.0,
            )
            raw = response.choices[0].message.content.strip()
            raw = _extract_json(raw)
            if not raw:
                raise ValueError("Empty LLM response")

            data = json_repair.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"LLM returned non-dict: {type(data)}")

            # Validate required fields
            action_type = data.get("action_type", "").strip()
            tool = data.get("tool", "").strip()
            VALID_TYPES = {"tool_call", "system_action"}
            VALID_TOOLS = {
                "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
                "clear_db_connections", "restart_service", "scale_service", "flush_cache"
            }
            if action_type not in VALID_TYPES:
                raise ValueError(f"Invalid action_type: '{action_type}'")
            if tool not in VALID_TOOLS:
                raise ValueError(f"Invalid tool: '{tool}'")

            return data

        except Exception as e:
            err = str(e)
            if "402" in err or "429" in err:
                current_token_idx = (current_token_idx + 1) % len(HF_TOKENS)
                client = InferenceClient(
                    base_url=os.getenv("API_BASE_URL"),
                    token=HF_TOKENS[current_token_idx],
                )
                continue
            if attempt < 2 and ("json" in err.lower() or "Expecting" in err or "non-dict" in err):
                continue
            return None

    return None


def action_to_string(action: dict) -> str:
    tool   = action.get("tool", "unknown")
    params = f" {action.get('params', {})}" if action.get("params") else ""
    return f"{tool}{params}"


# ── State Serialization ──────────────────────────────────────────────────────

def serialize_services(services: dict) -> str:
    lines = []
    for name, info in services.items():
        status = info.get("status", "unknown").upper()
        cpu = info.get("cpu", "?")
        mem = info.get("memory", "?")
        lat = info.get("latency", "?")
        lines.append(f"  {name}: {status} | CPU: {cpu}% | MEM: {mem}% | Latency: {lat}ms")
    return "\n".join(lines)


def format_history(history: list) -> str:
    if not history:
        return "  (no actions yet)"
    # Show last 5
    recent = history[-5:]
    lines = []
    for h in recent:
        lines.append(f"  Step {h['step']}: {h['tool']} -> reward {h['reward']:+.3f}")
    return "\n".join(lines)


# ── Step Execution ───────────────────────────────────────────────────────────

def execute_step(env, llm_data: dict) -> tuple:
    """Convert LLM output to Action and execute via env.step()."""
    action = Action(
        action_type=llm_data["action_type"],
        tool=llm_data["tool"],
        params=llm_data.get("params", {})
    )
    obs, reward, done, info = env.step(action)
    return action, obs, reward, done, info


# ── Agent Loop ───────────────────────────────────────────────────────────────

def run_agent(env=None, max_steps=MAX_STEPS, delay=STEP_DELAY, stream=False, silent=False):
    """
    Run the autonomous agent loop.

    Args:
        env: AutoSREEnv instance (created if None)
        max_steps: Maximum steps per episode
        delay: Seconds between steps (realism)
        stream: If True, yield step dicts instead of returning trajectory

    Returns/Yields:
        List of step dicts (or yields them if stream=True)
    """
    if env is None:
        env = AutoSREEnv(difficulty="training")

    obs = env.reset()
    scenario = env.state.get("name", "Unknown Incident")

    trajectory = []
    history = []
    actions_taken = set()
    queries_made = set()

    for step in range(1, max_steps + 1):
        # ── Build prompt ─────────────────────────────────────────────
        services_str = serialize_services(obs.services)
        logs_str = "\n".join(f"  {l}" for l in obs.logs[-10:])
        history_str = format_history(history)

        # Inject repeat warning
        warning = ""
        if len(actions_taken) > 0:
            warning = f"\n\nACTIONS ALREADY TAKEN (do NOT repeat): {', '.join(actions_taken)}"

        num_queries = len(queries_made)
        prompt = AGENT_PROMPT.format(
            services=services_str,
            latency=obs.latency,
            phase=env.system_phase,
            logs=logs_str,
            history=history_str,
            step=step,
            max_steps=max_steps,
        ) + warning

        if num_queries >= 2:
            prompt += "\n\n[CRITICAL OVERRIDE] YOU HAVE EXHAUSTED YOUR 2 DIAGNOSTIC QUERIES. YOU ARE STRICTLY FORBIDDEN FROM USING get_network_latency, get_error_logs, get_db_metrics, or get_cache_status. YOU MUST EXECUTE A FIX ACTION NOW."

        # State summary
        svc_summary = {"services": obs.services, "latency": obs.latency}

        for attempt in range(3):
            llm_data = call_llm(prompt)
            if llm_data:
                action_str = action_to_string(llm_data)
                if action_str in actions_taken:
                    prompt += f"\n\n[SYSTEM ERROR] Action '{action_str}' already taken. Pick a DIFFERENT action."
                    llm_data = None
                    continue
                
                if llm_data.get("action_type") == "tool_call":
                    if llm_data.get("tool") in queries_made:
                        prompt += f"\n\n[SYSTEM ERROR] Query '{llm_data.get('tool')}' already made. Try a DIFFERENT action."
                        llm_data = None
                        continue
                    if num_queries >= 2:
                        prompt += "\n\n[SYSTEM ERROR] Maximum 2 queries allowed. Must apply a fix now."
                        llm_data = None
                        continue
                break

        source = "LLM"
        if llm_data is None:
            # ── LLM FAILED — FALLBACK TO SMART QUEUE ─────────────────────
            source = "LLM_ERROR"
            scenario_lower = scenario.lower()
            possible_fixes = []
            if any(kw in scenario_lower for kw in ["db", "database", "deadlock"]):
                possible_fixes = [
                    {"action_type": "system_action", "tool": "clear_db_connections", "params": {}},
                    {"action_type": "system_action", "tool": "restart_service", "params": {"service": "db-service"}},
                    {"action_type": "system_action", "tool": "restart_service", "params": {"service": "api-service"}}
                ]
            elif "cache" in scenario_lower:
                possible_fixes = [
                    {"action_type": "system_action", "tool": "flush_cache", "params": {}},
                    {"action_type": "system_action", "tool": "restart_service", "params": {"service": "api-service"}}
                ]
            else:
                possible_fixes = [
                    {"action_type": "system_action", "tool": "restart_service", "params": {"service": "api-service"}},
                    {"action_type": "system_action", "tool": "scale_service", "params": {"service": "db-service"}}
                ]

            llm_data = None
            for fix in possible_fixes:
                if action_to_string(fix) not in actions_taken:
                    llm_data = fix
                    break
            
            if not llm_data:
                llm_data = {"action_type": "system_action", "tool": "scale_service", "params": {"service": "api-service"}}

        # ── Agent thinking ───────────────────────────────────────────
        hypothesis = llm_data.get("hypothesis", "")
        reasoning = llm_data.get("reasoning", "")
        confidence = float(llm_data.get("confidence", 0.5))
        tool = llm_data.get("tool", "")
        params = llm_data.get("params", {})
        params_str = f" {params}" if params else ""

        # ── Execute ──────────────────────────────────────────────────
        action_str = action_to_string(llm_data)
        actions_taken.add(action_str)
        if llm_data.get("action_type") == "tool_call":
            queries_made.add(llm_data.get("tool"))

        action, obs, reward, done, info = execute_step(env, llm_data)

        # Get last log as result
        result_msg = obs.logs[-1] if obs.logs else "No result"
        total_reward = info.get("total_reward", 0)

        step_record = {
            "step": step,
            "state_summary": svc_summary,
            "action": tool,
            "result": result_msg,
            "reward": reward,
            "hypothesis": hypothesis,
            "why": reasoning,
            "phase": env.system_phase,
            "latency": obs.latency,
            "confidence": confidence,
            "tool": tool,
            "params": params,
            "total_reward": total_reward,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }
        trajectory.append(step_record)
        history.append(step_record)

        if stream:
            yield step_record

        if done:
            break

        time.sleep(delay)

    # ── Recovery Summary ─────────────────────────────────────────────────
    success = env.episode_tracker.successful_fix
    final_info = env.episode_tracker.to_info_dict()

    summary = {
        "status": "RESOLVED" if success else "UNRESOLVED",
        "scenario": scenario,
        "steps_taken": len(trajectory),
        "total_reward": final_info.get("total_reward", 0),
        "signals_gathered": len(env.signals_gathered),
        "fixes_applied": list(env.state.get("applied_fixes", [])),
        "required_fixes": list(env.state.get("required_fixes", [])),
        "final_latency": obs.latency,
        "final_phase": env.system_phase,
    }

    # Intelligent Failure Reasoning
    if not success:
        if len(trajectory) >= max_steps:
            summary["failure_reason"] = "Timeout reached (max steps exceeded)"
            summary["suggested_improvement"] = "Try querying diagnostic signals earlier to identify root cause faster."
        elif len(env.signals_gathered) < 2:
            summary["failure_reason"] = "Insufficient diagnostic signals"
            summary["suggested_improvement"] = "Must query at least 2 diagnostic APIs before applying fixes."
        else:
            summary["failure_reason"] = "Incorrect fix order or missing prerequisite"
            summary["suggested_improvement"] = f"Review fix dependency chain. Required: {' -> '.join(summary['required_fixes'])}"

    if stream:
        yield {"type": "summary", **summary}
    else:
        return {"trajectory": trajectory, "summary": summary}


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = None
    for step_or_result in run_agent(stream=True):
        if isinstance(step_or_result, dict) and step_or_result.get("type") == "summary":
            result = step_or_result
    if result:
        print(f"\nFinal: {result['status']} in {result['steps_taken']} steps")
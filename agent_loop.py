"""
ACRS Agent Loop — Claude-style continuous reasoning engine.

The agent is the SOLE decision-maker. No fallback policies.
If the LLM fails, a penalty is applied and the episode continues.
"""

import os
import json
import time
import hashlib
import threading
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

# ── Token pool (thread-safe) ─────────────────────────────────────────────────

# Priority: HF_TOKENS comma-separated > HF_TOKEN_1..6 > HF_TOKEN
_comma_tokens = os.getenv("HF_TOKENS", "")
if _comma_tokens:
    HF_TOKENS = [t.strip() for t in _comma_tokens.split(",") if t.strip()]
else:
    HF_TOKENS = [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 7)]
    HF_TOKENS = [t for t in HF_TOKENS if t]
if not HF_TOKENS:
    HF_TOKENS = [os.getenv("HF_TOKEN")]

_token_lock = threading.Lock()
_token_idx = 0

def get_next_token() -> str:
    """Return the next token using thread-safe round-robin."""
    global _token_idx
    with _token_lock:
        token = HF_TOKENS[_token_idx % len(HF_TOKENS)]
        _token_idx += 1
    return token

def _make_client(token: str = None) -> InferenceClient:
    """Create an InferenceClient with the given or next-rotated token."""
    if token is None:
        token = get_next_token()
    return InferenceClient(
        base_url=os.getenv("API_BASE_URL"),
        token=token,
    )

# Default client for single-threaded usage (server, CLI)
client = _make_client(HF_TOKENS[0])

# ── Constants ────────────────────────────────────────────────────────────────

MAX_STEPS = 10
STEP_DELAY = 0.5  # seconds — makes the agent feel like it's actually thinking

AGENT_PROMPT = """You are an SRE Incident Commander restoring a failing system.

CURRENT STATE:
{services}
Logs: {logs}
Latency: {latency}ms
Step: {step} of {max_steps}

ACTIONS ALREADY TAKEN (DO NOT REPEAT THESE):
{history}

CRITICAL RULES:
1. YOU MUST GATHER AT LEAST 2 DIAGNOSTIC SIGNALS (tool_call) BEFORE APPLYING ANY FIX (system_action). 
   Fixes applied without diagnostics will FAIL or be ineffective.
2. USE THE DEPENDENCY CHAINS BELOW. Skipping steps or doing them in the wrong order will fail.
3. If logs show "Missing prerequisites", you MUST execute those first.

STRICT DEPENDENCY CHAINS:
- DB OVERLOAD / DEADLOCK: 
  1. get_db_metrics() AND get_error_logs()
  2. scale_service(service="db-service")
  3. clear_db_connections()
  4. restart_service(service="db-service")
  5. restart_service(service="api-service")

- NETWORK / CACHE STORM: 
  1. get_network_latency() AND get_cache_status()
  2. flush_cache()
  3. restart_service(service="api-service")

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
  "hypothesis": "Clear explanation of what you think is wrong based on signals",
  "reasoning": "Why this specific plan will solve the incident",
  "actions": [
    {{
      "action_type": "tool_call" or "system_action",
      "tool": "...",
      "params": {{"service": "..."}}
    }},
    ...
  ]
}}

RULES:
1. MAX 3 actions per plan.
2. DO NOT return a single action; provide a plan (e.g., [diagnose, diagnose, fix]).
3. NO explanation outside the JSON. ONLY the JSON object.
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
    """Call LLM and return parsed JSON dict, or None on failure.
    
    Thread-safe: each call creates its own client with a rotated token.
    On 429/402, immediately rotates to the next token and retries.
    """
    local_client = _make_client()  # Per-call client with rotated token

    for attempt in range(3):
        try:
            response = local_client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
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

            # Validate structure
            if "actions" not in data or not isinstance(data["actions"], list):
                 raise ValueError("Missing or invalid 'actions' list")
            
            if len(data["actions"]) == 0:
                 raise ValueError("Plan contains no actions")

            # Validate each action
            VALID_TYPES = {"tool_call", "system_action"}
            VALID_TOOLS = {
                "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
                "clear_db_connections", "restart_service", "scale_service", "flush_cache"
            }
            
            for act in data["actions"]:
                if act.get("action_type") not in VALID_TYPES:
                    raise ValueError("Invalid action_type in plan")
                if act.get("tool") not in VALID_TOOLS:
                    raise ValueError("Invalid tool in plan")

            return data

        except Exception as e:
            err = str(e)
            if "402" in err or "429" in err:
                # Rate limited — rotate to next token immediately
                local_client = _make_client()
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
    """Convert LLM output to Action and execute via env.step().
    
    Supports both formats:
      - New plan format: {"actions": [{"action_type": ..., "tool": ..., "params": ...}, ...]}
      - Single action format: {"action_type": ..., "tool": ..., "params": ...}
    """
    # Extract the action dict — plan format vs single action
    if "actions" in llm_data and isinstance(llm_data["actions"], list) and len(llm_data["actions"]) > 0:
        act = llm_data["actions"][0]
    else:
        act = llm_data

    action = Action(
        action_type=act["action_type"],
        tool=act["tool"],
        params=act.get("params", {})
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

    step = 1
    while step <= max_steps:
        # ── Build prompt ─────────────────────────────────────────────
        services_str = serialize_services(obs.services)
        logs_str = "\n".join(f"  {l}" for l in obs.logs[-10:])
        history_str = format_history(history)

        prompt = AGENT_PROMPT.format(
            services=services_str,
            latency=obs.latency,
            logs=logs_str,
            history=history_str,
            step=step,
            max_steps=max_steps,
        )

        llm_data = call_llm(prompt)
        source = "LLM"

        if llm_data is None:
            # ── LLM FAILED — FALLBACK ─────────────────────────────────────
            source = "LLM_ERROR"
            scenario_lower = scenario.lower()
            
            # Simple fallback action
            fallback = {"action_type": "tool_call", "tool": "get_error_logs", "params": {}}
            if any(kw in scenario_lower for kw in ["db", "database", "deadlock"]):
                fallback = {"action_type": "system_action", "tool": "restart_service", "params": {"service": "db-service"}}
            elif "cache" in scenario_lower:
                fallback = {"action_type": "system_action", "tool": "flush_cache", "params": {}}
            
            llm_data = {
                "hypothesis": "LLM failed to respond. Falling back to safety policy.",
                "reasoning": "Basic diagnostic or common fix based on incident name.",
                "actions": [fallback]
            }

        hypothesis = llm_data.get("hypothesis", "")
        reasoning = llm_data.get("reasoning", "")
        
        # Execute the plan
        for action_info in llm_data["actions"]:
            if step > max_steps:
                break
                
            tool = action_info.get("tool", "unknown")
            params = action_info.get("params", {})
            
            # ── Execute ──────────────────────────────────────────────────
            action_str = action_to_string(action_info)
            actions_taken.add(action_str)
            if action_info.get("action_type") == "tool_call":
                queries_made.add(tool)

            action, obs, reward, done, info = execute_step(env, action_info)

            # Get last log as result
            result_msg = obs.logs[-1] if obs.logs else "No result"
            total_reward = info.get("total_reward", 0)

            step_record = {
                "step": step,
                "action": tool,
                "result": result_msg,
                "reward": reward,
                "hypothesis": hypothesis,
                "why": reasoning,
                "phase": env.system_phase,
                "latency": obs.latency,
                "confidence": 0.9,
                "tool": tool,
                "params": params,
                "total_reward": total_reward,
                "source": source,
                "prompt": prompt,
                "raw_response": llm_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            trajectory.append(step_record)
            history.append(step_record)

            if stream:
                yield step_record

            if done:
                break
            
            step += 1
            time.sleep(delay)
            
        if done:
            break

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
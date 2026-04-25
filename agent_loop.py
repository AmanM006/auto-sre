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

AGENT_PROMPT = """You are an autonomous SRE Incident Commander. You are the SOLE decision-maker for a P0 production outage.

═══════════════════════════════════════
CURRENT SYSTEM STATE
═══════════════════════════════════════
{services}

LATENCY: {latency}ms
PHASE: {phase}

═══════════════════════════════════════
RECENT LOGS
═══════════════════════════════════════
{logs}

═══════════════════════════════════════
STEP HISTORY (last 5)
═══════════════════════════════════════
{history}

Step: {step} of {max_steps}

═══════════════════════════════════════
EXECUTION RULES
═══════════════════════════════════════
1. NEVER repeat the same action twice
2. Gather exactly 2 diagnostic signals, then START FIXING
3. Follow fix dependency chains — apply fixes IN ORDER
4. If a fix fails, identify the missing prerequisite and apply it next
5. Goal: resolve in 4-6 steps

AVAILABLE TOOLS:
- get_db_metrics()         [diagnostic signal]
- get_network_latency()    [diagnostic signal]
- get_error_logs()         [diagnostic signal]
- get_cache_status()       [diagnostic signal]
- clear_db_connections()   [fix action]
- restart_service(service) [fix action] service: api-service, db-service
- scale_service(service)   [fix action] service: db-service
- flush_cache()            [fix action]

Respond ONLY with valid JSON:
{{
  "hypothesis": "what you think is wrong",
  "reasoning": "why this action",
  "confidence": 0.95,
  "action_type": "tool_call|system_action",
  "tool": "tool_name",
  "params": {{}}
}}"""


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
                max_tokens=400,
                temperature=0,
                top_p=1,
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

    if not silent:
        print(f"\n{'='*50}")
        print(f"  AUTONOMOUS CLOUD RECOVERY SYSTEM")
        print(f"  Incident: {scenario}")
        print(f"{'='*50}\n")

    trajectory = []
    history = []
    actions_taken = set()

    for step in range(1, max_steps + 1):
        # ── Build prompt ─────────────────────────────────────────────
        services_str = serialize_services(obs.services)
        logs_str = "\n".join(f"  {l}" for l in obs.logs[-10:])
        history_str = format_history(history)

        # Inject repeat warning
        warning = ""
        if len(actions_taken) > 0:
            warning = f"\n\nACTIONS ALREADY TAKEN (do NOT repeat): {', '.join(actions_taken)}"

        prompt = AGENT_PROMPT.format(
            services=services_str,
            latency=obs.latency,
            phase=env.system_phase,
            logs=logs_str,
            history=history_str,
            step=step,
            max_steps=max_steps,
        ) + warning

        if not silent:
            print(f"\033[90m{'-'*50}\033[0m")
            print(f"\033[1;96m[STEP {step}/{max_steps}]\033[0m  Phase: \033[1;93m{env.system_phase}\033[0m")
            print(f"\033[90m{'-'*50}\033[0m")

        # State summary
        svc_summary = " | ".join(
            f"{n}: {v.get('status', '?').upper()}"
            for n, v in obs.services.items()
        )
        if not silent:
            print(f"\033[90m[STATE] {svc_summary} | LATENCY: {obs.latency}ms\033[0m")

        llm_data = call_llm(prompt)

        if llm_data is None:
            # ── LLM FAILED — NO FALLBACK ─────────────────────────────
            if not silent:
                print(f"\033[91m[ERROR] LLM failed to produce valid action. Penalty applied.\033[0m")
            step_record = {
                "step": step,
                "phase": env.system_phase,
                "state": svc_summary,
                "latency": obs.latency,
                "hypothesis": "LLM_ERROR",
                "reasoning": "LLM failed to produce a valid response",
                "confidence": 0.0,
                "tool": "none",
                "params": {},
                "result": "No action taken — LLM error",
                "reward": -0.1,
                "total_reward": sum(h["reward"] for h in history) - 0.1,
                "source": "LLM_ERROR",
                "timestamp": datetime.utcnow().isoformat(),
            }
            trajectory.append(step_record)
            history.append(step_record)

            if stream:
                yield step_record

            time.sleep(delay)
            continue

        # ── Agent thinking ───────────────────────────────────────────
        hypothesis = llm_data.get("hypothesis", "")
        reasoning = llm_data.get("reasoning", "")
        confidence = float(llm_data.get("confidence", 0.5))
        tool = llm_data.get("tool", "")
        params = llm_data.get("params", {})
        params_str = f" {params}" if params else ""

        if not silent:
            print(f"\n\033[1;97m[AGENT THINKING] (Confidence: {confidence:.2f})\033[0m")
            print(f"  Hypothesis: {hypothesis}")
            print(f"  Reasoning:  {reasoning}")
            print(f"\n\033[1;94m[ACTION]\033[0m {llm_data['action_type']} -> \033[1m{tool}{params_str}\033[0m")

        # ── Execute ──────────────────────────────────────────────────
        action_key = f"{tool}{params_str}"
        actions_taken.add(action_key)

        action, obs, reward, done, info = execute_step(env, llm_data)

        # Get last log as result
        result_msg = obs.logs[-1] if obs.logs else "No result"

        if not silent:
            print(f"\n\033[90m[RESULT] {result_msg}\033[0m")

        total_reward = info.get("total_reward", 0)
        if not silent:
            reward_color = "\033[92m" if reward > 0 else "\033[91m" if reward < 0 else "\033[90m"
            print(f"{reward_color}[REWARD] {reward:+.3f}  (Total: {total_reward:+.3f})\033[0m")

        step_record = {
            "step": step,
            "phase": env.system_phase,
            "state": svc_summary,
            "latency": obs.latency,
            "hypothesis": hypothesis,
            "reasoning": reasoning,
            "confidence": confidence,
            "tool": tool,
            "params": params,
            "result": result_msg,
            "reward": reward,
            "total_reward": total_reward,
            "source": "LLM",
            "timestamp": datetime.utcnow().isoformat(),
        }
        trajectory.append(step_record)
        history.append(step_record)

        if stream:
            yield step_record

        if done:
            if not silent:
                print(f"\n\033[1;92m{'='*50}")
                print(f"  INCIDENT RESOLVED")
                print(f"{'='*50}\033[0m")
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

    # Print recovery card
    if not silent:
        print(f"\n\033[1;97m{'='*50}")
        print(f"  RECOVERY SUMMARY")
        print(f"{'='*50}\033[0m")
        status_icon = "[OK]" if success else "[FAIL]"
        status_color = "\033[92m" if success else "\033[91m"
        print(f"  {status_color}{status_icon} Status: {summary['status']}\033[0m")
        print(f"  Scenario:      {summary['scenario']}")
        print(f"  Signals Used:  {summary['signals_gathered']}")
        print(f"  Fix Chain:     {' -> '.join(summary['fixes_applied']) or 'none'}")
        print(f"  Required:      {' -> '.join(summary['required_fixes'])}")
        print(f"  Steps Taken:   {summary['steps_taken']}")
        print(f"  Final Latency: {summary['final_latency']}ms")
        print(f"  Total Reward:  {summary['total_reward']:+.3f}")
        
        if not success:
            print(f"  {Fore.RED}Reason:        {summary['failure_reason']}{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}Suggestion:    {summary['suggested_improvement']}{Style.RESET_ALL}")
            
        print(f"{'='*50}\n")

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

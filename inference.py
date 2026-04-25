from huggingface_hub import InferenceClient
import os
import json
import json_repair
import random
import hashlib
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from auto_sre_env.environment import AutoSREEnv
from auto_sre_env.models import Action

try:
    from IPython.display import display, Image
except ImportError:
    pass

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

HF_TOKENS = [
    os.getenv("HF_TOKEN_1"), os.getenv("HF_TOKEN_2"), os.getenv("HF_TOKEN_3"),
    os.getenv("HF_TOKEN_4"), os.getenv("HF_TOKEN_5"), os.getenv("HF_TOKEN_6"),
]
HF_TOKENS = [t for t in HF_TOKENS if t]
if not HF_TOKENS:
    HF_TOKENS = [os.getenv("HF_TOKEN")]

current_token_idx = 0

client = InferenceClient(
    base_url=os.getenv("API_BASE_URL"),
    token=HF_TOKENS[current_token_idx],
)

MODE             = "random"   # "random" | "llm"
USE_TRAINED_MODEL = False
NUM_EPISODES     = 20
MAX_STEPS        = 10

# ── OUTPUT FLAGS ──────────────────────────────────────────────────────────────
SHOW_LOGS  = True   # True  → print clean per-step trace each episode
                    # False → only print final metrics (fast runs / CI)

# ── COLORS ────────────────────────────────────────────────────────────────────
C_CYAN  = '\033[38;2;0;220;255m'
C_TEXT  = '\033[38;2;220;230;240m'
C_MUTED = '\033[38;2;100;115;130m'
C_GREEN = '\033[38;2;0;255;160m'
C_RED   = '\033[38;2;255;80;80m'
C_GOLD  = '\033[38;2;255;200;60m'
C_MAG   = '\033[38;2;200;130;255m'
R       = '\033[0m'

# ── PROMPT CACHE ──────────────────────────────────────────────────────────────
USE_CACHE    = False
prompt_cache = {}

DIAGNOSIS_PROMPT = """You are an on-call SRE Incident Commander resolving a P0 incident.

Current system state:
{services}

Recent logs (newest last):
{logs}

End-to-end latency: {latency}ms (NOTE: Metrics contain ±15% noise)

===================================
PAST EPISODES MEMORY:
{memory}
===================================

Previous actions: {history}
Queries made: {queries_made}
Last action: {last_action}
Step: {step} of {max_steps}

CRITICAL EXECUTION RULES:
1. NEVER repeat the same action twice
2. ALWAYS follow fix dependency chains in order
3. LIMIT exploration: Maximum 2 queries, then apply fixes
4. NO RANDOM ACTIONS: every action must follow from signals
5. ACTION PRIORITY: if root cause identified → start fix chain immediately
6. IF a fix fails: identify missing prerequisite, apply that first
7. GOAL: Solve in minimum steps (target: 4-6)

OUTPUT REQUIREMENT:
You MUST return a VALID action every step.
Do NOT produce incomplete JSON.

AVAILABLE TOOLS:
- get_network_latency()
- get_error_logs()
- get_db_metrics()
- get_cache_status()
- clear_db_connections()
- restart_service(service_name)  # api-service | db-service
- scale_service(service_name)
- flush_cache()

Respond ONLY in JSON:
{{
  "hypothesis": "...",
  "why": "...",
  "action_type": "tool_call|system_action",
  "tool": "...",
  "params": {{...}}
}}"""


# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────

def _sep(char='─', width=52):
    print(f"{C_MUTED}{char * width}{R}")

def _status_color(status):
    s = str(status).upper()
    if s == 'RUNNING':                    return C_GREEN
    if s in ('DOWN', 'OVERLOADED'):       return C_RED
    return C_GOLD

def _svc_val(svc, key, fallback='?'):
    """Safely read a value from a service entry that may be a dict or object."""
    if svc is None:
        return fallback
    if isinstance(svc, dict):
        return svc.get(key, fallback)
    return getattr(svc, key, fallback)

def obs_to_snap(obs):
    """
    Convert an obs object (dataclass / namedtuple / dict) to a plain
    {'services': {name: {...}}, 'latency': int} dict safe for printing.
    """
    if obs is None:
        return None

    # -- pull services --
    raw_services = getattr(obs, 'services', None)
    if raw_services is None and isinstance(obs, dict):
        raw_services = obs.get('services', {})

    services = {}
    if raw_services:
        for name, svc in (raw_services.items() if isinstance(raw_services, dict) else vars(raw_services).items()):
            services[name] = {
                'status':          _svc_val(svc, 'status', 'unknown'),
                'cpu':             _svc_val(svc, 'cpu', None),
                'memory':          _svc_val(svc, 'memory', None),
                'connections':     _svc_val(svc, 'connections', None),
                'max_connections': _svc_val(svc, 'max_connections', None),
            }

    # -- pull latency --
    latency = getattr(obs, 'latency', None)
    if latency is None and isinstance(obs, dict):
        latency = obs.get('latency', '?')

    return {'services': services, 'latency': latency}

def format_state_summary(services):
    """Return a colored one-line service status string."""
    if not services:
        return "?"
    parts = []
    for name, info in services.items():
        status = str(info.get('status', 'unknown')).upper()
        c = _status_color(status)
        parts.append(f"{c}{name.upper()}: {status}{R}")
    return " | ".join(parts)


def print_clean_step(step_num, tool, params, result, reward, total_reward,
                     state=None, done=False):
    """
    Print a single step card in the canonical format:

        [STEP N]
        ────────
        STATE:   ...
        SIGNAL:  ...
        ACTION:  ...
        RESULT:  ...
        REWARD:  ...
        ────────
    """
    print(f"\n{C_CYAN}[STEP {step_num}]{R}")
    _sep()

    if state:
        services = state.get("services", {})
        latency  = state.get("latency", "?")
        svc_str  = format_state_summary(services)
        print(f"{C_CYAN}STATE:{R}")
        print(f"  {svc_str} | {C_MUTED}LATENCY: {latency}ms{R}")

        # SIGNAL block — always print, show ? when data missing
        db  = services.get("db-service",    {})
        api = services.get("api-service",   {})
        signals = []

        db_cpu  = db.get('cpu')
        db_conn = db.get('connections')
        db_max  = db.get('max_connections')
        conn_str = f"{db_conn}/{db_max}" if db_conn is not None or db_max is not None else "?/?"
        signals.append(f"DB CPU: {db_cpu if db_cpu is not None else '?'}%  Connections: {conn_str}")

        api_cpu = api.get('cpu')
        signals.append(f"API CPU: {api_cpu if api_cpu is not None else '?'}%")

        print(f"{C_CYAN}SIGNAL:{R}")
        for s in signals:
            print(f"  {C_MUTED}{s}{R}")

    # ACTION
    param_str = ""
    if params:
        param_str = "  " + "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"{C_CYAN}ACTION:{R}")
    print(f"  {C_TEXT}{tool}{R}{C_MUTED}{param_str}{R}")

    # RESULT
    print(f"{C_CYAN}RESULT:{R}")
    print(f"  {result}")

    # REWARD
    r_color = C_GREEN if reward >= 0 else C_RED
    print(f"{C_CYAN}REWARD:{R}")
    print(f"  {r_color}{reward:+.3f}{R}  {C_MUTED}(Total: {total_reward:+.3f}){R}")

    if done:
        print(f"\n  {C_GREEN}✅ INCIDENT RESOLVED{R}")

    _sep()


def print_episode_summary(ep, scenario_name, info, key_actions, source_majority):
    """Clean episode summary block."""
    success = info.get("success", False)
    color   = C_GREEN if success else C_RED
    icon    = "✔" if success else "✘"
    status  = "SUCCESS" if success else "FAILED"
    if success and info.get("steps", 99) <= 4:
        status = "SUCCESS (Optimal)"

    _sep('═', 52)
    print(f"{color}  Episode {ep}  ·  {status}{R}")
    _sep('═', 52)
    print(f"  {color}{icon} Scenario:{R}    {scenario_name}")
    print(f"  {color}{icon} Steps:{R}       {info.get('steps', '?')}")
    print(f"  {color}{icon} Reward:{R}      {info.get('total_reward', 0):+.3f}")
    print(f"  {color}{icon} Source:{R}      {source_majority}")

    if key_actions:
        chain = f"{C_MUTED} → {R}".join(f"{C_TEXT}{a}{R}" for a in key_actions)
        print(f"  {color}{icon} Fix Chain:{R}   {chain}")

    print()


def log_step_metrics(step, reward, total_reward, source):
    """Minimal inline metric log (used when SHOW_LOGS=False)."""
    src_color = C_MAG if source == "FALLBACK" else C_CYAN
    print(f"  {C_MUTED}Step {step}{R}  {src_color}[{source}]{R}  "
          f"{C_GREEN if reward >= 0 else C_RED}{reward:+.3f}{R}  "
          f"{C_MUTED}Total: {total_reward:.3f}{R}")


# ── AGENT IMPLEMENTATIONS ─────────────────────────────────────────────────────

def action_to_string(action: Action) -> str:
    tool   = action.tool or "unknown"
    params = f" {action.params}" if action.params else ""
    return f"{tool}{params}"


def random_policy():
    choice = random.choice([
        Action(action_type="system_action", tool="restart_service", params={"service": "api-service"}),
        Action(action_type="system_action", tool="scale_service",   params={"service": "db-service"}),
        Action(action_type="system_action", tool="flush_cache",     params={}),
        Action(action_type="tool_call",     tool="get_network_latency", params={}),
        Action(action_type="tool_call",     tool="get_db_metrics",      params={}),
    ])
    return choice, action_to_string(choice), "RANDOM"


def naive_baseline_agent(obs, step_count):
    """Baseline: exhaustive investigation + brute-force all fixes."""
    playbook = [
        Action(action_type="tool_call",     tool="get_db_metrics",        params={}),
        Action(action_type="tool_call",     tool="get_cache_status",       params={}),
        Action(action_type="tool_call",     tool="get_network_latency",    params={}),
        Action(action_type="tool_call",     tool="get_error_logs",         params={}),
        Action(action_type="tool_call",     tool="clear_db_connections",   params={}),
        Action(action_type="system_action", tool="flush_cache",            params={}),
        Action(action_type="system_action", tool="restart_service",        params={"service": "api-service"}),
        Action(action_type="system_action", tool="restart_service",        params={"service": "db-service"}),
        Action(action_type="system_action", tool="scale_service",          params={"service": "db-service"}),
    ]
    idx    = min(step_count, len(playbook) - 1)
    action = playbook[idx]
    return action, action_to_string(action), "BASELINE"


def fallback_policy(obs, action_history):
    """Deterministic fallback used when LLM fails or repeats queries."""
    history_str = " ".join(action_history)
    logs_str    = " ".join(obs.logs)
    db_svc      = obs.services.get("db-service", {})
    api_svc     = obs.services.get("api-service", {})

    if "get_db_metrics" not in history_str:
        return Action(action_type="tool_call", tool="get_db_metrics", params={})

    if "exhaust" in logs_str or "overload" in logs_str or db_svc.get("latency", 0) > 1000:
        if "clear_db_connections" not in history_str:
            return Action(action_type="tool_call", tool="clear_db_connections", params={})
        if "restart_service" not in history_str:
            return Action(action_type="system_action", tool="restart_service", params={"service": "db-service"})

    if "network" in logs_str or "Traffic" in logs_str or "timeout" in logs_str.lower():
        if "get_network_latency" not in history_str:
            return Action(action_type="tool_call", tool="get_network_latency", params={})

    if db_svc.get("status") in ("running", "unknown") and api_svc.get("status") == "down":
        if "api-service" not in history_str:
            return Action(action_type="system_action", tool="restart_service", params={"service": "api-service"})

    if "Deadlock" in logs_str or obs.latency >= 1500:
        if "flush_cache" not in history_str:
            return Action(action_type="system_action", tool="flush_cache", params={})

    for q in ["get_db_metrics", "get_network_latency", "get_error_logs"]:
        if q not in history_str:
            return Action(action_type="tool_call", tool=q, params={})

    if "api-service" not in history_str:
        return Action(action_type="system_action", tool="restart_service", params={"service": "api-service"})
    elif "flush_cache" not in history_str:
        return Action(action_type="system_action", tool="flush_cache", params={})
    else:
        return Action(action_type="system_action", tool="scale_service", params={"service": "db-service"})


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def call_llm(prompt):
    global current_token_idx, client

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]

    if USE_CACHE and prompt_hash in prompt_cache:
        print(f"{C_MUTED}[CACHE HIT] {prompt_hash}{R}")
        return prompt_cache[prompt_hash]

    attempts   = 0
    last_error = None

    while attempts < 3:
        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0,
                top_p=1,
            )
            raw  = response.choices[0].message.content.strip()
            raw  = _extract_json(raw)
            if not raw:
                raise ValueError("Empty response from LLM")

            data = json_repair.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"LLM returned non-dict: {type(data)}")

            action_type = data.get("action_type", "").strip()
            tool        = data.get("tool", "").strip()
            params      = data.get("params", {})

            VALID_TYPES = {"tool_call", "system_action"}
            VALID_TOOLS = {
                "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
                "clear_db_connections", "restart_service", "scale_service", "flush_cache",
            }

            if not action_type or action_type not in VALID_TYPES:
                raise ValueError(f"Invalid action_type: '{action_type}'")
            if not tool or tool not in VALID_TOOLS:
                raise ValueError(f"Invalid tool: '{tool}'")

            action = Action(action_type=action_type, tool=tool, params=params)

            if USE_CACHE:
                prompt_cache[prompt_hash] = action
            return action

        except Exception as e:
            err_msg    = str(e)
            last_error = e
            if "402" in err_msg or "429" in err_msg:
                attempts          += 1
                current_token_idx  = (current_token_idx + 1) % len(HF_TOKENS)
                print(f"{C_GOLD}[TOKEN SWITCH] → index {current_token_idx}{R}")
                client = InferenceClient(
                    base_url=os.getenv("API_BASE_URL"),
                    token=HF_TOKENS[current_token_idx],
                )
                continue
            else:
                if attempts < 3 and ("json" in err_msg.lower() or "Expecting" in err_msg or "non-dict" in err_msg):
                    attempts += 1
                    print(f"{C_GOLD}[JSON RETRY {attempts}/3] {e}{R}")
                    continue
                raise e

    print(f"{C_RED}[FATAL] All tokens exhausted{R}")
    return None


def llm_agent(obs, action_history, memory, episode_memory):
    """Intelligent agent with escalation, loop detection, and structured reasoning."""
    history_str  = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(action_history)) or "  none yet"
    mem_str      = memory if memory else "No past episodes. First run."
    queries_str  = ", ".join(episode_memory["queries_made"]) if episode_memory["queries_made"] else "none yet"
    last_action  = episode_memory["last_action"] or "none yet"
    current_step = episode_memory["step"]
    num_queries  = len(episode_memory["queries_made"])

    escalation   = ""
    if current_step >= 3 and num_queries >= 1:
        escalation = "\n\n🚨 ESCALATION: Enough queries. Apply a corrective fix NOW."

    loop_warning = ""
    if len(episode_memory["actions_taken"]) >= 2:
        if episode_memory["actions_taken"][-1] == episode_memory["actions_taken"][-2]:
            loop_warning = "\n\n⚠️ LOOP DETECTED: Last 2 actions were IDENTICAL. Choose a completely different strategy."

    prompt = DIAGNOSIS_PROMPT.format(
        services=json.dumps(obs.services, indent=2),
        logs="\n".join(obs.logs[-10:]),
        latency=obs.latency,
        history=history_str,
        memory=mem_str,
        queries_made=queries_str,
        last_action=last_action,
        step=current_step,
        max_steps=MAX_STEPS,
    )
    if escalation:   prompt += escalation
    if loop_warning: prompt += loop_warning

    try:
        action = call_llm(prompt)
        if action is None:
            raise ValueError("LLM returned None (all tokens exhausted)")
        source = "LLM"
    except Exception as e:
        print(f"{C_RED}[LLM ERROR] {e}{R}")
        action = Action(action_type="tool_call", tool="get_error_logs", params={})
        source = "LLM_ERROR"

    return action, action_to_string(action), source


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def run_loop():
    print(f"\n{C_TEXT}{'═'*52}{R}")
    print(f"{C_CYAN}  ACRS EVALUATION RUN{R}")
    print(f"  Episodes: {NUM_EPISODES}   Mode: {MODE.upper()}   Steps: {MAX_STEPS}")
    print(f"{C_TEXT}{'═'*52}{R}\n")

    # ── PHASE 1: BASELINE ─────────────────────────────────────────────────────
    print(f"{C_GOLD}── PHASE 1: NAIVE BASELINE AGENT {'─'*19}{R}")
    env = AutoSREEnv(difficulty="training")
    obs = env.reset()
    info = {"total_reward": 0.0, "steps": 0, "success": False}
    baseline_actions = []

    for i in range(MAX_STEPS):
        action, action_str, source = naive_baseline_agent(obs, i)
        baseline_actions.append(action_str)
        obs, reward, done, info = env.step(action)

        if SHOW_LOGS:
            result_str = f"reward {reward:+.3f}"
            print_clean_step(
                step_num=i + 1,
                tool=action.tool,
                params=action.params,
                result=result_str,
                reward=reward,
                total_reward=info["total_reward"],
                state=obs_to_snap(obs),
                done=done,
            )
        else:
            log_step_metrics(i + 1, reward, info["total_reward"], source)

        if done:
            break

    print_episode_summary("Baseline", "NaiveAgent", info, baseline_actions[:4], "BASELINE")

    # ── PHASE 2: INTELLIGENT AGENT ────────────────────────────────────────────
    print(f"{C_GOLD}── PHASE 2: {MODE.upper()} AGENT {'─'*30}{R}\n")

    cross_episode_memory = ""
    episodes_x, rewards, steps_list, success_list = [], [], [], []

    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n{C_CYAN}>>> EPISODE {ep}{R}")

        env  = AutoSREEnv(difficulty="training")
        obs  = env.reset()
        scenario_name = env.state.get("name", "Unknown")
        print(f"  {C_MUTED}Scenario: {C_TEXT}{scenario_name}{R}")

        action_history = []
        source_counts  = {}
        total_intercept_penalty = 0.0
        key_actions    = []  # track fix-chain actions for summary

        episode_memory = {
            "queries_made":  set(),
            "actions_taken": [],
            "last_action":   None,
            "step":          0,
        }

        for step in range(MAX_STEPS):
            if MODE == "random":
                action, action_str, source = random_policy()
            elif MODE == "llm":
                action, action_str, source = llm_agent(obs, action_history, cross_episode_memory, episode_memory)
            else:
                raise ValueError(f"Unknown MODE: {MODE}")

            # ── ACTION FILTERING ──────────────────────────────────────────────
            if MODE == "llm" and source == "LLM":
                if action_str in episode_memory["actions_taken"]:
                    print(f"{C_GOLD}[WARN] Repeated action '{action_str}'{R}")
                    total_intercept_penalty -= 0.2
                elif action.action_type == "tool_call":
                    if action.tool in episode_memory["queries_made"]:
                        print(f"{C_GOLD}[WARN] Repeated query '{action.tool}'{R}")
                        total_intercept_penalty -= 0.2

            action_history.append(action_str)
            source_counts[source] = source_counts.get(source, 0) + 1

            obs, reward, done, info = env.step(action)

            episode_memory["actions_taken"].append(action_str)
            if action.action_type == "tool_call":
                episode_memory["queries_made"].add(action.tool)
            episode_memory["last_action"] = action_str
            episode_memory["step"]       += 1

            # Track non-query actions for the fix chain summary
            if action.action_type == "system_action":
                key_actions.append(action.tool)

            # Apply intercept penalty
            if total_intercept_penalty < 0:
                reward              += total_intercept_penalty
                info["total_reward"] += total_intercept_penalty
                total_intercept_penalty = 0.0

            if SHOW_LOGS:
                result_str = info.get("result", f"reward {reward:+.3f}")
                print_clean_step(
                    step_num=step + 1,
                    tool=action.tool,
                    params=action.params,
                    result=result_str,
                    reward=reward,
                    total_reward=info["total_reward"],
                    state=obs_to_snap(obs),
                    done=done,
                )
            else:
                log_step_metrics(step + 1, reward, info["total_reward"], source)

            if done:
                break

        majority_source = max(source_counts, key=source_counts.get) if source_counts else "UNKNOWN"
        print_episode_summary(ep, scenario_name, info, key_actions, majority_source)

        # Build cross-episode memory
        mistakes = []
        if info.get('repeated_queries', 0) > 0: mistakes.append(f"{info['repeated_queries']} repeated queries")
        if info.get('blind_actions', 0)    > 0: mistakes.append(f"{info['blind_actions']} blind actions")
        mistakes_str = ", ".join(mistakes) if mistakes else "none"

        cross_episode_memory += (
            f"Episode {ep}:\nReward: {info['total_reward']:.3f}\n"
            f"Steps: {info['steps']}\nSuccess: {info['success']}\n"
            f"Mistakes: {mistakes_str}\n\n"
        )
        blocks = cross_episode_memory.strip().split("\n\n")
        if len(blocks) > 5:
            cross_episode_memory = "\n\n".join(blocks[-5:]) + "\n\n"

        episodes_x.append(ep)
        rewards.append(info.get("total_reward", 0))
        steps_list.append(info.get("steps", 0))
        success_list.append(1 if info.get("success") else 0)

    # ── FINAL METRICS ─────────────────────────────────────────────────────────
    print(f"\n{C_TEXT}{'═'*52}{R}")
    print(f"{C_CYAN}  EVALUATION COMPLETE{R}")
    print(f"{C_TEXT}{'═'*52}{R}")
    print(f"  Avg Reward:    {sum(rewards)/len(rewards):.3f}")
    print(f"  Avg Steps:     {sum(steps_list)/len(steps_list):.2f}")
    print(f"  Success Rate:  {sum(success_list)/len(success_list)*100:.1f}%")

    if len(steps_list) > 1:
        first, last = steps_list[0], steps_list[-1]
        if first > 0:
            reduction = ((first - last) / first) * 100
            color = C_GREEN if reduction > 0 else C_RED
            print(f"  Step Reduction: {color}{reduction:.0f}%{R}")

    print()

    # ── GRAPHS ────────────────────────────────────────────────────────────────
    try:
        plt.figure()
        plt.plot(rewards, marker='o', color='#00dcff', linewidth=2)
        plt.title(f"{MODE.upper()} — Reward vs Episode", fontsize=13)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{MODE}_reward_curve.png", dpi=120)
        try:
            display(Image(f"{MODE}_reward_curve.png"))
            plt.close()
        except NameError:
            plt.show()

        plt.figure()
        plt.plot(steps_list, marker='o', color='#ffd200', linewidth=2)
        plt.title(f"{MODE.upper()} — Steps vs Episode", fontsize=13)
        plt.xlabel("Episode")
        plt.ylabel("Steps Taken")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{MODE}_steps_curve.png", dpi=120)
        try:
            display(Image(f"{MODE}_steps_curve.png"))
            plt.close()
        except NameError:
            plt.show()

        print(f"  {C_MUTED}Saved: {MODE}_reward_curve.png  {MODE}_steps_curve.png{R}")

    except Exception as e:
        print(f"{C_RED}[WARN] Graph generation failed: {e}{R}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        print(f"\n{C_CYAN}>>> 5-SECOND WOW DEMO MODE <<<{R}\n")
        print(f"{C_RED}Baseline     → FAIL ✘{R}")
        print(f"{C_GOLD}Episode 1    → Inefficient ⚠{R}")
        print(f"{C_GREEN}Episode 5    → Optimal ✅{R}\n")
    else:
        run_loop()
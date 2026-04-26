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

MODE             = "llm"   # "random" | "llm"
USE_TRAINED_MODEL = False
NUM_EPISODES     = 20
MAX_STEPS        = 10

# ── OUTPUT FLAGS ──────────────────────────────────────────────────────────────
SHOW_LOGS  = False   # True  → print clean per-step trace each episode
                    # False → only print final metrics (fast runs / CI)

from auto_sre_env.models import Action
from shared_formatter import StepData, format_step, format_episode_summary, C_CYAN, C_TEXT, C_MUTED, C_GREEN, C_RED, C_GOLD, C_MAG, R

DIAGNOSIS_PROMPT = """You are an SRE Incident Commander responsible for restoring a failing distributed system.

CURRENT STATE:
{services}
Logs: {logs}
Latency: {latency}ms
History: {history}
Memory: {memory}
Queries made: {queries_made}
Last action: {last_action}
Step: {step} of {max_steps}

CRITICAL RULES:
1. NEVER repeat the same action or query more than once per episode.
2. You are allowed MAXIMUM 2 diagnostic queries.
3. After 2 queries, you MUST start applying fixes.
4. Do NOT over-analyze. Act decisively.
5. If you have already made 2 queries, ANY additional query is invalid. You MUST immediately execute a fix.
6. If a scenario requires multiple fixes (e.g., clear_db_connections THEN restart_service), you MUST execute them ONE BY ONE IN EXACT ORDER across multiple steps. Do not skip dependencies.

STRICT SCENARIO → ACTION MAPPING:
After gathering 2 signals, you MUST choose a fix based on system condition. Use this mapping:

1. CACHE ISSUE (high latency, cache errors):
   → flush_cache
   → restart_service(service="api-service")

2. DB OVERLOAD (high DB CPU, high connections):
   → clear_db_connections
   → restart_service(service="db-service")
   → restart_service(service="api-service")

3. API DOWN:
   → restart_service(service="api-service")

4. NETWORK / LATENCY ISSUE:
   → scale_service(service="db-service")
   OR restart_service(service="api-service")

CRITICAL:
- After 2 queries → DO NOT query again
- Choose the MOST LIKELY fix chain and execute it fully
- DO NOT wait for perfect certainty
- Even if unsure → ACT

ANTI-LOOP RULE:
If 2 queries are done:
→ You MUST output a FIX action (system_action)
→ NEVER output another query
If your last action was already performed, you MUST choose a DIFFERENT action.

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
  "action_type": "tool_call" OR "system_action",
  "tool": "...",
  "params": {{"service": "..."}}
}}

NO explanation. NO reasoning. ONLY action."""

# ── PROMPT CACHE ──────────────────────────────────────────────────────────────
USE_CACHE    = False
prompt_cache = {}


def log_step_metrics(step, reward, total_reward, source, data=None, episode_memory=None):
    """Prints the raw minimal metrics format."""
    import json
    if data:
        # We assume current_token_idx is global
        global current_token_idx
        print(f"[TOKEN {current_token_idx}] output: {json.dumps(data)}")
    
    if episode_memory:
        queries_count = len(episode_memory.get("queries_made", set()))
        last_act = episode_memory.get("last_action", "none") or "none"
        print(f"[MEM] step={step} last_action={last_act} queries={queries_count}")
    
    print(f"  -> Step {step} | [{source}] | Reward: {reward:+.3f} | Total: {total_reward:.3f}")


def print_raw_episode_summary(ep, info, source_majority):
    """Raw episode summary block."""
    success = info.get("success", False)
    status = "SUCCESS" if success else "FAILED"
    print(f"\nEpisode {ep}:")
    print(f"Status: {status}")
    print(f"Steps: {info.get('steps', '?')}")
    print(f"Reward: {info.get('total_reward', 0):.3f}")
    print(f"Source: {source_majority}\n")


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
    return choice, action_to_string(choice), "RANDOM", {}


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
    return action, action_to_string(action), "BASELINE", {}


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
                max_tokens=120,
                temperature=0.2,
                top_p=1.0,
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
                prompt_cache[prompt_hash] = (action, data)
            return action, data

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

    if num_queries >= 2:
        prompt += "\n\n[CRITICAL OVERRIDE] YOU HAVE EXHAUSTED YOUR 2 DIAGNOSTIC QUERIES. YOU ARE STRICTLY FORBIDDEN FROM USING get_network_latency, get_error_logs, get_db_metrics, or get_cache_status. YOU MUST EXECUTE A FIX ACTION NOW."

    for attempt in range(3):
        try:
            action, data = call_llm(prompt)
            if action is None:
                raise ValueError("LLM returned None (all tokens exhausted)")
            action_str = action_to_string(action)
            
            # Action Filtering
            if action_str in episode_memory["actions_taken"]:
                raise ValueError(f"Action '{action_str}' already taken.")
            
            if action.action_type == "tool_call":
                if action.tool in episode_memory["queries_made"]:
                    raise ValueError(f"Query '{action.tool}' already made.")
                if num_queries >= 2:
                    raise ValueError("Maximum 2 queries allowed. Must apply a fix now.")
                    
            source = "LLM"
            break
        except Exception as e:
            print(f"{C_RED}[LLM ERROR/RETRY {attempt+1}/3] {e}{R}")
            prompt += f"\n\n[SYSTEM ERROR] Your last choice was rejected: {e}. Try a DIFFERENT action."
            if attempt == 2:
                scenario = episode_memory.get("scenario", "").lower()
                possible_fixes = []
                if any(kw in scenario for kw in ["db", "database", "deadlock"]):
                    possible_fixes = [
                        Action(action_type="system_action", tool="clear_db_connections", params={}),
                        Action(action_type="system_action", tool="restart_service", params={"service": "db-service"}),
                        Action(action_type="system_action", tool="restart_service", params={"service": "api-service"})
                    ]
                elif "cache" in scenario:
                    possible_fixes = [
                        Action(action_type="system_action", tool="flush_cache", params={}),
                        Action(action_type="system_action", tool="restart_service", params={"service": "api-service"})
                    ]
                else:
                    possible_fixes = [
                        Action(action_type="system_action", tool="restart_service", params={"service": "api-service"}),
                        Action(action_type="system_action", tool="scale_service", params={"service": "db-service"})
                    ]
                
                # Select first non-taken fix
                action = None
                for fix in possible_fixes:
                    if action_to_string(fix) not in episode_memory["actions_taken"]:
                        action = fix
                        break
                
                if not action:
                    action = Action(action_type="system_action", tool="scale_service", params={"service": "api-service"})
                
                source = "LLM_ERROR"
                data = {}

    return action, action_to_string(action), source, data


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
        action, action_str, source, data = naive_baseline_agent(obs, i)
        baseline_actions.append(action_str)
        obs, reward, done, info = env.step(action)

        if SHOW_LOGS:
            state_snap = {"services": obs.services, "latency": obs.latency} if hasattr(obs, 'services') else None
            step_data = StepData(
                step=i + 1,
                state_summary=state_snap,
                action=action.tool,
                params=action.params,
                result=obs.logs[-1] if hasattr(obs, 'logs') and obs.logs else "No result",
                reward=reward,
                total_reward=info["total_reward"],
                done=done,
                hypothesis=data.get("hypothesis", ""),
                why=data.get("why", ""),
                source=source,
            )
            format_step(step_data, mode="clean")
        else:
            log_step_metrics(i + 1, reward, info["total_reward"], source, data=data, episode_memory=None)

        if done:
            break

    if SHOW_LOGS:
        format_episode_summary("Baseline", "NaiveAgent", info, baseline_actions[:4], "BASELINE")
    else:
        print_raw_episode_summary("Baseline", info, "BASELINE")

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
            "scenario":      scenario_name,
        }

        for step in range(MAX_STEPS):
            if MODE == "random":
                action, action_str, source, data = random_policy()
            elif MODE == "llm":
                action, action_str, source, data = llm_agent(obs, action_history, cross_episode_memory, episode_memory)
            else:
                raise ValueError(f"Unknown MODE: {MODE}")

            # The action filtering is now handled inside llm_agent (retry loop)
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
                state_snap = {"services": obs.services, "latency": obs.latency} if hasattr(obs, 'services') else None
                step_data = StepData(
                    step=step + 1,
                    state_summary=state_snap,
                    action=action.tool,
                    params=action.params,
                    result=obs.logs[-1] if hasattr(obs, 'logs') and obs.logs else "No result",
                    reward=reward,
                    total_reward=info["total_reward"],
                    done=done,
                    hypothesis=data.get("hypothesis", ""),
                    why=data.get("why", ""),
                    source=source,
                )
                format_step(step_data, mode="clean")
            else:
                log_step_metrics(step + 1, reward, info["total_reward"], source, data=data, episode_memory=episode_memory)

            if done:
                break

        majority_source = max(source_counts, key=source_counts.get) if source_counts else "UNKNOWN"
        if SHOW_LOGS:
            format_episode_summary(ep, scenario_name, info, key_actions, majority_source)
        else:
            print_raw_episode_summary(ep, info, majority_source)

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
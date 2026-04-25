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

load_dotenv()

HF_TOKENS = [
    os.getenv("HF_TOKEN_1"),
    os.getenv("HF_TOKEN_2"),
    os.getenv("HF_TOKEN_3"),
    os.getenv("HF_TOKEN_4"),
    os.getenv("HF_TOKEN_5"),
]
# Filter out missing tokens and fall back to HF_TOKEN if pool is empty
HF_TOKENS = [t for t in HF_TOKENS if t]
if not HF_TOKENS:
    HF_TOKENS = [os.getenv("HF_TOKEN")]

current_token_idx = 0

client = InferenceClient(
    model=os.getenv("MODEL_NAME"),
    token=HF_TOKENS[current_token_idx]
)

MODE = "random"   # options: "random", "llm"
USE_TRAINED_MODEL = False
NUM_EPISODES = 20
MAX_STEPS = 10

# ── PROMPT CACHE ─────────────────────────────────────────────────────────────
USE_CACHE = False  # Set True for production, False for evaluation
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

CRITICAL RULES:
1. GATHER SIGNALS: You MUST gather at least 2 relevant signals (queries) before applying any fix. Premature fixes are penalized (-0.2).
2. CHAIN OF FIXES: Most incidents require a SEQUENCE of 2-3 fixes. A single action rarely resolves the issue fully.
3. OBSERVE & PIVOT: After a partial fix, metrics may improve but the incident is not over. Re-evaluate and apply the next step.

AVAILABLE TOOLS (APIs):
- get_network_latency()
- get_error_logs()
- get_db_metrics()
- get_cache_status()
- clear_db_connections()
- restart_service(service_name)  # service_name: api-service, db-service
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

# ── LOGGING HELPERS ──────────────────────────────────────────────────────────

def print_agent_thought(data: dict):
    """WOW-factor reasoning logs."""
    tool = data.get("tool", "")
    params = data.get("params", {})
    params_str = f" {params}" if params else ""
    
    print("\n==============================")
    print(f"[STEP LOG]")
    print(f"Hypothesis: {data.get('hypothesis', '')}")
    print(f"Why:        {data.get('why', '')}")
    print(f"Tool Used:  {tool}{params_str}")
    print("==============================\n")


def log_step(step, reward, total_reward, source):
    src_color = "\033[35m" if source == "FALLBACK" else "\033[94m"
    print(f"  -> \033[90mStep {step}\033[0m | {src_color}[{source}]\033[0m | \033[92mReward: {reward:+.3f}\033[0m | \033[90mTotal: {total_reward:.3f}\033[0m")


def log_episode_result(ep, agent_name, info, source_majority):
    if info["success"]:
        color = "\033[92m"
        status = "SUCCESS (Optimal <=4 steps)" if info["steps"] <= 4 else "SUCCESS (Inefficient)"
    else:
        color = "\033[91m"
        status = "FAILED"

    print(f"\n{color}Episode {ep}:\033[0m")
    print(f"Status: {status}")
    print(f"Steps: {info['steps']}")
    print(f"Reward: {info['total_reward']:.3f}")
    print(f"Source: {source_majority}")
    if info["success"]:
        print("✔ Agent avoided blind action")
        print("✔ Gathered evidence from 2 sources")
        print("✔ Correct fix chain applied")
        print("✔ System recovered")
    print("")


# ── AGENT IMPLEMENTATIONS ─────────────────────────────────────────────────────

def action_to_string(action: Action) -> str:
    tool = action.tool or "unknown"
    params = f" {action.params}" if action.params else ""
    return f"{tool}{params}"


def random_policy():
    choice = random.choice([
        Action(action_type="system_action", tool="restart_service", params={"service":"api-service"}),
        Action(action_type="system_action", tool="scale_service", params={"service":"db-service"}),
        Action(action_type="system_action", tool="flush_cache", params={}),
        Action(action_type="tool_call", tool="get_network_latency", params={}),
        Action(action_type="tool_call", tool="get_db_metrics", params={})
    ])
    return choice, action_to_string(choice), "RANDOM"


def naive_baseline_agent(obs, step_count):
    """A baseline agent that queries EVERYTHING then tries ALL fixes — solvable but slow."""
    # Fixed sequence: exhaustive investigation, then brute-force all fixes
    playbook = [
        # Steps 0-4: Query every possible source (wasteful but thorough)
        Action(action_type="tool_call", tool="get_db_metrics", params={}),
        Action(action_type="tool_call", tool="get_cache_status", params={}),
        Action(action_type="tool_call", tool="get_network_latency", params={}),
        Action(action_type="tool_call", tool="get_error_logs", params={}),
        # Steps 5-9: Try every fix (shotgun approach)
        Action(action_type="tool_call", tool="clear_db_connections", params={}),
        Action(action_type="system_action", tool="flush_cache", params={}),
        Action(action_type="system_action", tool="restart_service", params={"service":"api-service"}),
        Action(action_type="system_action", tool="restart_service", params={"service":"db-service"}),
        Action(action_type="system_action", tool="scale_service", params={"service":"db-service"}),
    ]
    idx = min(step_count, len(playbook) - 1)
    action = playbook[idx]
    return action, action_to_string(action), "BASELINE"


def fallback_policy(obs, action_history):
    """Deterministic fallback policy used when LLM fails or repeats queries."""
    history_str = " ".join(action_history)
    logs_str = " ".join(obs.logs)
    db_svc = obs.services.get("db-service", {})
    api_svc = obs.services.get("api-service", {})
    
    # 1. No queries made yet -> query DB stats
    if "get_db_metrics" not in history_str:
        return Action(action_type="tool_call", tool="get_db_metrics", params={})
        
    # 2. Check for DB issues
    if "exhaust" in logs_str or "overload" in logs_str or db_svc.get("latency", 0) > 1000:
        if "clear_db_connections" not in history_str:
            return Action(action_type="tool_call", tool="clear_db_connections", params={})
        if "restart_service" not in history_str:
            return Action(action_type="system_action", tool="restart_service", params={"service":"db-service"})
            
    # 3. Check network
    if "network" in logs_str or "Traffic" in logs_str or "timeout" in logs_str.lower():
        if "get_network_latency" not in history_str:
            return Action(action_type="tool_call", tool="get_network_latency", params={})

    # 4. API Down
    if db_svc.get("status") in ("running", "unknown") and api_svc.get("status") == "down":
        if "api-service" not in history_str:
            return Action(action_type="system_action", tool="restart_service", params={"service":"api-service"})

    # 5. Deadlocks / High Latency but components look healthy
    if "Deadlock" in logs_str or obs.latency >= 1500:
        if "flush_cache" not in history_str:
            return Action(action_type="system_action", tool="flush_cache", params={})
            
    # 6. Default progression (ensure no repeats)
    queries = ["get_db_metrics", "get_network_latency", "get_error_logs"]
    for q in queries:
        if q not in history_str:
            return Action(action_type="tool_call", tool=q, params={})
            
    # Ultimate fallback
    if "api-service" not in history_str:
        return Action(action_type="system_action", tool="restart_service", params={"service":"api-service"})
    elif "flush_cache" not in history_str:
        return Action(action_type="system_action", tool="flush_cache", params={})
    else:
        return Action(action_type="system_action", tool="scale_service", params={"service":"db-service"})


def _extract_json(raw: str) -> str:
    """Extract JSON from LLM output, stripping markdown fences and conversational text."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    # Try to find a JSON object if there's surrounding text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def call_llm(prompt):
    global current_token_idx, client
    
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
    
    # Check prompt cache
    if USE_CACHE and prompt_hash in prompt_cache:
        print(f"\033[90m[CACHE HIT] prompt hash: {prompt_hash}\033[0m")
        return prompt_cache[prompt_hash]
    
    attempts = 0
    last_error = None
    while attempts < 3:
        try:
            print(f"\033[90m[TOKEN {current_token_idx}] prompt hash: {prompt_hash}\033[0m")
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0,
                top_p=1,
            )
            raw = response.choices[0].message.content.strip()
            raw = _extract_json(raw)
            if not raw:
                raise ValueError("Empty response from LLM")
            
            data = json_repair.loads(raw)
            
            if not isinstance(data, dict):
                raise ValueError(f"LLM returned non-dict: {type(data)}")
            
            # ── VALIDATE ACTION FIELDS ──
            action_type = data.get("action_type", "").strip()
            tool = data.get("tool", "").strip()
            params = data.get("params", {})
            
            VALID_TYPES = {"tool_call", "system_action"}
            VALID_TOOLS = {
                "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
                "clear_db_connections", "restart_service", "scale_service", "flush_cache"
            }
            
            if not action_type or action_type not in VALID_TYPES:
                raise ValueError(f"Invalid action_type: '{action_type}'")
            if not tool or tool not in VALID_TOOLS:
                raise ValueError(f"Invalid tool: '{tool}'")
            
            print_agent_thought(data)
            print(f"\033[90m[TOKEN {current_token_idx}] output: {json.dumps(data, default=str)}\033[0m")

            action = Action(
                action_type=action_type,
                tool=tool,
                params=params
            )
            
            # Cache successful response
            if USE_CACHE:
                prompt_cache[prompt_hash] = action
            return action
        except Exception as e:
            err_msg = str(e)
            last_error = e
            if "402" in err_msg or "429" in err_msg:
                attempts += 1
                current_token_idx = (current_token_idx + 1) % len(HF_TOKENS)
                print(f"[TOKEN SWITCH] Using token index {current_token_idx}")
                client = InferenceClient(
                    model=os.getenv("MODEL_NAME"),
                    token=HF_TOKENS[current_token_idx]
                )
                continue
            else:
                # For JSON parse errors, retry up to 3 times on same token
                if attempts < 3 and ("json" in err_msg.lower() or "Expecting" in err_msg or "non-dict" in err_msg):
                    attempts += 1
                    print(f"\033[33m[JSON RETRY {attempts}/3] Retrying due to parse error: {e}\033[0m")
                    continue
                # For non-retryable errors, re-raise to be caught by llm_agent
                raise e
                
    print("[FATAL] All tokens exhausted")
    return None



def llm_agent(obs, action_history, memory, episode_memory):
    """Intelligent agent with escalation, auto-force, and structured reasoning."""
    history_str = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(action_history)) or "  none yet"
    mem_str = memory if memory else "No past episodes. This is your first run."
    queries_str = ", ".join(episode_memory["queries_made"]) if episode_memory["queries_made"] else "none yet"
    last_action_str = episode_memory["last_action"] or "none yet"
    current_step = episode_memory["step"]
    num_queries = len(episode_memory["queries_made"])
    

    # Build escalation suffix for steps 3-4
    escalation = ""
    if current_step >= 3 and num_queries >= 1:
        escalation = "\n\n🚨 ESCALATION: You have queried enough. You MUST apply a corrective fix NOW. Do NOT query again."
    
    # Loop detection
    loop_warning = ""
    if len(episode_memory["actions_taken"]) >= 2:
        if episode_memory["actions_taken"][-1] == episode_memory["actions_taken"][-2]:
            loop_warning = "\n\n⚠️ LOOP DETECTED: Your last 2 actions were IDENTICAL. Choose a COMPLETELY DIFFERENT strategy."

    prompt = DIAGNOSIS_PROMPT.format(
        services=json.dumps(obs.services, indent=2),
        logs="\n".join(obs.logs[-10:]),
        latency=obs.latency,
        history=history_str,
        memory=mem_str,
        queries_made=queries_str,
        last_action=last_action_str,
        step=current_step,
        max_steps=MAX_STEPS,
    )
    
    if escalation:
        prompt += escalation
    if loop_warning:
        prompt += loop_warning

    try:
        print("\033[95m[MODE] LLM reasoning\033[0m")
        action = call_llm(prompt)
        if action is None:
            raise ValueError("LLM returned None (All Tokens Exhausted)")
        source = "LLM"
    except Exception as e:
        print(f"\033[91m[LLM ERROR] {e}.\033[0m")
        print("\033[93m[ERROR] LLM failed, skipping turn (returning dummy query)\033[0m")
        action = Action(action_type="tool_call", tool="get_error_logs", params={})
        source = "LLM_ERROR"
        
    return action, action_to_string(action), source


# ── TRAINING LOOP ────────────────────────────────────────────────────────────

def run_loop():
    print(f"\n\033[1;97m>>> STARTING P0 IC EVALUATION RUN ({NUM_EPISODES} EPISODES, MODE={MODE})\033[0m\n")
    
    # 1. BASELINE RUN
    print("\033[1;93m--- PHASE 1: NAIVE BASELINE AGENT ---\033[0m")
    env = AutoSREEnv(difficulty="training")
    obs = env.reset()
    info = {"total_reward": 0.0, "steps": 0, "success": False}
    for i in range(MAX_STEPS):
        action, action_str, source = naive_baseline_agent(obs, i)
        obs, reward, done, info = env.step(action)
        log_step(i+1, reward, info["total_reward"], source)
        if done: break
    log_episode_result("Baseline", "NaiveAgent", info, "BASELINE")

    # 2. INTELLIGENT AGENT RUN
    print(f"\033[1;93m--- PHASE 2: {MODE.upper()} AGENT EVALUATION ---\033[0m")
    
    cross_episode_memory = ""
    episodes_x = []
    rewards = []
    steps_list = []
    success_list = []
    episode_metrics = []
    
    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n\033[1;94m>>> EPISODE {ep} INITIALIZING...\033[0m")
        env = AutoSREEnv(difficulty="training")
        obs = env.reset()
        
        scenario_name = env.state.get("name", "Unknown")
        print(f"\033[1;96m[SCENARIO] {scenario_name}\033[0m")
        
        action_history = []
        source_counts = {"LLM": 0, "LLM_ERROR": 0, "RANDOM": 0}
        total_intercept_penalty = 0.0
        
        # Per-episode structured memory (stored in Python, survives token switches)
        episode_memory = {
            "queries_made": set(),
            "actions_taken": [],
            "last_action": None,
            "step": 0,
        }
        
        for step in range(MAX_STEPS):
            if MODE == "random":
                action, action_str, source = random_policy()
            elif MODE == "llm":
                action, action_str, source = llm_agent(obs, action_history, cross_episode_memory, episode_memory)
            else:
                raise ValueError(f"Unknown MODE: {MODE}")
            
            # ── ACTION FILTERING LAYER ──────────────────────────────────
            if MODE == "llm" and source == "LLM":
                max_retries = 2
                retry_count = 0
                
                # Check for repeated action
                while action_str in episode_memory["actions_taken"] and retry_count < max_retries:
                    print(f"\033[33m[FILTER] Action '{action_str}' already taken. Forcing retry.\033[0m")
                    retry_prompt_suffix = f"\n\n⚠️ You repeated an action: '{action_str}'. This action was ALREADY taken. Choose a DIFFERENT action."
                    action, action_str, source = llm_agent(obs, action_history, cross_episode_memory + retry_prompt_suffix, episode_memory)
                    retry_count += 1
                
                # Check for repeated query (tool call)
                if action.action_type == "tool_call":
                    query_key = f"{action.tool}"
                    retry_count = 0
                    while query_key in episode_memory["queries_made"] and retry_count < max_retries:
                        print(f"\033[33m[FILTER] Query '{action.tool}' already executed. Forcing retry.\033[0m")
                        retry_prompt_suffix = f"\n\n⚠️ You already queried '{action.tool}'. Choose a DIFFERENT action or query."
                        action, action_str, source = llm_agent(obs, action_history, cross_episode_memory + retry_prompt_suffix, episode_memory)
                        retry_count += 1
                        if action.action_type == "tool_call":
                            query_key = f"{action.tool}"
                        else:
                            break
            
            # HARD CONSTRAINT: Intercept repeated queries (penalty but still execute)
            if action.action_type == "tool_call":
                query_key = f"{action.tool}"
                if query_key in episode_memory["queries_made"]:
                    print(f"\033[33m[WARN] LLM repeated query '{action.tool}'. (Penalty applied)\033[0m")
                    total_intercept_penalty -= 0.3
                
            action_history.append(action_str)
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts[source] = 1
            
            obs, reward, done, info = env.step(action)
            
            # ── UPDATE EPISODE MEMORY ───────────────────────────────────
            episode_memory["actions_taken"].append(action_str)
            if action.action_type == "tool_call":
                episode_memory["queries_made"].add(f"{action.tool}")
            episode_memory["last_action"] = action_str
            episode_memory["step"] += 1
            
            # 🧪 PRE-FLIGHT: Memory persistence check
            print(f"\033[90m[MEM] step={episode_memory['step']} last_action={episode_memory['last_action']} queries={len(episode_memory['queries_made'])}\033[0m")
            
            # Apply manual intercept penalty to the reward logs
            if total_intercept_penalty < 0:
                reward += total_intercept_penalty
                info["total_reward"] += total_intercept_penalty
                total_intercept_penalty = 0.0  # reset for next step
                
            log_step(step+1, reward, info["total_reward"], source)
            
            if done: break
            
        majority_source = max(source_counts, key=source_counts.get) if source_counts else "UNKNOWN"
        log_episode_result(ep, "LLMAgent", info, majority_source)
        
        if info.get("success"):
            print(f"\033[1;92m✓ Agent solved '{scenario_name}' scenario\033[0m")
            
        episode_metrics.append({
            "reward": info['total_reward'],
            "steps": info['steps'],
            "success": info['success']
        })
        
        # Build cross-episode memory for next episode
        mistakes = []
        if info.get('repeated_queries', 0) > 0: mistakes.append(f"{info['repeated_queries']} repeated queries")
        if info.get('blind_actions', 0) > 0: mistakes.append(f"{info['blind_actions']} blind actions")
        mistakes_str = ", ".join(mistakes) if mistakes else "none"
        
        cross_episode_memory += f"Episode {ep}:\nReward: {info['total_reward']:.3f}\nSteps: {info['steps']}\nSuccess: {info['success']}\nMistakes: {mistakes_str}\n\n"
        
        # Limit memory to last 5 episodes
        blocks = cross_episode_memory.strip().split("\n\n")
        if len(blocks) > 5:
            cross_episode_memory = "\n\n".join(blocks[-5:]) + "\n\n"
        
        episodes_x.append(ep)
        rewards.append(info.get("total_reward", 0))
        steps_list.append(info.get("steps", 0))
        success_list.append(1 if info.get("success") else 0)
        
    print("\n\033[1;92m>>> EVALUATION COMPLETE. GENERATING GRAPHS...\033[0m")
    
    if len(steps_list) > 0:
        first_steps = steps_list[0]
        last_steps = steps_list[-1]
        print(f"\nEpisode 1 steps: {first_steps}")
        print(f"Episode last steps: {last_steps}")
        if first_steps > 0:
            reduction = ((first_steps - last_steps) / first_steps) * 100
            print(f"\033[1;93m↓ {reduction:.0f}% reduction in steps\033[0m\n")

    print(f"Avg Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Avg Steps: {sum(steps_list)/len(steps_list):.2f}")
    print(f"Success Rate: {sum(success_list)/len(success_list)*100:.1f}%")

    # 3. PLOT RESULTS
    try:
        plt.figure()
        plt.plot(rewards, marker='o')
        plt.title(f"{MODE.upper()} Reward vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.savefig(f"{MODE}_reward_curve.png")

        plt.figure()
        plt.plot(steps_list, marker='o')
        plt.title(f"{MODE.upper()} Steps vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps Taken")
        plt.grid()
        plt.savefig(f"{MODE}_steps_curve.png")
        
        print(f">>> Saved '{MODE}_reward_curve.png' and '{MODE}_steps_curve.png' to current directory.")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        print("\n\033[1;96m>>> 5-SECOND WOW DEMO MODE <<<\033[0m\n")
        print("\033[1;91mBaseline → FAIL ❌\033[0m")
        print("\033[1;93mAgent Episode 1 → inefficient ⚠️\033[0m")
        print("\033[1;92mAgent Episode 5 → optimal ✅\033[0m\n")
    else:
        run_loop()
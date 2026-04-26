"""
ACRS Fine-Tuning — Colab Cells
================================
Copy each section into a separate Colab cell.
Run in order: Cell 1 (SFT) → Cell 2 (Test) → Cell 3 (GRPO)

Prerequisites (run these cells first):
    !pip install unsloth trl datasets

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="sft_dataset.jsonl", split="train")
"""


"""
import json
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ── Correct scenario chains (must match environment's required_fixes) ──────────
VALID_TOOLS = {
    "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
    "clear_db_connections", "restart_service", "scale_service", "flush_cache",
}

SCENARIO_CHAINS = {
    "cascading db failure": [
        {"action_type": "tool_call",     "tool": "get_db_metrics",       "params": {}},
        {"action_type": "tool_call",     "tool": "get_error_logs",       "params": {}},
        {"action_type": "system_action", "tool": "clear_db_connections", "params": {}},
        {"action_type": "system_action", "tool": "restart_service",      "params": {"service": "db-service"}},
        {"action_type": "system_action", "tool": "restart_service",      "params": {"service": "api-service"}},
    ],
    "stale cache storm": [
        {"action_type": "tool_call",     "tool": "get_network_latency",  "params": {}},
        {"action_type": "tool_call",     "tool": "get_cache_status",     "params": {}},
        {"action_type": "system_action", "tool": "flush_cache",          "params": {}},
        {"action_type": "system_action", "tool": "restart_service",      "params": {"service": "api-service"}},
    ],
    "network latency storm": [
        {"action_type": "tool_call",     "tool": "get_network_latency",  "params": {}},
        {"action_type": "tool_call",     "tool": "get_cache_status",     "params": {}},
        {"action_type": "system_action", "tool": "flush_cache",          "params": {}},
        {"action_type": "system_action", "tool": "restart_service",      "params": {"service": "api-service"}},
    ],
    "distributed deadlock": [
        {"action_type": "tool_call",     "tool": "get_db_metrics",       "params": {}},
        {"action_type": "tool_call",     "tool": "get_error_logs",       "params": {}},
        {"action_type": "system_action", "tool": "flush_cache",          "params": {}},
        {"action_type": "system_action", "tool": "clear_db_connections", "params": {}},
        {"action_type": "system_action", "tool": "restart_service",      "params": {"service": "api-service"}},
    ],
    "hybrid failure": [
        {"action_type": "tool_call",     "tool": "get_network_latency",  "params": {}},
        {"action_type": "tool_call",     "tool": "get_error_logs",       "params": {}},
        {"action_type": "system_action", "tool": "scale_service",        "params": {"service": "db-service"}},
        {"action_type": "system_action", "tool": "flush_cache",          "params": {}},
        {"action_type": "system_action", "tool": "restart_service",      "params": {"service": "api-service"}},
    ],
}

def to_strict_example(example):
    user = [m for m in example["messages"] if m["role"] == "user"][0]
    asst = [m for m in example["messages"] if m["role"] == "assistant"][0]

    try:
        action = json.loads(asst["content"])
    except:
        return None

    # Validate: must have a real tool
    if not isinstance(action, dict) or action.get("tool") not in VALID_TOOLS:
        return None

    # Detect scenario from the user prompt
    lower = user["content"].lower()
    scenario = ""
    for name in SCENARIO_CHAINS:
        if name in lower:
            scenario = name
            break

    chain = SCENARIO_CHAINS.get(scenario)
    if chain:
        # Find where this action sits in the correct chain
        current_tool = action["tool"]
        idx = None
        for i, c in enumerate(chain):
            if c["tool"] == current_tool:
                idx = i
                break

        if idx is not None:
            # Teach the model the NEXT 2-3 correct steps from here
            plan_actions = chain[idx : idx + 3]
        else:
            plan_actions = [action]
    else:
        plan_actions = [action]

    plan = {
        "hypothesis": f"Addressing {scenario or 'incident'} based on current signals.",
        "reasoning": f"Following dependency chain for {scenario or 'this incident'}.",
        "actions": plan_actions,
    }

    return {
        "messages": [
            user,
            {"role": "assistant", "content": json.dumps(plan)}
        ]
    }

train_ds = dataset.map(to_strict_example).filter(lambda x: x is not None)
print(f"✅ SFT examples after conversion: {len(train_ds)}")

def format_chat(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

train_ds = train_ds.map(format_chat)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=3e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs_sft",
        max_grad_norm=1.0,
    ),
)

print("🚀 SFT training...")
trainer.train()
"""



"""
import json
from auto_sre_env.environment import AutoSREEnv
from auto_sre_env.models import Action

VALID_TOOLS = {
    "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
    "clear_db_connections", "restart_service", "scale_service", "flush_cache",
}
FIX_TOOLS = {"clear_db_connections", "restart_service", "scale_service", "flush_cache"}

def extract_plan(text):
    try:
        start = text.find("{")
        if start == -1:
            return []
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            if depth == 0:
                data = json.loads(text[start:i+1])
                if "actions" in data:
                    return [a for a in data["actions"]
                            if isinstance(a, dict) and a.get("tool") in VALID_TOOLS]
                return []
        return []
    except:
        return []

def test_agent(num_episodes=10):
    success_count = 0

    for ep in range(num_episodes):
        env = AutoSREEnv(difficulty="training")
        obs = env.reset()
        scenario = env.state.get("name", "Unknown")

        print(f"\\n{'='*50}")
        print(f"EPISODE {ep} — {scenario}")
        print(f"{'='*50}")

        done = False
        step = 1
        history_lines = []

        while not done and step <= 10:
            # Build prompt that MATCHES the SFT training format
            services_str = "\\n".join(
                f"  {name}: {info.get('status','?').upper()} | CPU: {info.get('cpu','?')}%"
                for name, info in obs.services.items()
            )
            logs_str = "\\n".join(f"  {l}" for l in obs.logs[-5:])
            hist_str = "\\n".join(history_lines) if history_lines else "  (none)"

            prompt = f\"\"\"You are an SRE Incident Commander restoring a failing system.

CURRENT STATE:
{services_str}
Latency: {obs.latency}ms
Logs:
{logs_str}
Step: {step} of 10

ACTIONS TAKEN:
{hist_str}

AVAILABLE TOOLS:
- get_network_latency()
- get_error_logs()
- get_db_metrics()
- get_cache_status()
- clear_db_connections()
- restart_service(service="api-service" OR "db-service")
- scale_service(service="db-service")
- flush_cache()

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "hypothesis": "...",
  "reasoning": "...",
  "actions": [
    {{"action_type": "tool_call" or "system_action", "tool": "...", "params": {{}}}}
  ]
}}

NO explanation outside JSON. MAX 3 actions per plan.
\"\"\"

            # Use chat template to match SFT training format
            messages = [{"role": "user", "content": prompt}]
            chat_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat_input, return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            plan = extract_plan(text)

            if not plan:
                print(f"  Step {step}: ❌ INVALID OUTPUT")
                print(f"  Last 200 chars: {text[-200:]}")
                break

            for action_dict in plan:
                tool = action_dict.get("tool", "?")
                params = action_dict.get("params", {})
                # Force correct action_type for fix tools
                atype = "system_action" if tool in FIX_TOOLS else "tool_call"

                try:
                    action = Action(action_type=atype, tool=tool, params=params)
                    obs, reward, done, info = env.step(action)
                    history_lines.append(f"  Step {step}: {tool} -> reward {reward:+.3f}")
                    print(f"  Step {step}: {tool} {params or ''} → {reward:+.3f}")
                    step += 1
                except Exception as e:
                    print(f"  Step {step}: {tool} FAILED: {e}")
                    break

                if done:
                    break

        if done:
            print(f"  ✅ SUCCESS in {step-1} steps")
            success_count += 1
        else:
            print(f"  ❌ FAILED")

    print(f"\\n{'='*50}")
    print(f"SUCCESS RATE: {success_count}/{num_episodes}")

test_agent(10)
"""



"""
from trl import GRPOConfig, GRPOTrainer
import json
from unsloth import PatchFastRL
from auto_sre_env.environment import AutoSREEnv
from auto_sre_env.models import Action

PatchFastRL("GRPO", FastLanguageModel)

VALID_TOOLS = {
    "get_network_latency", "get_error_logs", "get_db_metrics", "get_cache_status",
    "clear_db_connections", "restart_service", "scale_service", "flush_cache",
}
FIX_TOOLS = {"clear_db_connections", "restart_service", "scale_service", "flush_cache"}

def extract_plan(completion):
    try:
        text = completion[0]["content"]
        start = text.find("{")
        if start == -1: return []
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            if depth == 0:
                data = json.loads(text[start:i+1])
                if "actions" in data:
                    return [a for a in data["actions"]
                            if isinstance(a, dict) and a.get("tool") in VALID_TOOLS]
                return []
        return []
    except:
        return []

# Prompt: use existing user message (already has full context from SFT dataset)
def extract_prompts(example):
    msgs = [m for m in example["messages"] if m["role"] == "user"]
    return {"prompt": msgs}

rl_ds = dataset.map(extract_prompts)

# ── Reward with PARTIAL CREDIT (prevents all -3.0 → NaN) ──────────────────────
def live_env_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            plan = extract_plan(completion)

            # No valid plan at all → -1.0 (not -3.0, avoid NaN from zero variance)
            if not plan:
                rewards.append(-1.0)
                continue

            # Has valid JSON with correct schema → start at +0.5
            reward = 0.5

            env = AutoSREEnv(difficulty="training")
            obs = env.reset()
            done = False

            for step_dict in plan:
                tool = step_dict.get("tool", "")
                params = step_dict.get("params", {})
                atype = "system_action" if tool in FIX_TOOLS else "tool_call"

                try:
                    action = Action(action_type=atype, tool=tool, params=params)
                    obs, step_reward, done, info = env.step(action)
                    # Give credit for each step that doesn't fail
                    reward += max(step_reward, 0)
                    reward -= 0.05  # small efficiency penalty
                except:
                    reward -= 0.2
                    break

                if done:
                    break

            if done:
                reward += 5.0  # big bonus for full resolution

            rewards.append(round(reward, 3))
        except:
            rewards.append(-1.0)

    return rewards

cfg = GRPOConfig(
    output_dir="grpo_output",
    learning_rate=2e-6,
    max_steps=50,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=2,

    max_prompt_length=2048,
    max_completion_length=400,   # was 120! too short for hypothesis+reasoning+actions

    logging_steps=1,
    report_to="none",
    seed=3407,

    temperature=0.7,            # was 0.01! need exploration for reward variance
    top_p=0.9,
    repetition_penalty=1.2,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[live_env_reward],
    args=cfg,
    train_dataset=rl_ds,
)

print("🚀 GRPO training...")
trainer.train()
"""

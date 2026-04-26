import json
import os
import sys
import concurrent.futures

# Optional: Try to import tqdm for progress bar, otherwise fallback to simple print
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"{desc}...")
        for i, item in enumerate(iterable):
            if i > 0 and i % 10 == 0:
                print(f"  Progress: {i}/{kwargs.get('total', len(iterable))}")
            yield item

from auto_sre_env.environment import AutoSREEnv
from agent_loop import run_agent

NUM_EPISODES = 4000  # Start with 1000 to verify stability
OUTPUT_FILE = "sft_dataset.jsonl"
CONCURRENCY = 40  # 30-50 is the sweet spot with 6+ tokens

def run_single_episode():
    """Worker function to run a single episode and return its successful records."""
    trajectory = []
    summary = {}
    
    try:
        # Run the agent silently with no delay
        for item in run_agent(max_steps=10, delay=0, stream=True, silent=True):
            if item.get("type") == "summary":
                summary = item
            else:
                trajectory.append(item)
                
        episode_records = []
        is_success = False
        
        # Only extract data from episodes where the incident was successfully resolved
        if summary.get("status") == "RESOLVED":
            is_success = True
            for step in trajectory:
                prompt = step.get("prompt")
                raw_response = step.get("raw_response")
                
                # Exclude any steps that completely failed to get any JSON or Action at all
                if prompt is None or raw_response is None or not isinstance(raw_response, dict):
                    continue

                # Handle multi-step plans: skip empty, extract first action for SFT
                if "actions" in raw_response:
                    if len(raw_response["actions"]) == 0:
                        continue
                    action = raw_response["actions"][0]
                else:
                    action = raw_response
                
                # Format single action to JSON for SFT training
                response_str = json.dumps(action, indent=2)
                
                record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_str}
                    ]
                }
                episode_records.append(record)
                
        return is_success, episode_records
    except Exception as e:
        # Catch unexpected API crashes so the whole thread pool doesn't die
        return False, []


def generate_dataset():
    print(f"Generating {NUM_EPISODES} episodes of Auto-SRE for SFT dataset...")
    print(f"Concurrency level: {CONCURRENCY} threads")
    print(f"Goal: Run episodes, ignore failures, and save perfect runs to {OUTPUT_FILE}\n")
    
    successful_episodes = 0
    dataset_records = []
    
    # Use ThreadPoolExecutor to run episodes in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # Submit all tasks
        futures = [executor.submit(run_single_episode) for _ in range(NUM_EPISODES)]
        
        # Process them as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_EPISODES, desc="Running Episodes"):
            is_success, records = future.result()
            if is_success:
                successful_episodes += 1
                dataset_records.extend(records)

    print(f"\n[DONE] Generated {len(dataset_records)} QA pairs from {successful_episodes}/{NUM_EPISODES} successful episodes.")
    
    # Save in standard JSONL format natively supported by Hugging Face Datasets & Unsloth
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for record in dataset_records:
            f.write(json.dumps(record) + "\n")
            
    print(f"Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
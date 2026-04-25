import random
from datetime import datetime
from .models import Action, Observation, State
from .tasks import get_scenarios

from .sub_agent_router import route_action
from .judge import EpisodeTracker, compute_delegation_reward

class AutoSREEnv:
    def __init__(self, difficulty="hard"):
        self.state = None
        self.done = False
        self.step_count = 0
        self.last_score = 0.0
        self.difficulty = difficulty
        self.episode_tracker = EpisodeTracker()
        self.signals_gathered = 0
        self.system_phase = "NORMAL"

    def reset(self):
        # Pick one of the 5 new scenarios
        scenarios = get_scenarios()
        self.state = random.choice(scenarios)
        
        self.done = False
        self.step_count = 0
        self.last_score = 0.0
        self.episode_tracker = EpisodeTracker()
        self.signals_gathered = 0
        self.system_phase = "CHAOS"

        # Partial observability: Start with an alert
        ts = datetime.utcnow().strftime("%H:%M:%S")
        scenario_name = self.state.get("name", "Unknown Incident")
        print(f"\033[91m[CHAOS] Injecting scenario: {scenario_name}\033[0m")
        alert = f"[{ts}] CRITICAL system: [CHAOS] Injecting scenario: {scenario_name}. E2E Latency {self.state['latency']}ms."
        
        self.state["logs"].append(alert)
        self.system_phase = "DEGRADED"

        return Observation(
            services={k: {"status": v["status"]} for k, v in self.state["services"].items()},
            logs=[alert],
            latency=0,
        )

    def step(self, action: Action):
        self.episode_tracker.steps += 1
        ts = datetime.utcnow().strftime("%H:%M:%S")

        # 0. TRANSLATE NEW API TOOLS TO LEGACY ACTIONS
        if action.action_type == "tool_call":
            if action.tool == "get_db_metrics":
                action.target, action.query = "@db-admin", "db_load"
            elif action.tool == "get_network_latency":
                action.target, action.query = "@network-eng", "latency_breakdown"
            elif action.tool == "get_error_logs":
                action.target, action.query = "@network-eng", "error_rate"
            elif action.tool == "get_cache_status":
                action.target, action.query = "@db-admin", "cache_status"
            elif action.tool == "clear_db_connections":
                action.target, action.delegate_action = "@db-admin", "clear_connections"
        elif action.action_type == "system_action":
            if action.tool == "restart_service":
                action.action_type = "restart"
                action.target = action.params.get("service", "")
                if action.target == "db-service":
                    action.action_type, action.target, action.delegate_action = "delegate", "@db-admin", "restart_db"
            elif action.tool == "scale_service":
                action.action_type = "scale"
                action.target = action.params.get("service", "")
            elif action.tool == "flush_cache":
                action.action_type = "flush_cache"
                action.target = "cache"

        # 1. HANDLE DELEGATION (QUERIES)
        if action.target and action.target.startswith("@"):
            if not action.delegate_action:
                self.signals_gathered += 1
            return self._handle_delegation(action)

        # 2. HANDLE FIXES
        is_fix = action.action_type in ["restart", "scale", "flush_cache"]
        
        # PENALTY: Fix before collecting 2 signals
        penalty = 0.0
        if is_fix and self.signals_gathered < 2:
            penalty = -0.5
            self.state["logs"].append(f"[{ts}] WARN    judge: PREMATURE FIX ATTEMPTED (-0.5). Not enough diagnostics gathered.")

        # EXECUTE FIX LOGIC
        fix_key = action.delegate_action or action.action_type
        if action.action_type == "restart" and action.target == "api-service":
            fix_key = "restart_api"
        elif action.action_type == "scale" and action.target == "db-service":
            fix_key = "scale_db"
        elif action.action_type == "flush_cache":
            fix_key = "flush_cache"
        elif action.action_type == "restart" and action.target == "db-service":
            fix_key = "restart_db"
            
        # Check if fix is in required list
        if fix_key in self.state["required_fixes"]:
            # Check dependency: can't restart API if DB is still overloaded
            if fix_key == "restart_api" and ("restart_db" in self.state["required_fixes"] and "restart_db" not in self.state["applied_fixes"]):
                self.state["logs"].append(f"[{ts}] ERROR   api-service: restart failed — backend DB still unstable")
            else:
                if fix_key not in self.state["applied_fixes"]:
                    self.state["applied_fixes"].append(fix_key)
                    self.state["logs"].append(f"[{ts}] INFO    system: applied fix '{fix_key}' — metrics stabilizing...")
                    self._apply_partial_effect(fix_key)

        # 3. UPDATE STATE & PROPAGATE
        # (Simplified propagation for the new logic)
        self._update_metrics()

        # 4. GRADING & DONE CONDITION
        # ALL required fixes must be applied, AND metrics must be healthy
        all_fixes_done = all(f in self.state["applied_fixes"] for f in self.state["required_fixes"])
        metrics_healthy = (self.state["latency"] < 200 and 
                          all(s["status"] == "running" for s in self.state["services"].values()))
        
        if all_fixes_done and metrics_healthy:
            self.done = True
            self.episode_tracker.successful_fix = True
            self.system_phase = "RECOVERY"
            base_reward = 1.0 if self.signals_gathered >= 2 else 0.2
            reward = base_reward + penalty
        else:
            self.done = False
            if self.state["latency"] > 1000:
                self.system_phase = "FAILURE"
            else:
                self.system_phase = "DEGRADED"
            reward = penalty - 0.05 # Step penalty

        self.step_count += 1
        self.episode_tracker.total_reward += reward
        
        return self._make_observation(), round(reward, 3), self.done, self.episode_tracker.to_info_dict()

    def _handle_delegation(self, action):
        obs, reward, done, info = self._handle_delegation_core(action)
        return obs, round(reward, 3), done, info

    def _handle_delegation_core(self, action):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        query_key = action.query or action.delegate_action or "summary"
        novelty = self.episode_tracker.record_query(action.target, query_key)
        
        response = route_action(action, self.state)
        self.state["logs"].append(f"[{ts}] INFO    {response.agent}: {response.message}")
        
        penalty = 0.0
        if action.delegate_action and self.signals_gathered < 2:
            penalty = -0.5
            self.state["logs"].append(f"[{ts}] WARN    judge: PREMATURE FIX ATTEMPTED (-0.5). Not enough diagnostics gathered.")
            
        reward = -0.05 + penalty # Base step penalty
        if novelty: reward += 0.1 # Reward for new info
        
        # Check if sub-agent mutation (e.g. clear_connections) happened
        if action.delegate_action and action.delegate_action in self.state["required_fixes"]:
            if action.delegate_action not in self.state["applied_fixes"]:
                self.state["applied_fixes"].append(action.delegate_action)
                self._apply_partial_effect(action.delegate_action)
        
        # Check done condition after delegation too
        all_fixes_done = all(f in self.state["applied_fixes"] for f in self.state["required_fixes"])
        if all_fixes_done:
            self._update_metrics()
            if self.state["latency"] < 200 and all(s["status"] == "running" for s in self.state["services"].values()):
                self.done = True
                self.episode_tracker.successful_fix = True
                self.system_phase = "RECOVERY"
                base_reward = 0.5 if self.signals_gathered >= 2 else 0.1
                reward += base_reward
        else:
            self.done = False
            if self.state["latency"] > 1000:
                self.system_phase = "FAILURE"
            else:
                self.system_phase = "DEGRADED"

        self.step_count += 1
        self.episode_tracker.total_reward += reward
        return self._make_observation(), reward, self.done, self.episode_tracker.to_info_dict()

    def _apply_partial_effect(self, fix):
        """Reduces metrics slightly for partial fixes."""
        if fix == "clear_connections":
            self.state["services"]["db-service"]["cpu"] = 60
            self.state["latency"] -= 500
        elif fix == "restart_db":
            self.state["services"]["db-service"]["status"] = "running"
            self.state["services"]["db-service"]["latency"] = 50
        elif fix == "flush_cache":
            self.state["services"]["cache-service"]["status"] = "running"
            self.state["latency"] -= 1000
        elif fix == "restart_api":
            self.state["services"]["api-service"]["status"] = "running"
        elif fix == "scale_db":
            self.state["services"]["db-service"]["cpu"] -= 30

    def _update_metrics(self):
        """Dynamic metric degradation over time."""
        for name, info in self.state["services"].items():
            if info["status"] in ["overloaded", "degraded"]:
                if "cpu" in info and info["cpu"] < 100:
                    info["cpu"] = min(100, info["cpu"] + random.randint(2, 5))
                if "latency" in info:
                    info["latency"] += random.randint(50, 150)
                self.state["latency"] += random.randint(50, 200)# If not all fixes applied, keep some bad metrics
        all_done = all(f in self.state["applied_fixes"] for f in self.state["required_fixes"])
        if not all_done:
            self.state["latency"] = max(300, self.state["latency"] - 100)
        else:
            self.state["latency"] = 50
            for s in self.state["services"].values():
                s["status"] = "running"
                s["cpu"] = 20

    def _make_observation(self):
        return Observation(
            services=self.state["services"],
            logs=self.state["logs"][-20:],
            latency=self.state["latency"]
        )
        
    def get_state(self):
        from .models import State
        return State(step_count=self.step_count, done=self.done, system_phase=self.system_phase)
from typing import Dict

ACTION_COSTS = {
    "restart":     0.05,
    "scale":       0.03,
    "flush_cache": 0.01,
    "drain":       0.08,
}

def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1, never 0.0 or 1.0."""
    return round(max(0.01, min(0.99, score)), 3)


def compute_shaped_reward(base_score: float, action, step_count: int, state: dict) -> float:
    # Time penalty after step 3
    time_penalty = max(0.0, (step_count - 3) * 0.02)

    # Action cost
    action_cost = ACTION_COSTS.get(action.action_type, 0.0)

    # Penalty for acting on a healthy service unnecessarily
    wrong_action_penalty = 0.0
    if action.target and action.target in state.get("services", {}):
        target_status = state["services"][action.target].get("status", "")
        if action.action_type == "restart" and target_status == "running":
            wrong_action_penalty = 0.1

    final = base_score - time_penalty - action_cost - wrong_action_penalty
    return round(max(0.01, min(0.99, final)), 3)


def grade_easy(state: Dict) -> float:
    svc = state["services"].get("api-service", {})
    if svc.get("status") == "running":
        return _clamp(1.0)
    if svc.get("cpu", 0) > 0:   # partial: service is starting
        return _clamp(0.3)
    return _clamp(0.0)


def grade_medium(state: Dict) -> float:
    latency = state.get("latency", 9999)
    db_status = state["services"].get("db-service", {}).get("status", "")

    if latency < 500:
        return _clamp(1.0)
    elif latency < 800:
        return _clamp(0.7)
    elif latency < 1000:
        return _clamp(0.5)
    elif db_status == "running":
        return _clamp(0.2)   # db fixed but latency not yet recovered
    return _clamp(0.0)


def grade_hard(state: Dict) -> float:
    api_status = state["services"].get("api-service", {}).get("status", "")
    db_status  = state["services"].get("db-service",  {}).get("status", "")

    if api_status == "running" and db_status == "running":
        return _clamp(1.0)
    elif db_status == "running" and api_status == "degraded":
        return _clamp(0.6)
    elif db_status == "running":
        return _clamp(0.4)
    elif db_status == "degraded":
        return _clamp(0.2)
    return _clamp(0.0)
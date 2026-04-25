"""
Chaos Engine — rule-based failure injection for the IC training environment.

Wraps the existing dependency_graph.py and log_generator.py into a clean
interface. No LLM — purely deterministic scenario setup.
"""

from .dependency_graph import propagate_failures
from .log_generator import generate_logs
from .tasks import get_easy_task, get_medium_task, get_hard_task, get_impossible_task
import random


def inject_failure(state: dict, difficulty: str) -> dict:
    """Apply chaos injection to an existing state based on difficulty.

    This wraps the existing task generators and applies additional failure
    propagation to ensure cascading effects are properly simulated.

    Args:
        state: The current environment state dict.
        difficulty: One of "easy", "medium", "hard".

    Returns:
        The mutated state with failures injected.
    """
    if difficulty == "easy":
        return _inject_easy(state)
    elif difficulty == "medium":
        return _inject_medium(state)
    elif difficulty == "hard":
        return _inject_hard(state)
    elif difficulty == "impossible":
        return state # Custom state already built
    return state


def get_initial_state(difficulty: str) -> dict:
    """Create a fresh initial state for a given difficulty.

    Delegates to existing task generators, then applies failure propagation.

    Args:
        difficulty: One of "easy", "medium", "hard".

    Returns:
        A fully initialized state dict ready for the environment.
    """
    if difficulty == "training":
        difficulty = random.choice(["easy", "medium", "hard", "impossible"])
        
    if difficulty == "easy":
        state = get_easy_task()
    elif difficulty == "medium":
        state = get_medium_task()
    elif difficulty == "hard":
        state = get_hard_task()
    elif difficulty == "impossible":
        state = get_impossible_task()
    else:
        state = get_easy_task()

    # Apply cascading failure propagation
    state["services"] = propagate_failures(state["services"])
    return state


def get_initial_alert(state: dict, difficulty: str) -> str:
    """Generate a high-level alert string for partial observability.

    The agent sees only this alert at reset time — full details must be
    discovered via sub-agent queries.

    Args:
        state: The initial environment state.
        difficulty: The difficulty level.

    Returns:
        A short, realistic alert message.
    """
    if difficulty == "training":
        # At reset, we don't know which one was picked because state is already built.
        # We deduce the scenario from the goal in state.
        goal = state.get("goal")
        if goal == "restart_api":
            difficulty = "easy"
        elif goal == "reduce_latency":
            difficulty = "medium"
        elif goal == "fix_cascade":
            difficulty = "hard"
        elif goal == "fix_deadlock":
            difficulty = "impossible"

    alerts = {
        "easy": "P0 ALERT: api-service health check failing — 503 responses on all endpoints. Downstream impact unknown.",
        "medium": "P0 ALERT: End-to-end latency spike detected (>1500ms). Multiple services may be affected. Root cause unclear.",
        "hard": "P0 ALERT: Cascading failure in progress. api-service DOWN, db-service unresponsive. Multiple teams paged.",
        "impossible": "P0 ALERT: Distributed deadlock detected. API and DB report healthy, but end-to-end latency is 5000ms. Immediate investigation required.",
    }
    return alerts.get(difficulty, "P0 ALERT: System anomaly detected. Investigation required.")


def mask_services_for_partial_obs(services: dict) -> dict:
    """Mask granular metrics from services for partial observability.

    Preserves service names and status (the "dashboard view") but hides
    CPU, memory, and latency numbers — those require sub-agent queries.

    Args:
        services: The full services dict.

    Returns:
        A masked copy with metrics zeroed out.
    """
    masked = {}
    for name, info in services.items():
        masked[name] = {
            "status": info.get("status", "unknown"),
            "cpu": 0,       # masked — query @db-admin or @network-eng
            "memory": 0,    # masked
            "latency": 0,   # masked
        }
    return masked


# ─────────────────────────────────────────────────────────────────────────────
# Internal scenario injectors
# ─────────────────────────────────────────────────────────────────────────────

def _inject_easy(state: dict) -> dict:
    """Easy: API health check failure + downstream timeout."""
    state["services"]["api-service"].update(
        {"status": "down", "cpu": 0, "memory": 0, "latency": 0}
    )
    state["services"] = propagate_failures(state["services"])
    state["logs"].extend(generate_logs(state["services"]))
    return state


def _inject_medium(state: dict) -> dict:
    """Medium: Cache degradation causing latency spike."""
    state["latency"] = 1500
    state["services"]["api-service"].update(
        {"status": "running", "cpu": 85, "memory": 70, "latency": 420}
    )
    state["services"] = propagate_failures(state["services"])
    state["logs"].extend(generate_logs(state["services"]))
    return state


def _inject_hard(state: dict) -> dict:
    """Hard: DB overload cascading to API crash."""
    state["services"]["db-service"].update(
        {"status": "overloaded", "cpu": 98, "memory": 95, "latency": 2200}
    )
    state["services"]["api-service"].update(
        {"status": "down", "cpu": 10, "memory": 20, "latency": 0}
    )
    state["services"] = propagate_failures(state["services"])
    state["logs"].extend(generate_logs(state["services"]))
    return state

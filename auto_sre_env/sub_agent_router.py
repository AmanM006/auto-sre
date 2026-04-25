"""
Sub-Agent Router — domain-scoped agents with upgraded diagnostics and stochastic noise.
"""

import random
from datetime import datetime
from .models import Action, SubAgentResponse

def _add_noise(value, noise_level=0.1):
    """Adds 10-20% stochastic noise to a metric value."""
    if isinstance(value, (int, float)):
        factor = 1 + random.uniform(-noise_level, noise_level)
        return round(value * factor, 2)
    return value

def route_action(action: Action, state: dict) -> SubAgentResponse:
    target = action.target
    if target == "@network-eng":
        return _handle_network_eng(action, state)
    elif target == "@db-admin":
        return _handle_db_admin(action, state)
    else:
        return SubAgentResponse(
            agent=target,
            response_type="error",
            data={},
            message=f"Unknown sub-agent: {target}",
        )

def _handle_network_eng(action: Action, state: dict) -> SubAgentResponse:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    query = action.query or "summary"
    services = state.get("services", {})
    latency = state.get("latency", 0)
    api_svc = services.get("api-service", {})
    api_status = api_svc.get("status", "unknown")

    if query == "traffic_status":
        data = {
            "request_rate_rps": _add_noise(1200 if api_status != "down" else 0),
            "error_rate_pct": _add_noise(95 if api_status == "down" else 40 if api_status != "running" else 2),
            "load_balancer_status": "unhealthy" if api_status in ("down", "overloaded", "degraded") else "healthy",
            "end_to_end_latency_ms": _add_noise(latency),
        }
        return SubAgentResponse(agent="@network-eng", response_type="metrics", data=data, 
                                message=f"LB Status: {data['load_balancer_status']}, E2E Latency: {data['end_to_end_latency_ms']}ms")

    elif query == "latency_breakdown":
        breakdown = {k: {"latency_ms": _add_noise(v.get("latency", 0))} for k, v in services.items()}
        return SubAgentResponse(agent="@network-eng", response_type="metrics", data=breakdown,
                                message=f"Latency breakdown: {breakdown}")

    elif query == "error_rate":
        # New query
        error_rate = 98 if api_status == "down" else 45 if api_status == "degraded" else 3
        return SubAgentResponse(agent="@network-eng", response_type="metrics", 
                                data={"error_rate_pct": _add_noise(error_rate)},
                                message=f"Current error rate: {error_rate}%")

    elif query == "upstream_health":
        # New query
        health = "critical" if api_status == "down" else "unstable" if api_status == "degraded" else "healthy"
        return SubAgentResponse(agent="@network-eng", response_type="metrics", 
                                data={"upstream_status": health},
                                message=f"Upstream health check: {health}")

    else:
        return SubAgentResponse(agent="@network-eng", response_type="metrics", data={"status": "running"},
                                message="Network sub-agent operational. Use 'traffic_status', 'latency_breakdown', 'error_rate', or 'upstream_health'.")

def _handle_db_admin(action: Action, state: dict) -> SubAgentResponse:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    db_svc = state.get("services", {}).get("db-service", {})

    if action.delegate_action:
        # Mutations — only modify service metrics. Fix tracking is done by environment.py.
        act = action.delegate_action
        if act == "clear_connections":
            db_svc["status"] = "running" if db_svc["cpu"] < 110 else "degraded"
            db_svc["cpu"] = max(40, db_svc["cpu"] - 40)
            return SubAgentResponse(agent="@db-admin", response_type="action_result", data={"success": True}, message="Connections cleared.")
        elif act == "restart_db":
            db_svc["status"] = "running"
            db_svc["cpu"] = 30
            return SubAgentResponse(agent="@db-admin", response_type="action_result", data={"success": True}, message="DB restarted.")
        elif act == "scale_db":
            db_svc["cpu"] = max(20, db_svc["cpu"] - 50)
            return SubAgentResponse(agent="@db-admin", response_type="action_result", data={"success": True}, message="DB scaled.")
        return SubAgentResponse(agent="@db-admin", response_type="error", data={}, message="Unknown action.")

    query = action.query or "summary"
    if query == "db_load":
        data = {"cpu": _add_noise(db_svc.get("cpu", 0)), "mem": _add_noise(db_svc.get("memory", 0)), "latency": _add_noise(db_svc.get("latency", 0))}
        return SubAgentResponse(agent="@db-admin", response_type="metrics", data=data, message=f"DB Load: {data}")

    elif query == "lock_status":
        # New query
        locked = db_svc.get("cpu", 0) > 90 or "deadlock" in state.get("goal", "")
        return SubAgentResponse(agent="@db-admin", response_type="metrics", data={"locked": locked}, 
                                message=f"Lock status: {'CONTENTION DETECTED' if locked else 'Clean'}")

    elif query == "slow_queries":
        # New query
        count = 15 if db_svc.get("cpu", 0) > 80 else 2
        return SubAgentResponse(agent="@db-admin", response_type="metrics", data={"slow_query_count": count},
                                message=f"Found {count} queries exceeding 500ms threshold")

    elif query == "connection_stats":
        active = 95 if db_svc.get("status") == "overloaded" else 40
        return SubAgentResponse(agent="@db-admin", response_type="metrics", 
                                data={"active": _add_noise(active), "max": 100},
                                message=f"Connections: {active}/100 active")

    elif query == "cache_status":
        cache_svc = state.get("services", {}).get("cache-service", {})
        status = cache_svc.get("status", "running")
        hit_rate = 15 if status != "running" else 98
        return SubAgentResponse(agent="@db-admin", response_type="metrics",
                                data={"status": status, "hit_rate_pct": _add_noise(hit_rate)},
                                message=f"Cache status: {status}, Hit Rate: {hit_rate}%")

    return SubAgentResponse(agent="@db-admin", response_type="metrics", data={"status": "online"},
                            message="DB sub-agent operational. Use 'db_load', 'lock_status', 'slow_queries', 'connection_stats', or 'cache_status'.")

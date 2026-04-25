import random
from datetime import datetime, timedelta

def _ts(seconds_ago):
    return (datetime.utcnow() - timedelta(seconds=seconds_ago)).strftime("%H:%M:%S")

def get_scenarios():
    """Returns a list of 5 complex scenario variants."""
    return [
        # 1. DB Overload + API Crash (Cascading Failure)
        {
            "name": "Cascading DB Failure",
            "goal": "fix_cascade_v2",
            "services": {
                "api-service": {"status": "down", "cpu": 0, "memory": 20, "latency": 0},
                "db-service":  {"status": "overloaded", "cpu": 98, "memory": 95, "latency": 2200},
                "cache-service": {"status": "running", "cpu": 30, "memory": 40, "latency": 5},
            },
            "latency": 3500,
            "required_fixes": ["clear_connections", "restart_db", "restart_api"],
            "applied_fixes": [],
            "logs": [
                f"[{_ts(60)}] ERROR   db-service: connection pool exhausted (active=100/100)",
                f"[{_ts(45)}] CRITICAL api-service: backend connection failure — process exiting",
                f"[{_ts(30)}] WARN    load-balancer: api-service unhealthy, removing from pool",
                f"[{_ts(15)}] INFO    system: multiple health checks failing across services",
            ]
        },
        # 2. Cache Inconsistency + Stale Reads
        {
            "name": "Stale Cache Storm",
            "goal": "fix_cache_v2",
            "services": {
                "api-service": {"status": "running", "cpu": 40, "memory": 50, "latency": 1500},
                "db-service":  {"status": "running", "cpu": 20, "memory": 30, "latency": 10},
                "cache-service": {"status": "degraded", "cpu": 85, "memory": 90, "latency": 500},
            },
            "latency": 2000,
            "required_fixes": ["flush_cache", "restart_api"],
            "applied_fixes": [],
            "logs": [
                f"[{_ts(60)}] WARN    cache-service: memory fragmentation high (85%)",
                f"[{_ts(45)}] ERROR   api-service: inconsistent data detected for key user_session",
                f"[{_ts(30)}] WARN    cache-service: cache miss rate spiked to 92%",
                f"[{_ts(15)}] INFO    api-service: retrying requests due to stale data",
            ]
        },
        # 3. Network Latency Spike (Ambiguous)
        {
            "name": "Network Latency Storm",
            "goal": "fix_network_v2",
            "services": {
                "api-service": {"status": "running", "cpu": 20, "memory": 30, "latency": 1200},
                "db-service":  {"status": "running", "cpu": 15, "memory": 25, "latency": 5},
                "cache-service": {"status": "running", "cpu": 10, "memory": 20, "latency": 2},
            },
            "latency": 1500,
            "required_fixes": ["flush_cache", "restart_api"],
            "applied_fixes": [],
            "logs": [
                f"[{_ts(60)}] INFO    api-service: process running but requests are slow",
                f"[{_ts(45)}] WARN    network: high packet loss detected on hop 7 (12%)",
                f"[{_ts(30)}] ERROR   gateway: 504 Gateway Timeout for 15% of requests",
                f"[{_ts(15)}] INFO    metrics: api-service internal latency is low (25ms), e2e is high",
            ]
        },
        # 4. Deadlock + Connection Exhaustion
        {
            "name": "Distributed Deadlock",
            "goal": "fix_deadlock_v2",
            "services": {
                "api-service": {"status": "running", "cpu": 95, "memory": 80, "latency": 3000},
                "db-service":  {"status": "running", "cpu": 10, "memory": 95, "latency": 1200},
                "cache-service": {"status": "running", "cpu": 5, "memory": 10, "latency": 2},
            },
            "latency": 5000,
            "required_fixes": ["flush_cache", "clear_connections", "restart_api"],
            "applied_fixes": [],
            "logs": [
                f"[{_ts(60)}] ERROR   db-service: distributed lock wait timeout",
                f"[{_ts(45)}] WARN    api-service: thread pool saturation (200/200 threads active)",
                f"[{_ts(30)}] INFO    cache-service: locking keys forever — no TTL being applied",
                f"[{_ts(15)}] ERROR   system: circular dependency detected (API -> DB -> Cache -> API)",
            ]
        },
        # 5. Mixed DB + Network Issue
        {
            "name": "Hybrid Failure",
            "goal": "fix_mixed_v2",
            "services": {
                "api-service": {"status": "degraded", "cpu": 70, "memory": 80, "latency": 1800},
                "db-service":  {"status": "running", "cpu": 92, "memory": 85, "latency": 800},
                "cache-service": {"status": "running", "cpu": 10, "memory": 20, "latency": 5},
            },
            "latency": 2800,
            "required_fixes": ["scale_db", "flush_cache", "restart_api"],
            "applied_fixes": [],
            "logs": [
                f"[{_ts(60)}] WARN    db-service: high query latency (800ms)",
                f"[{_ts(45)}] ERROR   network: congestion detected on db subnet",
                f"[{_ts(30)}] WARN    api-service: backpressure triggered — dropping requests",
                f"[{_ts(15)}] INFO    system: resolving db bottleneck may reveal network issues",
            ]
        }
    ]

# Keep original functions for backward compatibility but point to the new generator
def get_easy_task():
    return get_scenarios()[0]

def get_medium_task():
    return get_scenarios()[1]

def get_hard_task():
    return get_scenarios()[2]
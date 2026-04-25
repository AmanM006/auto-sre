from datetime import datetime, timedelta
import random

ERROR_TEMPLATES = {
    "db-service": {
        "overloaded": [
            "[{ts}] ERROR   db-service: connection pool exhausted (active={val}/100)",
            "[{ts}] WARN    db-service: slow query detected — execution time {val}ms",
            "[{ts}] ERROR   db-service: deadlock detected on table `transactions`",
        ],
        "down": [
            "[{ts}] CRITICAL db-service: process exited with code 137 (OOM killed)",
            "[{ts}] ERROR   db-service: failed to bind port 5432 — address in use",
        ],
        "degraded": [
            "[{ts}] WARN    db-service: response time elevated ({val}ms)",
            "[{ts}] WARN    db-service: connection queue depth {val}",
        ],
    },
    "api-service": {
        "down": [
            "[{ts}] CRITICAL api-service: health check failed 3 consecutive times",
            "[{ts}] ERROR   api-service: upstream db-service unreachable — connection refused",
        ],
        "degraded": [
            "[{ts}] WARN    api-service: response time {val}ms exceeds SLA (500ms)",
            "[{ts}] ERROR   api-service: 503 rate {val}% of requests in last 60s",
        ],
    },
    "cache-service": {
        "degraded": [
            "[{ts}] WARN    cache-service: hit rate dropped to {val}%",
            "[{ts}] INFO    cache-service: eviction rate elevated",
        ],
    },
}

RED_HERRINGS = [
    "[{ts}] INFO    metrics-exporter: scrape successful (15s interval)",
    "[{ts}] DEBUG   cache-service: TTL expiry sweep complete",
    "[{ts}] INFO    log-agent: buffer flushed (2.1MB)",
    "[{ts}] DEBUG   healthcheck: /ping returned 200 in 3ms",
]

def generate_logs(services: dict, base_time: datetime = None) -> list:
    base_time = base_time or datetime.utcnow()
    logs = []

    for service, info in services.items():
        status = info.get("status", "running")
        templates = ERROR_TEMPLATES.get(service, {}).get(status, [])
        for i, tmpl in enumerate(templates):
            ts = (base_time - timedelta(seconds=30 * (len(templates) - i))).strftime("%H:%M:%S")
            val = random.randint(80, 99)
            logs.append(tmpl.format(ts=ts, val=val))

    # Add 1-2 red herring logs
    ts = base_time.strftime("%H:%M:%S")
    for rh in random.sample(RED_HERRINGS, k=min(2, len(RED_HERRINGS))):
        logs.append(rh.format(ts=ts))

    random.shuffle(logs)
    return logs
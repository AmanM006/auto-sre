DEPENDENCY_GRAPH = {
    "api-service":     ["db-service", "cache-service"],
    "db-service":      ["storage-service"],
    "cache-service":   [],
    "storage-service": [],
    "auth-service":    ["db-service"],
}

def propagate_failures(services: dict) -> dict:
    for service, deps in DEPENDENCY_GRAPH.items():
        if service not in services:
            continue
        for dep in deps:
            if dep not in services:
                continue
            if services[dep]["status"] in ("down", "overloaded"):
                services[service]["cpu"] = min(100, services[service]["cpu"] + 20)
                services[service]["latency"] = services[service].get("latency", 100) * 1.5
                if services[service]["cpu"] >= 95:
                    services[service]["status"] = "degraded"
    return services
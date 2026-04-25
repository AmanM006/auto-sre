RUNBOOKS = {
    "RB-001": {
        "name": "Service Restart",
        "description": "Gracefully restart a named service",
        "action_type": "restart",
        "params": ["target_service"],
        "estimated_duration_sec": 30,
    },
    "RB-002": {
        "name": "Horizontal Scale-Out",
        "description": "Increase replica count for a service",
        "action_type": "scale",
        "params": ["target_service"],
        "estimated_duration_sec": 90,
    },
    "RB-003": {
        "name": "Cache Invalidation",
        "description": "Flush distributed cache layer",
        "action_type": "flush_cache",
        "params": [],
        "estimated_duration_sec": 5,
    },
    "RB-004": {
        "name": "Traffic Drain",
        "description": "Drain traffic from a service before maintenance",
        "action_type": "drain",
        "params": ["target_service"],
        "estimated_duration_sec": 15,
    },
}
from pydantic import BaseModel
from typing import Dict, List, Optional

# Action sent by agent
class Action(BaseModel):
    action_type: str                        # "restart", "scale", "flush_cache", or "delegate"
    target: Optional[str] = None            # service name OR "@network-eng" / "@db-admin"
    query: Optional[str] = None             # sub-agent query: "traffic_status", "db_load", etc.
    delegate_action: Optional[str] = None   # sub-agent mutation: "restart_db", "clear_connections"


# Response from a sub-agent (domain-scoped, partial view)
class SubAgentResponse(BaseModel):
    agent: str                              # "@network-eng" or "@db-admin"
    response_type: str                      # "metrics", "logs", "action_result"
    data: Dict                              # domain-specific payload
    message: str                            # human-readable summary for logs


# What agent sees
class Observation(BaseModel):
    services: Dict
    logs: List[str]
    latency: int


# Internal state (optional but good practice)
class State(BaseModel):
    step_count: int
    done: bool
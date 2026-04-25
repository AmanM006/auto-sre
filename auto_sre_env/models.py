from pydantic import BaseModel
from typing import Dict, List, Optional

# Action sent by agent
class Action(BaseModel):
    action_type: str                        # "tool_call", "system_action", "restart", "scale", "flush_cache", "delegate"
    
    # New ACRS Tool API
    tool: Optional[str] = None              # "get_db_metrics", "restart_service", etc.
    params: dict = {}                       # {"service": "api-service"}
    
    # Legacy compatibility fields
    target: Optional[str] = None            
    query: Optional[str] = None             
    delegate_action: Optional[str] = None   


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
    system_phase: str = "NORMAL"
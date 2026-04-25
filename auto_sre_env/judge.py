"""
Judge — enhanced reward system for the IC training environment.

Wraps the existing grader.py and adds delegation-aware reward shaping:
  +1.0  correct fix (from base grader)
  +0.3  useful query (new information discovered)
  -0.1  redundant query (same domain, no new info)
  -0.2  repeated exact query
  -0.5  wrong fix (action that doesn't change state)

Tracks per-episode metrics for the info dict.
"""

from typing import Dict, List, Optional


class EpisodeTracker:
    """Tracks per-episode metrics for reward shaping and info reporting."""

    def __init__(self):
        self.steps: int = 0
        self.queries_made: List[str] = []         # "agent:query" keys
        self.actions_taken: List[str] = []         # "action_type:target" keys
        self.delegation_count: int = 0
        self.total_reward: float = 0.0
        self.repeated_queries: int = 0
        self.useful_queries: int = 0
        self.wrong_fixes: int = 0
        self.successful_fix: bool = False
        self.blind_actions: int = 0

    def has_required_queries(self) -> bool:
        """Returns True if the agent has made at least 1 sub-agent query."""
        return len(self.queries_made) >= 1

    def record_query(self, agent: str, query: str) -> str:
        """Record a sub-agent query and return its novelty status.

        Returns:
            "new"       — first time this exact query was made
            "redundant" — same agent queried before, but different query
            "repeated"  — exact same agent+query combination seen before
        """
        key = f"{agent}:{query}"
        self.delegation_count += 1

        if key in self.queries_made:
            self.repeated_queries += 1
            return "repeated"

        # Check if agent was queried before (different query)
        agent_queried_before = any(q.startswith(f"{agent}:") for q in self.queries_made)
        self.queries_made.append(key)

        if agent_queried_before:
            # Same agent, new query — still useful but less so
            self.useful_queries += 1
            return "new"
        else:
            self.useful_queries += 1
            return "new"

    def record_action(self, action_type: str, target: str) -> bool:
        """Record an action. Returns True if it's a repeat."""
        key = f"{action_type}:{target}"
        is_repeat = key in self.actions_taken
        self.actions_taken.append(key)
        return is_repeat

    def to_info_dict(self) -> Dict:
        """Export episode metrics for the step info dict."""
        return {
            "steps": self.steps,
            "queries": len(self.queries_made),
            "unique_queries": len(set(self.queries_made)),
            "repeated_queries": self.repeated_queries,
            "useful_queries": self.useful_queries,
            "delegation_count": self.delegation_count,
            "wrong_fixes": self.wrong_fixes,
            "blind_actions": self.blind_actions,
            "success": self.successful_fix,
            "total_reward": round(self.total_reward, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────────────────────────────────────

# Reward constants
REWARD_USEFUL_QUERY = 0.3
REWARD_REDUNDANT_QUERY = -0.1
REWARD_REPEATED_QUERY = -0.2
REWARD_WRONG_FIX = -0.5
REWARD_BLIND_ACTION = -1.0
STEP_PENALTY = -0.05


def compute_delegation_reward(
    action,
    query_novelty: str,
    sub_agent_response,
    tracker: EpisodeTracker,
) -> float:
    """Compute reward for a delegation (sub-agent query/action).

    Args:
        action: The Action object.
        query_novelty: "new", "redundant", or "repeated" from tracker.record_query().
        sub_agent_response: The SubAgentResponse from the sub-agent.
        tracker: The episode tracker.

    Returns:
        Reward for this delegation step.
    """
    # Sub-agent write actions (delegate_action) — reward based on success
    if action.delegate_action:
        success = sub_agent_response.data.get("success", False)
        if success:
            return 0.5  # partial credit — the fix worked but may not fully resolve
        else:
            tracker.wrong_fixes += 1
            return REWARD_WRONG_FIX

    # Sub-agent read queries — reward based on novelty
    if query_novelty == "new":
        return REWARD_USEFUL_QUERY
    elif query_novelty == "redundant":
        return REWARD_REDUNDANT_QUERY
    elif query_novelty == "repeated":
        return REWARD_REPEATED_QUERY

    return 0.0


def compute_fix_reward(
    base_score: float,
    action,
    tracker: EpisodeTracker,
    state_changed: bool,
) -> float:
    """Compute reward for a direct fix action (legacy path).

    Wraps the existing grader score and applies penalties for wrong fixes.

    Args:
        base_score: The raw score from the existing grader (0.0–1.0).
        action: The Action object.
        tracker: The episode tracker.
        state_changed: Whether the action actually changed the system state.

    Returns:
        Shaped reward for this fix step.
    """
    # HARD CONSTRAINT: Penalize heavily if taking action without querying
    if not tracker.has_required_queries():
        tracker.blind_actions += 1
        return REWARD_BLIND_ACTION

    if not state_changed and base_score < 0.5:
        tracker.wrong_fixes += 1
        return REWARD_WRONG_FIX + STEP_PENALTY

    return base_score + STEP_PENALTY

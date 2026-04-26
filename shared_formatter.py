"""
shared_formatter.py — ACRS Unified Step Formatter
Produces identical output for both CLI and inference runs.
"""

from __future__ import annotations
from typing import Optional

# ── COLORS ────────────────────────────────────────────────────────────────────
C_CYAN  = '\033[38;2;0;220;255m'
C_TEXT  = '\033[38;2;220;230;240m'
C_MUTED = '\033[38;2;100;115;130m'
C_GREEN = '\033[38;2;0;255;160m'
C_RED   = '\033[38;2;255;80;80m'
C_GOLD  = '\033[38;2;255;200;60m'
C_MAG   = '\033[38;2;200;130;255m'
R       = '\033[0m'


# ── TYPES ─────────────────────────────────────────────────────────────────────

class StepData:
    """
    Structured container for a single agent step.
    Returned by agent_loop and consumed by format_step().
    """
    __slots__ = (
        "step", "state_summary", "action", "params", "result",
        "reward", "total_reward", "done",
        "hypothesis", "why", "source", "confidence",
    )

    def __init__(
        self,
        step: int,
        state_summary: Optional[dict],  # {"services": {...}, "latency": int}
        action: str,                    # tool name string
        params: dict,
        result: str,
        reward: float,
        total_reward: float,
        hypothesis: str = "",
        why: str = "",
        source: str = "",
        confidence: float = 0.0,
    ):
        self.step          = step
        self.state_summary = state_summary
        self.action        = action
        self.params        = params
        self.result        = result
        self.reward        = reward
        self.total_reward  = total_reward
        self.done          = done
        self.hypothesis    = hypothesis
        self.why           = why
        self.source        = source
        self.confidence    = confidence


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _sep(char: str = '─', width: int = 52) -> str:
    return f"{C_MUTED}{char * width}{R}"


def _status_color(status: str) -> str:
    s = status.upper()
    if s == 'RUNNING':
        return C_GREEN
    if s in ('DOWN', 'OVERLOADED'):
        return C_RED
    return C_GOLD


def _format_services(services: dict) -> str:
    if not services:
        return "?"
    parts = []
    for name, info in services.items():
        status = info.get('status', 'unknown').upper()
        c = _status_color(status)
        parts.append(f"{c}{name.upper()}: {status}{R}")
    return " | ".join(parts)


def _format_signals(services: dict) -> list[str]:
    signals = []
    db  = services.get("db-service", {})
    api = services.get("api-service", {})
    if db.get("cpu") is not None:
        signals.append(
            f"DB CPU: {db['cpu']}%  "
            f"Connections: {db.get('connections', '?')}/{db.get('max_connections', '?')}"
        )
    if api.get("cpu") is not None:
        signals.append(f"API CPU: {api['cpu']}%")
    return signals


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def format_step(step: StepData, mode: str = "clean") -> None:
    """
    Print a formatted step card to stdout.

    Parameters
    ----------
    step : StepData
        Structured data for one agent step.
    mode : "clean" | "debug" | "silent"
        - clean  → STATE, SIGNAL, ACTION, RESULT, REWARD
        - debug  → adds THINK and WHY blocks
        - silent → no output
    """
    if mode == "silent":
        return

    print(f"\n{C_CYAN}[STEP {step.step}]{R}")
    
    if step.confidence > 0:
        c = C_GREEN if step.confidence >= 0.8 else C_GOLD if step.confidence >= 0.5 else C_RED
        print(f"{C_MAG}CONFIDENCE:{R} {c}{step.confidence:.2f}{R}")

    print(_sep())

    # ── STATE ─────────────────────────────────────────────────────────────────
    if step.state_summary:
        services = step.state_summary.get("services", {})
        latency  = step.state_summary.get("latency", "?")
        svc_str  = _format_services(services)
        print(f"{C_CYAN}STATE:{R}")
        print(f"  {svc_str} | {C_MUTED}LATENCY: {latency}ms{R}")

        signals = _format_signals(services)
        if signals:
            print(f"{C_CYAN}SIGNAL:{R}")
            for s in signals:
                print(f"  {C_MUTED}{s}{R}")

    # ── DEBUG: hypothesis / why ───────────────────────────────────────────────
    if mode == "debug":
        if step.hypothesis:
            print(f"{C_MAG}THINK:{R}")
            print(f"  {step.hypothesis}")
        if step.why:
            print(f"{C_MAG}WHY:{R}")
            print(f"  {step.why}")

    # ── ACTION ────────────────────────────────────────────────────────────────
    param_str = ""
    if step.params:
        param_str = "  " + "  ".join(f"{k}={v}" for k, v in step.params.items())
    print(f"{C_CYAN}ACTION:{R}")
    print(f"  {C_TEXT}{step.action}{R}{C_MUTED}{param_str}{R}")

    # ── RESULT ────────────────────────────────────────────────────────────────
    print(f"{C_CYAN}RESULT:{R}")
    print(f"  {step.result}")

    # ── REWARD ────────────────────────────────────────────────────────────────
    r_color = C_GREEN if step.reward >= 0 else C_RED
    print(f"{C_CYAN}REWARD:{R}")
    print(
        f"  {r_color}{step.reward:+.3f}{R}  "
        f"{C_MUTED}(Total: {step.total_reward:+.3f}){R}"
    )

    # ── DONE ──────────────────────────────────────────────────────────────────
    if step.done:
        print(f"\n  {C_GREEN}✅ INCIDENT RESOLVED{R}")

    print(_sep())


def format_episode_summary(
    ep,
    scenario_name: str,
    info: dict,
    key_actions: list[str],
    source_majority: str,
) -> None:
    """Print a clean end-of-episode summary block."""
    success = info.get("success", False)
    color   = C_GREEN if success else C_RED
    icon    = "✔" if success else "✘"
    status  = "SUCCESS" if success else "FAILED"
    if success and info.get("steps", 99) <= 4:
        status = "SUCCESS (Optimal)"

    print(_sep('═', 52))
    print(f"{color}  Episode {ep}  ·  {status}{R}")
    print(_sep('═', 52))
    print(f"  {color}{icon} Scenario:{R}    {scenario_name}")
    print(f"  {color}{icon} Steps:{R}       {info.get('steps', '?')}")
    print(f"  {color}{icon} Reward:{R}      {info.get('total_reward', 0):+.3f}")
    print(f"  {color}{icon} Source:{R}      {source_majority}")

    if key_actions:
        chain = f"{C_MUTED} → {R}".join(f"{C_TEXT}{a}{R}" for a in key_actions)
        print(f"  {color}{icon} Fix Chain:{R}   {chain}")

    print()
#!/usr/bin/env python3
"""
ACRS CLI — Autonomous Cloud Recovery System Command Line Interface

A developer tool to control and monitor the ACRS agent via terminal.
Interacts with the ACRS FastAPI backend.
"""

import argparse
import sys
import os
import json
import requests
import colorama
from colorama import Fore, Style
from shared_formatter import (
    StepData, format_step, format_episode_summary,
    C_CYAN, C_TEXT, C_MUTED, C_GREEN, C_RED, C_GOLD, R,
    _format_services,
)

colorama.init(autoreset=True)

DEFAULT_HOST = "http://localhost:8000"

VIEW_MODE = "clean"  # set globally once args are parsed


# ── BANNER ────────────────────────────────────────────────────────────────────

def print_welcome_banner():
    C_BLUE = '\033[38;2;0;130;255m'

    cwd     = os.getcwd()
    cwd_str = cwd if len(cwd) <= 14 else "..." + cwd[-11:]

    print(f"\n{C_CYAN}╔══ ACRS CLI {C_TEXT}v1.0.0{C_CYAN} {'═'*58}╗\033[0m")
    print(f"{C_CYAN}║\033[0m                                                                             {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_BLUE}      ▄█████▄        {C_CYAN}SYSTEM STATUS: {C_GREEN}ONLINE                                   {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_BLUE}      ██▄▄▄██        {C_MUTED}{'-'*52}    {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_BLUE}      ███████        {C_TEXT}Quick Start:                                            {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_BLUE}      ██▄▄▄██        {C_TEXT}• Run {C_CYAN}`acrs reset`{C_TEXT} to init a chaos scenario             {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_BLUE}      ▀█████▀        {C_TEXT}• Run {C_CYAN}`acrs run`{C_TEXT} to start the autonomous agent          {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║\033[0m                                                                             {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_CYAN}   Autonomous SRE    {C_TEXT}Recent Activity:                                        {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║{C_MUTED}   {cwd_str:<14}    {C_MUTED}No recent incidents                                     {C_CYAN}║\033[0m")
    print(f"{C_CYAN}║\033[0m                                                                             {C_CYAN}║\033[0m")
    print(f"{C_CYAN}╚{'═'*77}╝\033[0m\n")


# ── PRINT HELPERS ─────────────────────────────────────────────────────────────

def print_error(msg):
    print(f"{C_RED}[ERROR] {msg}{R}")

def print_success(msg):
    print(f"{C_GREEN}[OK] {msg}{R}")

def print_warning(msg):
    print(f"{C_GOLD}[WARN] {msg}{R}")


def _api_response_to_step_data(data: dict, step_num: int) -> StepData:
    """
    Convert a raw API step response dict into a StepData object
    so format_step() can render it identically to inference output.
    """
    obs = data.get("observation", {})
    state_summary = None
    if obs:
        state_summary = {
            "services": obs.get("services", {}),
            "latency":  obs.get("latency"),
        }
    elif "state_summary" in data:
        state_summary = data["state_summary"]
    elif "state" in data:
        state_summary = data["state"]

    action = data.get("action") or data.get("tool", "unknown")

    return StepData(
        step          = data.get("step", step_num),
        state_summary = state_summary,
        action        = action,
        params        = data.get("params", {}),
        result        = data.get("result", ""),
        reward        = data.get("reward", 0.0),
        total_reward  = data.get("total_reward", 0.0),
        done          = data.get("done", False),
        hypothesis    = data.get("hypothesis", ""),
        why           = data.get("why", ""),
        source        = data.get("source", ""),
        confidence    = data.get("confidence", 0.0),
    )


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────

def print_final_summary(data: dict) -> None:
    """Print the clean final summary block."""
    status   = data.get('status', 'UNKNOWN')
    resolved = status == "RESOLVED"
    color    = C_GREEN if resolved else C_RED
    icon     = "✔" if resolved else "✘"

    _sep = lambda c='─', w=52: f"{C_MUTED}{c * w}{R}"

    print()
    print(_sep('═'))
    print(f"{C_CYAN}  FINAL SUMMARY{R}")
    print(_sep('═'))
    print(f"  {color}{icon} Scenario:{R}       {data.get('scenario', '?')}")
    print(f"  {color}{icon} Status:{R}         {color}{status}{R}")
    print(f"  {color}{icon} Steps Taken:{R}    {data.get('steps_taken', '?')}")
    print(f"  {color}{icon} Signals Used:{R}   {data.get('signals_used', '?')}")

    fixes = data.get('fixes_applied', [])
    if fixes:
        chain = f"{C_MUTED} → {R}".join(f"{C_TEXT}{f}{R}" for f in fixes)
        print(f"  {color}{icon} Fix Chain:{R}      {chain}")

    print(f"  {color}{icon} Final Latency:{R}  {data.get('final_latency', '?')}ms")
    print(f"  {color}{icon} Total Reward:{R}   {C_GREEN if resolved else C_RED}{data.get('total_reward', 0):+.2f}{R}")

    if not resolved:
        print()
        print(f"  {C_RED}Reason:{R}  {data.get('failure_reason', 'Unknown')}")
        print(f"  {C_GOLD}Tip:{R}     {data.get('suggested_improvement', '')}")

    print(_sep('═'))
    print()


# ── COMMANDS ──────────────────────────────────────────────────────────────────

def cmd_reset(args):
    print(f"Resetting environment at {args.host}...")
    try:
        res = requests.post(f"{args.host}/api/agent/reset", timeout=10)
        res.raise_for_status()
        data = res.json()
        print_success("Environment reset successfully.")
        print(f"  {C_CYAN}Scenario:{R} {data.get('scenario')}")
        print(f"  {C_CYAN}Phase:{R}    {data.get('phase')}")
        obs     = data.get('observation', {})
        svc_str = _format_services(obs.get('services', {}))
        print(f"  {C_CYAN}State:{R}    {svc_str} | Latency: {obs.get('latency')}ms")
    except Exception as e:
        print_error(f"Failed to reset environment: {e}")


def cmd_state(args):
    try:
        res = requests.get(f"{args.host}/api/state", timeout=5)
        res.raise_for_status()
        data = res.json()

        if not data.get("initialized"):
            print_warning("Environment not initialized. Run 'reset' first.")
            return

        _sep = lambda: print(f"{C_MUTED}{'─' * 52}{R}")

        _sep()
        print(f"{C_CYAN}CURRENT STATE{R}")
        print(f"  Scenario: {data.get('scenario')}")

        phase   = data.get('phase', '')
        p_color = C_GREEN if phase == 'NORMAL' else C_GOLD
        print(f"  Phase:    {p_color}{phase}{R}")
        print(f"  Latency:  {data.get('latency')}ms")
        print(f"  Reward:   {data.get('total_reward')}")
        print()
        print(f"{C_CYAN}SERVICES{R}")

        for name, info in data.get('services', {}).items():
            status = info.get('status', 'unknown').upper()
            color  = C_GREEN if status == 'RUNNING' else C_RED if status in ('DOWN', 'OVERLOADED') else C_GOLD
            print(f"  {name:<18} {color}{status:<12}{R} CPU: {info.get('cpu')}%  MEM: {info.get('memory')}%")

        _sep()
    except Exception as e:
        print_error(f"Failed to fetch state: {e}")


def cmd_logs(args):
    try:
        res = requests.get(f"{args.host}/api/logs", timeout=5)
        res.raise_for_status()
        logs = res.json().get("logs", [])

        print(f"{C_MUTED}{'─' * 52}{R}")
        print(f"{C_CYAN}SYSTEM LOGS{R}")

        if not logs:
            print(f"  {C_MUTED}No logs available.{R}")
        else:
            for log in logs:
                if "CRITICAL" in log or "[CHAOS]" in log:
                    print(f"  {C_RED}{log}{R}")
                elif "ERROR" in log:
                    print(f"  {C_RED}{log}{R}")
                elif "WARN" in log:
                    print(f"  {C_GOLD}{log}{R}")
                else:
                    print(f"  {C_MUTED}{log}{R}")

        print(f"{C_MUTED}{'─' * 52}{R}")
    except Exception as e:
        print_error(f"Failed to fetch logs: {e}")


def cmd_step(args):
    print(f"{C_CYAN}── Manual Step Mode ──────────────────────────────────{R}")
    tool = input("Tool name (e.g. get_db_metrics, restart_service): ").strip()
    if not tool:
        print_warning("No tool entered. Aborting.")
        return

    params = {}
    if tool in ["restart_service", "scale_service"]:
        service = input("Service name (api-service, db-service, cache-service): ").strip()
        if service:
            params["service"] = service

    action_type = "tool_call" if (tool.startswith("get_") or tool == "clear_db_connections") else "system_action"
    payload     = {"action_type": action_type, "tool": tool, "params": params}

    try:
        res = requests.post(f"{args.host}/api/agent/step", json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()

        step_data = _api_response_to_step_data(data, step_num=1)
        format_step(step_data, mode=args.view)

        if data.get("done"):
            print_success("Incident resolved!")

    except Exception as e:
        print_error(f"Step failed: {e}")


def cmd_explain(args):
    try:
        res = requests.get(f"{args.host}/api/state", timeout=5)
        res.raise_for_status()
        data = res.json()

        if not data.get("initialized"):
            print_warning("No active or completed incident to explain.")
            return

        _sep = lambda c='─', w=52: print(f"{C_MUTED}{c * w}{R}")

        _sep('═')
        print(f"{C_CYAN}  POST-MORTEM REPORT{R}")
        _sep('═')

        print(f"\n{C_CYAN}Root Cause:{R}    {data.get('scenario', '?')}")

        req_fixes  = data.get('required_fixes', [])
        diagnosis  = []
        if "scale_db"          in req_fixes: diagnosis.append("DB connections saturated under load")
        if "restart_api"       in req_fixes: diagnosis.append("API thread pool exhausted")
        if "flush_cache"       in req_fixes: diagnosis.append("Cache locks not released")
        if "clear_connections" in req_fixes: diagnosis.append("DB connection pool exhausted")
        if not diagnosis:
            diagnosis.append("System experiencing unexpected load or failure")

        print(f"\n{C_CYAN}Signals Used:{R}")
        for line in diagnosis:
            print(f"  {C_MUTED}• {line}{R}")

        print(f"\n{C_CYAN}Fix Strategy:{R}")
        for i, fix in enumerate(req_fixes, 1):
            print(f"  {i}. {fix.replace('_', ' ').title()}")

        outcome = "System recovered" if data.get('done') else "Incident unresolved"
        o_color = C_GREEN if data.get('done') else C_RED
        print(f"\n{C_CYAN}Outcome:{R}")
        print(f"  {o_color}{outcome} in {data.get('step', 0)} steps.{R}")

        _sep('═')

    except Exception as e:
        print_error(f"Failed to fetch state for explanation: {e}")


def cmd_run(args):
    view = args.view
    print(f"Starting autonomous agent at {args.host}...\n")
    import time

    step_counter = 0

    try:
        with requests.get(f"{args.host}/api/agent/run", stream=True, timeout=60) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                if not line:
                    continue
                decoded = line.decode('utf-8')
                if not decoded.startswith("data: "):
                    continue

                try:
                    data = json.loads(decoded[6:])
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")

                if event_type == "init":
                    if view != "silent":
                        print(f"{C_CYAN}>>> Scenario: {C_TEXT}{data.get('scenario')}{R}  {C_MUTED}({data.get('phase')}){R}")
                        print(f"{C_MUTED}Agent analyzing system state...{R}\n")
                    step_counter = 0
                    time.sleep(0.3)

                elif event_type == "summary":
                    summary = {
                        "scenario":              data.get("scenario", "?"),
                        "status":                data.get("status", "?"),
                        "steps_taken":           data.get("steps_taken"),
                        "signals_used":          data.get("signals_used", "?"),
                        "fixes_applied":         data.get("fixes_applied", []),
                        "final_latency":         data.get("final_latency"),
                        "total_reward":          data.get("total_reward", 0),
                        "failure_reason":        data.get("failure_reason"),
                        "suggested_improvement": data.get("suggested_improvement"),
                    }
                    print_final_summary(summary)

                elif event_type == "done":
                    break

                elif event_type == "error":
                    print_error(f"Agent Error: {data.get('message')}")
                    break

                else:
                    step_counter += 1
                    step_data = _api_response_to_step_data(data, step_num=step_counter)
                    
                    # Narration
                    if view == "clean":
                        action = step_data.action.lower()
                        if "get_" in action or "metrics" in action:
                            print(f"{C_MAG}🔍 Gathering signals...{R}")
                        elif "restart" in action or "scale" in action or "flush" in action:
                            print(f"{C_MAG}⚙️ Applying fix chain...{R}")
                        time.sleep(0.3)

                    format_step(step_data, mode=view)
                    time.sleep(0.2)

    except requests.exceptions.ReadTimeout:
        print_error("Request timed out waiting for agent.")
    except Exception as e:
        print_error(f"Agent run failed: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global VIEW_MODE

    if len(sys.argv) == 1:
        print_welcome_banner()
        print(f"  {C_TEXT}Available commands:{R} run, step, reset, state, explain, logs")
        print(f"  {C_TEXT}Use {C_CYAN}python {os.path.basename(sys.argv[0])} -h{C_TEXT} for more info.{R}\n")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ACRS Command Line Interface")
    parser.add_argument("--host", default=DEFAULT_HOST, help="FastAPI server URL")
    parser.add_argument(
        "--view",
        default="clean",
        choices=["clean", "debug", "silent"],
        help="Output verbosity: clean (default), debug (full LLM trace), silent (summary only)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run",     help="Run full autonomous agent")
    subparsers.add_parser("step",    help="Run step-by-step mode")
    subparsers.add_parser("reset",   help="Reset environment")
    subparsers.add_parser("state",   help="Show current system state")
    subparsers.add_parser("explain", help="Post-mortem explanation")
    subparsers.add_parser("logs",    help="Show recent logs")

    args      = parser.parse_args()
    VIEW_MODE = args.view

    try:
        requests.get(f"{args.host}/api/health", timeout=3)
    except requests.exceptions.RequestException:
        print_error(f"Could not connect to ACRS server at {args.host}.")
        print(f"  Ensure it is running: {C_GOLD}uvicorn auto_sre_env.server:app{R}")
        sys.exit(1)

    dispatch = {
        "run":     cmd_run,
        "step":    cmd_step,
        "reset":   cmd_reset,
        "state":   cmd_state,
        "explain": cmd_explain,
        "logs":    cmd_logs,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
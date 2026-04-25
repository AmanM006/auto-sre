"""
ACRS FastAPI Server — Extended API layer with SSE streaming.

Serves the agent loop, environment state, and the frontend dashboard.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from auto_sre_env.environment import AutoSREEnv
from auto_sre_env.models import Action
import json
import time
import os
import asyncio

app = FastAPI(title="ACRS — Autonomous Cloud Recovery System")

# ── Session State ────────────────────────────────────────────────────────────

env = AutoSREEnv()
session = {
    "initialized": False,
    "history": [],
    "obs": None,
    "done": False,
    "summary": None,
}


def _reset_session():
    global env, session
    env = AutoSREEnv(difficulty="training")
    obs = env.reset()
    session["initialized"] = True
    session["history"] = []
    session["obs"] = obs
    session["done"] = False
    session["summary"] = None
    return obs


def _obs_to_dict(obs):
    return {
        "services": obs.services if isinstance(obs.services, dict) else dict(obs.services),
        "logs": list(obs.logs),
        "latency": obs.latency,
    }


def _state_snapshot():
    """Full state snapshot for frontend."""
    if not session["initialized"]:
        return {"initialized": False}

    obs = session["obs"]
    return {
        "initialized": True,
        "phase": env.system_phase,
        "scenario": env.state.get("name", "Unknown"),
        "done": session["done"],
        "step": env.step_count,
        "latency": obs.latency if obs else 0,
        "services": obs.services if obs else {},
        "logs": list(obs.logs[-20:]) if obs else [],
        "applied_fixes": list(env.state.get("applied_fixes", [])),
        "required_fixes": list(env.state.get("required_fixes", [])),
        "signals": len(env.signals_gathered),
        "total_reward": round(env.episode_tracker.total_reward, 3),
    }


# ── Root Endpoints ───────────────────────────────────────────────────────────

@app.get("/api")
def api_root():
    return {
        "project": "ACRS — Autonomous Cloud Recovery System",
        "status": "running",
        "endpoints": {
            "reset": "POST /api/agent/reset",
            "step": "POST /api/agent/step",
            "run": "GET /api/agent/run (SSE stream)",
            "state": "GET /api/state",
            "logs": "GET /api/logs",
            "history": "GET /api/history",
            "metrics": "GET /api/metrics",
        }
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Agent Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/agent/reset")
def agent_reset():
    obs = _reset_session()
    return {
        "status": "ok",
        "scenario": env.state.get("name", "Unknown"),
        "phase": env.system_phase,
        "observation": _obs_to_dict(obs),
    }


@app.post("/api/agent/step")
async def agent_step(request: Request):
    """Execute a single agent step with provided action."""
    if not session["initialized"]:
        _reset_session()

    body = await request.json()

    action = Action(
        action_type=body.get("action_type", "tool_call"),
        tool=body.get("tool", ""),
        params=body.get("params", {}),
    )

    obs, reward, done, info = env.step(action)
    session["obs"] = obs
    session["done"] = done

    step_record = {
        "step": env.step_count,
        "phase": env.system_phase,
        "tool": body.get("tool", ""),
        "params": body.get("params", {}),
        "action_type": body.get("action_type", ""),
        "reward": reward,
        "total_reward": info.get("total_reward", 0),
        "done": done,
        "observation": _obs_to_dict(obs),
        "result": obs.logs[-1] if obs.logs else "",
    }
    session["history"].append(step_record)

    return step_record


@app.get("/api/agent/run")
async def agent_run():
    """
    Run the full autonomous agent loop via Server-Sent Events.
    Each step is streamed as an SSE event in real-time.
    Uses a queue to bridge the sync generator with async SSE.
    """
    import queue
    import threading
    from agent_loop import run_agent, STEP_DELAY

    step_queue = queue.Queue()

    def _run_in_thread():
        """Run the sync agent loop in a background thread, pushing steps to queue."""
        try:
            for step_data in run_agent(env=env, max_steps=10, delay=0, stream=True, silent=True):
                # Update session
                try:
                    session["obs"] = env._make_observation()
                except Exception:
                    pass
                session["done"] = env.done

                step_data["state"] = _state_snapshot()
                if step_data.get("type") != "summary":
                    session["history"].append(step_data)
                else:
                    session["summary"] = step_data

                step_queue.put(step_data)
        except Exception as e:
            step_queue.put({"type": "error", "message": str(e)})
        finally:
            step_queue.put(None)  # Sentinel: stream is done

    async def event_stream():
        # Send initial state before thread starts
        _reset_session()
        init_data = {
            "type": "init",
            "scenario": env.state.get("name", "Unknown"),
            "phase": env.system_phase,
            "state": _state_snapshot(),
        }
        yield f"data: {json.dumps(init_data)}\n\n"

        # Start agent in background thread
        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        # Stream steps from queue
        while True:
            # Non-blocking poll with async sleep
            try:
                step_data = step_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            if step_data is None:
                # Stream finished
                break

            yield f"data: {json.dumps(step_data, default=str)}\n\n"
            await asyncio.sleep(STEP_DELAY)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ── State / Logs / History / Metrics ─────────────────────────────────────────

@app.get("/api/state")
def get_state():
    return _state_snapshot()


@app.get("/api/logs")
def get_logs():
    if not session["initialized"] or not session["obs"]:
        return {"logs": []}
    return {"logs": list(session["obs"].logs[-20:])}


@app.get("/api/history")
def get_history():
    return {"history": session["history"]}


@app.get("/api/metrics")
def get_metrics():
    if not session["initialized"]:
        return {"latency": 0, "phase": "NORMAL", "services": {}}

    obs = session["obs"]
    services_summary = {}
    if obs:
        for name, info in obs.services.items():
            services_summary[name] = {
                "status": info.get("status", "unknown"),
                "cpu": info.get("cpu", 0),
                "memory": info.get("memory", 0),
            }

    return {
        "latency": obs.latency if obs else 0,
        "phase": env.system_phase,
        "step": env.step_count,
        "signals": len(env.signals_gathered),
        "total_reward": round(env.episode_tracker.total_reward, 3),
        "services": services_summary,
        "applied_fixes": list(env.state.get("applied_fixes", [])) if env.state else [],
        "required_fixes": list(env.state.get("required_fixes", [])) if env.state else [],
    }


# ── Static Files (Frontend) ─────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_frontend():
        index_path = os.path.join(frontend_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
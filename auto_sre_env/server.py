from fastapi import FastAPI
from .environment import AutoSREEnv
from .models import Action
from .runbook import RUNBOOKS
import gradio as gr
from gradio.routes import mount_gradio_app
from app import demo   # your gradio UI
app = FastAPI()

env = AutoSREEnv()


@app.get("/")
def root():
    return {
        "project": "Auto-SRE",
        "status": "running",
        "description": "AI-powered cloud debugging environment",
        "endpoints": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state"
        }
    }
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/runbooks")
def list_runbooks():
    return RUNBOOKS


@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict(),
    }


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.get_state().dict()

mount_gradio_app(app, demo, path="/")
from fastapi import FastAPI
from pydantic import BaseModel

from rlhf.models import Task, Rollout
from rlhf.store import Store
from rlhf.preferences import PreferenceLog, Preference
from rlhf.monitor import quality_report

app = FastAPI(title="RL Training Data Platform")
store = Store()
prefs = PreferenceLog()


class PreferenceIn(BaseModel):
    task_id: str
    rollout_a: str
    rollout_b: str
    annotator: str
    winner: str


@app.post("/v1/preferences")
def submit_preference(body: PreferenceIn) -> dict:
    """Record one human preference judgment."""
    prefs.record(Preference(**body.model_dump()))
    return {"recorded": True, "total": len(prefs.prefs)}


@app.get("/v1/quality")
def quality(task_id: str) -> dict:
    """Report feedback-quality health for one task's rollout pairs."""
    rollouts = [r.rollout_id for r in store.rollouts_for(task_id)]
    pairs = [(a, b) for i, a in enumerate(rollouts) for b in rollouts[i + 1:]]
    return quality_report(prefs, pairs)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

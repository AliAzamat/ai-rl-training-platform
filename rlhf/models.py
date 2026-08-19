from dataclasses import dataclass, field


@dataclass
class Task:
    """A unit of work researchers want an agent to do well."""
    task_id: str
    prompt: str
    rubric: str                      # what a good response should achieve


@dataclass
class Rollout:
    """One agent attempt at a task — the thing humans will judge."""
    rollout_id: str
    task_id: str
    response: str
    model: str                       # which agent/model produced it

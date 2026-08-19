from rlhf.models import Task, Rollout


class Store:
    """In-memory storage for the platform. Production: a real database."""

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.rollouts: dict[str, Rollout] = {}

    def add_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def add_rollout(self, rollout: Rollout) -> None:
        self.rollouts[rollout.rollout_id] = rollout

    def rollouts_for(self, task_id: str) -> list[Rollout]:
        """Every rollout produced for a given task."""
        return [r for r in self.rollouts.values() if r.task_id == task_id]

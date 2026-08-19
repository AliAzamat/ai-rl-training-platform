import time
from rlhf.reward import RewardModel
from rlhf.preferences import PreferenceLog
from rlhf.aggregate import aggregate_pair


def train_once(model: RewardModel, log: PreferenceLog,
               store, pairs: list[tuple[str, str]]) -> int:
    """One pass: aggregate each pair and take a gradient step on the winner."""
    steps = 0
    for a, b in pairs:
        label = aggregate_pair(log, a, b)
        if label is None:
            continue                 # skip ties / empty pairs
        w = store.rollouts[label["winner"]].response
        l = store.rollouts[label["loser"]].response
        model.train_pair(w, l)
        steps += 1
    return steps


def run_forever(model, log, store, pairs_fn, idle_sleep: float = 2.0) -> None:
    """The training worker: periodically retrain on the latest preferences."""
    while True:
        train_once(model, log, store, pairs_fn())
        time.sleep(idle_sleep)

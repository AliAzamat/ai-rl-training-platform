from rlhf.reward import RewardModel
from rlhf.store import Store


def ranking_accuracy(model: RewardModel, store: Store,
                     held_out: list[dict]) -> float:
    """Fraction of held-out labels where the model scores the winner higher.

    Each held-out item is {winner, loser} of rollout ids the model never saw.
    """
    if not held_out:
        return 0.0
    correct = 0
    for item in held_out:
        w = store.rollouts[item["winner"]].response
        l = store.rollouts[item["loser"]].response
        if model.score(w) > model.score(l):
            correct += 1
    return correct / len(held_out)


def split(labels: list[dict], holdout_frac: float = 0.2) -> tuple[list, list]:
    """Split labels into (train, held_out); the eval set is never trained on."""
    cut = int(len(labels) * (1 - holdout_frac))
    return labels[:cut], labels[cut:]

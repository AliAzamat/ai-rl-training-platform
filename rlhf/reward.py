import numpy as np


def features(response: str) -> np.ndarray:
    """A toy feature vector for a rollout. Production: embeddings/model signals."""
    words = response.split()
    return np.array([
        len(words) / 100.0,                          # length signal
        sum(c.isdigit() for c in response) / 50.0,   # specificity (numbers)
        response.count(".") / 10.0,                   # structure (sentences)
        1.0,                                          # bias term
    ])


class RewardModel:
    """A logistic reward model: learns weights so winners outscore losers."""

    def __init__(self, dim: int = 4, lr: float = 0.1) -> None:
        self.w = np.zeros(dim)
        self.lr = lr

    def score(self, response: str) -> float:
        """The scalar reward for a rollout — higher is better."""
        return float(self.w @ features(response))

    def train_pair(self, winner: str, loser: str) -> None:
        """One gradient step pushing the winner's score above the loser's."""
        fw, fl = features(winner), features(loser)
        diff = self.score(winner) - self.score(loser)
        prob_correct = 1.0 / (1.0 + np.exp(-diff))   # sigmoid of the margin
        # Gradient of the pairwise logistic loss: nudge w toward (fw - fl).
        self.w += self.lr * (1.0 - prob_correct) * (fw - fl)
